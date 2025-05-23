import torch
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MyTransformer_softmax import MyTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sharp import calculate_sharpe_ratio
from IC import calculate_ic
from symbol import handle_symbol
import random
from metrics import calculate_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    set_seed(42)

    pvo_file_path = "E:/Code/Quantitative Trading/data/ETTh1.csv"
    df = pd.read_csv(pvo_file_path)
    
    # 定义特征维度和目标列索引
    FEATURE_START_COL = 2  # 特征开始列（PVO）
    FEATURE_END_COL = 17    # 特征结束列（returns）
    TARGET_COL = 14         # 目标列索引（returns）
    SEQ_LENGTH = 60        # 序列长度
    BATCH_SIZE = 32        # 批次大小
    INPUT_DIM = FEATURE_END_COL - FEATURE_START_COL  # 输入特征维度
    
    # 分别获取特征数据和目标数据
    features = df.iloc[:, FEATURE_START_COL:FEATURE_END_COL].values  # 前3列作为特征
    # print(features)
    targets = df.iloc[:, TARGET_COL].values.reshape(-1, 1)  # 第4列作为目标
    # print(targets)
    
    # 修改目标值为类别标签（假设我们要预测涨跌二分类）
    # 将连续收益率转换为类别标签 (1表示上涨，0表示下跌)
    targets = (df.iloc[:, TARGET_COL].values > 0).astype(int)  # 转换为0/1标签
    targets = targets.reshape(-1, 1)  # 保持形状一致

    # 修改数据集类
    class MultiFeatureDataset(Dataset):
        def __init__(self, features, targets, seq_length=SEQ_LENGTH):
            self.features = torch.FloatTensor(features)
            self.targets = torch.LongTensor(targets)  # 注意改为LongTensor
            self.seq_length = seq_length
            
        def __len__(self):
            return len(self.features) - self.seq_length
        
        def __getitem__(self, idx):
            x = self.features[idx:idx+self.seq_length]  # 输入序列
            y = self.targets[idx+self.seq_length]       # 预测目标(类别)
            return x, y

    # 创建数据集时传入targets
    dataset = MultiFeatureDataset(features, targets, SEQ_LENGTH)
    train_size = int(0.8 * len(dataset))
    # 随机取数据,测试集是从整个数据集中随机抽取的20%的数据
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # 按时间顺序划分数据集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 修改模型初始化，启用softmax分类功能
    model = MyTransformer(
        input_dim=INPUT_DIM,
        use_softmax=True  # 启用softmax输出
    ).to(device)
    # criterion = torch.nn.MSELoss()
    # 修改损失函数为交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    # 训练模型
    epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # 验证模型
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item()
                _, predicted = torch.max(output.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total

        # 打印训练信息
        print(f'Epoch {epoch+1}/{epochs}, '
            f'Train Loss: {train_loss/len(train_loader):.4f}, '
            f'Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss/len(val_loader):.4f}, '
            f'Val Acc: {val_acc:.2f}%')
        
    #     val_loss = val_loss / len(val_loader)
    #     train_loss = train_loss / len(train_loader)
        
    #     # 更新学习率
    #     scheduler.step(val_loss)
        
    #     # 早停检查
    #     early_stopping(val_loss)
    #     if early_stopping.early_stop:
    #         # print(f"Early stopping triggered at epoch {epoch+1}")
    #         break
        
    #     # 保存最佳模型
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_model_state = model.state_dict()
        
    #     # 计算并打印当前epoch的MSE
    #     mse = calculate_metrics(model, val_loader, 'mse')
    #     print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}')

    #  # 加载最佳模型
    # model.load_state_dict(best_model_state)
    
    # 进行预测并计算分类指标
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(batch_y.numpy())

    # 计算分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n分类报告:")
    print(classification_report(actuals, predictions))
    print("\n混淆矩阵:")
    print(confusion_matrix(actuals, predictions))

    # 可视化预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:100], 'o-', label='Actual')
    plt.plot(predictions[:100], 'x--', label='Predicted')
    plt.title('分类预测结果对比 (前100个样本)')
    plt.xlabel('样本索引')
    plt.ylabel('类别')
    plt.legend()
    plt.show()

def train_with_returns_only():
    set_seed(42)

    pvo_file_path = "E:/Code/Quantitative Trading/data/多只股票多个时序特征时序特征.csv"
    df = pd.read_csv(pvo_file_path)
    
    # 定义参数
    RETURNS_COL = 4  # returns列索引
    SEQ_LENGTH = 60  # 序列长度
    BATCH_SIZE = 32  # 批次大小
    
    # 只获取returns数据
    returns_data = df.iloc[:, RETURNS_COL].values.reshape(-1, 1)
    
    # 自定义数据集类
    class ReturnsDataset(Dataset):
        def __init__(self, data, seq_length=SEQ_LENGTH):
            self.data = torch.FloatTensor(data)
            self.seq_length = seq_length
            
        def __len__(self):
            return len(self.data) - self.seq_length
        
        def __getitem__(self, idx):
            x = self.data[idx:idx+self.seq_length]  # 输入序列
            y = self.data[idx+self.seq_length]      # 预测目标
            return x, y

    # 创建数据集和数据加载器
    dataset = ReturnsDataset(returns_data, SEQ_LENGTH)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyTransformer(input_dim=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    # 训练模型
    epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            output = output.squeeze()
            batch_y = batch_y.squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                output = output.squeeze()
                batch_y = batch_y.squeeze()
                val_loss += criterion(output, batch_y).item()
        
        val_loss = val_loss / len(val_loader)
        train_loss = train_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        # 计算并打印当前epoch的MSE
        mse = calculate_metrics(model, val_loader, 'mse')
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 进行预测
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    # 打印预测结果
    print("\n预测结果：")
    print("预测值\t实际值")
    print("-" * 20)
    for i in range(len(predictions)):
        print(f"{float(predictions[i]):.4f}\t{float(actuals[i]):.4f}")

    # 创建时间索引
    time_steps = range(len(predictions))
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制实际值和预测值
    plt.plot(time_steps, actuals, label='实际值', color='blue')
    plt.plot(time_steps, predictions, label='预测值', color='red', linestyle='--')
    
    # 添加标题和标签
    plt.title('预测值与实际值对比')
    plt.xlabel('时间步')
    plt.ylabel('收益率')
    plt.legend()
    
    # 添加网格
    plt.grid(True)
    
    # 显示图形
    plt.show()

def compare_methods():
    print("=== 使用多特征训练（PVO, SSE, LIQ预测returns）===")
    main()  # 运行原来的多特征训练方法
    
    print("\n=== 使用单特征训练（returns预测returns）===")
    train_with_returns_only()  # 运行新的单特征训练方法

if __name__ == "__main__":
    # compare_methods()
    # train_with_returns_only()
    main()