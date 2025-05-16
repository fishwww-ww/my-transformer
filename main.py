import torch
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PVOTransformer import PVOTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_sharpe_ratio(model, data_loader, risk_free_rate=0.01):
    model.eval()
    returns = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            # 将输入数据移动到与模型相同的设备上
            batch_x = batch_x.to(next(model.parameters()).device)
            # 假设 batch_x 是输入特征，模型输出是预测的价格
            predicted_prices = model(batch_x).squeeze().cpu().numpy()
            # 计算每日收益率
            daily_returns = np.diff(predicted_prices) / predicted_prices[:-1]
            returns.extend(daily_returns)

    # 转换为 numpy 数组
    returns = np.array(returns)
    # 计算平均收益率和标准差
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # 计算夏普率
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

def main():
    pvo_file_path = "E:/Code/Quantitative Trading/data/pvo因子值.csv"
    # 读取CSV文件
    df = pd.read_csv(pvo_file_path)

    # 假设第二列是我们需要的特征
    # 从第 20 行开始
    data = df.iloc[19:, 1].values.reshape(-1, 1)

    # 数据标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # 自定义数据集
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, seq_length=10):
            self.data = torch.FloatTensor(data)
            self.seq_length = seq_length
            
        def __len__(self):
            return len(self.data) - self.seq_length
            
        def __getitem__(self, idx):
            x = self.data[idx:idx+self.seq_length]
            y = self.data[idx+self.seq_length, 0]  # 预测目标
            return x, y

    # 创建数据集和数据加载器
    seq_length = 10
    dataset = TimeSeriesDataset(data, seq_length)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PVOTransformer(input_dim=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        
        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item()
        
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.6f}')
    sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
    print(f'Sharpe Ratio: {sharpe_ratio:.4f}')

if __name__ == "__main__":
    main()


