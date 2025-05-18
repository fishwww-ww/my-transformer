import torch
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PVOTransformer import PVOTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sharp import calculate_sharpe_ratio
from IC import calculate_ic


def main():
    pvo_file_path = "E:/Code/Quantitative Trading/data/多只股票多个时序特征时序特征.csv"
    # 读取CSV文件
    df = pd.read_csv(pvo_file_path)

    # 假设第二列和第五列是我们需要的特征
    pvo = df.iloc[:, 1].values.reshape(-1, 1) # pvo
    sse = df.iloc[:, 2].values.reshape(-1, 1) # sse
    returns = df.iloc[:, 4].values.reshape(-1, 1) # return

    # 合并数据
    combined_data = np.hstack((pvo, returns))

    # 数据标准化
    scaler = StandardScaler()
    combined_data = scaler.fit_transform(combined_data)

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
    dataset = TimeSeriesDataset(combined_data, seq_length)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PVOTransformer(input_dim=2).to(device)  # 更新输入维度为2
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
    ic_value = calculate_ic(model, val_loader)
    print(f'Information Coefficient (IC): {ic_value:.4f}')

if __name__ == "__main__":
    main()


