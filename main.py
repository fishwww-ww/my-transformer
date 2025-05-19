import torch
import torch.utils.data as data
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from PVOTransformer import PVOTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sharp import calculate_sharpe_ratio
from IC import calculate_ic
from symbol import handle_symbol
import random
from metrics import calculate_metrics


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

def main():
    set_seed(42)  # 在程序开始时调用

    pvo_file_path = "E:/Code/Quantitative Trading/data/多只股票多个时序特征时序特征.csv"
    # 读取CSV文件
    df = pd.read_csv(pvo_file_path)

    # 假设第二列和第五列是我们需要的特征
    pvo = df.iloc[:, 1].values.reshape(-1, 1) # pvo
    sse = df.iloc[:, 2].values.reshape(-1, 1) # sse
    liq = df.iloc[:, 3].values.reshape(-1, 1) # liq
    returns = df.iloc[:, 4].values.reshape(-1, 1) # return
    date = df.iloc[:, 5].values.reshape(-1, 1) # date
    symbol = df.iloc[:, 6].values.reshape(-1, 1) # symbol
    
    data = handle_symbol(pvo, symbol) # key: symbol, value: pvo
    data_sharp = {} # key: symbol, value: sharpe ratio
    data_returns = handle_symbol(returns, symbol) # key: symbol, value: returns
    data_returns_sharp = {} # key: symbol, value: sharpe ratio
    data_rse = {} # key: symbol, value: rse
    data_returns_rse = {} # key: symbol, value: returns rse
    data_mae = {} # key: symbol, value: mae
    data_returns_mae = {} # key: symbol, value: returns mae
    data_mse = {} # key: symbol, value: mse
    data_returns_mse = {} # key: symbol, value: returns mse

    # 使用每支股票的pvo训练数据
    for sym, sym_data in data.items():
        # Convert sym_data to a numpy array and reshape if necessary
        sym_data = np.array(sym_data).reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # sym_data = scaler.fit_transform(sym_data)

        # 自定义数据集
        class TimeSeriesDataset(Dataset):
            def __init__(self, data, seq_length=60):
                self.data = torch.FloatTensor(data)
                self.seq_length = seq_length
                
            def __len__(self):
                return len(self.data) - self.seq_length
            
            def __getitem__(self, idx):
                x = self.data[idx:idx+self.seq_length]
                y = self.data[idx+self.seq_length, 0]  # 预测目标
                return x, y

        # Create dataset and dataloader
        seq_length = 60  # 修改为60个交易日
        dataset = TimeSeriesDataset(sym_data, seq_length)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PVOTransformer(input_dim=1).to(device)  # Update input_dim to 1 for single feature
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train model
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
            
            # Validate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    val_loss += criterion(output.squeeze(), batch_y).item()
            
            # print(f'Symbol: {sym}, Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.6f}')

        # Calculate Sharpe ratio for this symbol
        sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
        data_sharp[sym] = sharpe_ratio
        rse = calculate_metrics(model, val_loader,'rse')
        data_rse[sym] = rse
        mae = calculate_metrics(model, val_loader,'mae')
        data_mae[sym] = mae
        mse = calculate_metrics(model, val_loader,'mse')
        data_mse[sym] = mse
        # print(f'Symbol: {sym}, Sharpe Ratio: {sharpe_ratio:.4f}')

    # 对data_sharp进行排序
    sorted_data_sharp = sorted(data_sharp.items(), key=lambda x: x[1], reverse=True)
    sorted_data_rse = sorted(data_rse.items(), key=lambda x: x[1], reverse=True)
    sorted_data_mae = sorted(data_mae.items(), key=lambda x: x[1], reverse=True)
    sorted_data_mse = sorted(data_mse.items(), key=lambda x: x[1], reverse=True)
    # for sym, sharpe in sorted_data_sharp:
        # print(f'Symbol: {sym}, Sharpe Ratio: {sharpe:.4f}')

    # cpoy
    for sym, sym_data in data_returns.items():
        # Convert sym_data to a numpy array and reshape if necessary
        sym_data = np.array(sym_data).reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # sym_data = scaler.fit_transform(sym_data)

        # 自定义数据集
        class TimeSeriesDataset(Dataset):
            def __init__(self, data, seq_length=60):
                self.data = torch.FloatTensor(data)
                self.seq_length = seq_length
                
            def __len__(self):
                return len(self.data) - self.seq_length
            
            def __getitem__(self, idx):
                x = self.data[idx:idx+self.seq_length]
                y = self.data[idx+self.seq_length, 0]  # 预测目标
                return x, y

        # Create dataset and dataloader
        seq_length = 60  # 修改为60个交易日
        dataset = TimeSeriesDataset(sym_data, seq_length)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PVOTransformer(input_dim=1).to(device)  # Update input_dim to 1 for single feature
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train model
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
            
            # Validate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    val_loss += criterion(output.squeeze(), batch_y).item()
            
            # print(f'Symbol: {sym}, Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.6f}')

        # Calculate Sharpe ratio for this symbol
        sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
        data_returns_sharp[sym] = sharpe_ratio
        rse = calculate_metrics(model, val_loader,'rse')
        data_returns_rse[sym] = rse
        mae = calculate_metrics(model, val_loader,'mae')
        data_returns_mae[sym] = mae
        mse = calculate_metrics(model, val_loader,'mse')
        data_returns_mse[sym] = mse
        # print(f'Symbol: {sym}, Sharpe Ratio: {sharpe_ratio:.4f}')

    # 对data_sharp进行排序
    sorted_data_returns_sharp = sorted(data_returns_sharp.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_rse = sorted(data_returns_rse.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_mae = sorted(data_returns_mae.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_mse = sorted(data_returns_mse.items(), key=lambda x: x[1], reverse=True)
    # for sym, sharpe in sorted_data_returns_sharp:
        # print(f'Symbol: {sym}, Returns Sharpe Ratio: {sharpe:.4f}')    
        
    # 输出symbol,data_sharp中的sharp和data_returns_sharp中的sharp.按照data_sharp中的sharp排序
    for sym, sharpe in sorted_data_sharp:
        # print(f'Symbol: {sym}, PVO Sharpe Ratio: {data_sharp[sym]:.4f}, Returns Sharpe Ratio: {data_returns_sharp[sym]:.4f}')
        # print(f'Symbol: {sym},RSE: {data_rse[sym]:.4f}, Returns RSE: {data_returns_rse[sym]:.4f}')
        # print(f'Symbol: {sym}, MAE: {data_mae[sym]:.4f}, Returns MAE: {data_returns_mae[sym]:.4f}')
        print(f'Symbol: {sym}, MSE: {data_mse[sym]:.4f}, Returns MSE: {data_returns_mse[sym]:.4f}')

if __name__ == "__main__":
    main()