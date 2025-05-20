import torch
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PVOTransformer import PVOTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sharp import calculate_sharpe_ratio
from IC import calculate_ic
from symbol import handle_symbol
import random
from metrics import calculate_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

def calculate_returns(predictions, actuals):
    """计算预测收益率和实际收益率"""
    pred_returns = np.diff(predictions) / predictions[:-1]
    actual_returns = np.diff(actuals) / actuals[:-1]
    return pred_returns, actual_returns

def predict_returns(model, data_loader, device):
    """使用模型预测收益率"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算收益率
    pred_returns, actual_returns = calculate_returns(predictions, actuals)
    
    return pred_returns, actual_returns

def main():
    set_seed(42)  # 在程序开始时调用

    pvo_file_path = "E:/Code/Quantitative Trading/data/多只股票多个时序特征时序特征.csv"
    # 读取CSV文件
    df = pd.read_csv(pvo_file_path)

    pvo = df.iloc[:, 1].values.reshape(-1, 1) # pvo
    sse = df.iloc[:, 2].values.reshape(-1, 1) # sse
    liq = df.iloc[:, 3].values.reshape(-1, 1) # liq
    returns = df.iloc[:, 4].values.reshape(-1, 1) # return
    date = df.iloc[:, 5].values.reshape(-1, 1) # date
    symbol = df.iloc[:, 6].values.reshape(-1, 1) # symbol

    newData = df.iloc[:,1:7].values
    print(newData)
    
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

    # 存储预测收益率和实际收益率
    predicted_returns_dict = {}
    actual_returns_dict = {}

    # 使用每支股票的pvo训练数据
    for sym, sym_data in data.items():
        sym_data = np.array(sym_data).reshape(-1, 1)

        # 启用数据标准化
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

        # Train model
        epochs = 100
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    val_loss += criterion(output.squeeze(), batch_y).item()
            
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
            print(f'Symbol: {sym}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}')

        # 使用最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 预测收益率
        pred_returns, actual_returns = predict_returns(model, val_loader, device)
        predicted_returns_dict[sym] = pred_returns
        actual_returns_dict[sym] = actual_returns
        
        # 计算并打印收益率统计信息
        mean_pred_return = np.mean(pred_returns)
        std_pred_return = np.std(pred_returns)
        mean_actual_return = np.mean(actual_returns)
        std_actual_return = np.std(actual_returns)
        
        print(f"\nSymbol: {sym} 收益率统计:")
        print(f"预测收益率 - 均值: {mean_pred_return:.4f}, 标准差: {std_pred_return:.4f}")
        print(f"实际收益率 - 均值: {mean_actual_return:.4f}, 标准差: {std_actual_return:.4f}")
        
        # 计算收益率相关性
        correlation = np.corrcoef(pred_returns, actual_returns)[0,1]
        print(f"预测收益率与实际收益率的相关性: {correlation:.4f}\n")

        data_mse[sym] = mse
        sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
        data_sharp[sym] = sharpe_ratio
        rse = calculate_metrics(model, val_loader,'rse')
        data_rse[sym] = rse
        mae = calculate_metrics(model, val_loader,'mae')
        data_mae[sym] = mae

    # 对data_sharp进行排序
    sorted_data_sharp = sorted(data_sharp.items(), key=lambda x: x[1], reverse=True)
    sorted_data_rse = sorted(data_rse.items(), key=lambda x: x[1], reverse=True)
    sorted_data_mae = sorted(data_mae.items(), key=lambda x: x[1], reverse=True)
    sorted_data_mse = sorted(data_mse.items(), key=lambda x: x[1], reverse=True)
    # for sym, sharpe in sorted_data_sharp:
        # print(f'Symbol: {sym}, Sharpe Ratio: {sharpe:.4f}')
    # for sym,mse in sorted_data_mse:
        # print(f'Symbol: {sym}, MSE: {mse:.4f}')

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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

        # Train model
        epochs = 100
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    val_loss += criterion(output.squeeze(), batch_y).item()
            
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
            print(f'Symbol: {sym}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MSE: {mse:.4f}')

        # 使用最佳模型状态
        model.load_state_dict(best_model_state)
        data_returns_mse[sym] = mse

        # Calculate Sharpe ratio for this symbol
        sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
        data_returns_sharp[sym] = sharpe_ratio
        rse = calculate_metrics(model, val_loader,'rse')
        data_returns_rse[sym] = rse
        mae = calculate_metrics(model, val_loader,'mae')
        data_returns_mae[sym] = mae

    # 对data_sharp进行排序
    sorted_data_returns_sharp = sorted(data_returns_sharp.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_rse = sorted(data_returns_rse.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_mae = sorted(data_returns_mae.items(), key=lambda x: x[1], reverse=True)
    sorted_data_returns_mse = sorted(data_returns_mse.items(), key=lambda x: x[1], reverse=True)
    # for sym, sharpe in sorted_data_returns_sharp:
        # print(f'Symbol: {sym}, Returns Sharpe Ratio: {sharpe:.4f}')    
    # for sym,mse in sorted_data_returns_mse:
        # print(f'Symbol: {sym}, Returns MSE: {mse:.4f}')
        
    # 输出symbol,data_sharp中的sharp和data_returns_sharp中的sharp.按照data_sharp中的sharp排序
    # for sym, sharpe in sorted_data_sharp:
        # print(f'Symbol: {sym}, PVO Sharpe Ratio: {data_sharp[sym]:.4f}, Returns Sharpe Ratio: {data_returns_sharp[sym]:.4f}')
        # print(f'Symbol: {sym},RSE: {data_rse[sym]:.4f}, Returns RSE: {data_returns_rse[sym]:.4f}')
        # print(f'Symbol: {sym}, MAE: {data_mae[sym]:.4f}, Returns MAE: {data_returns_mae[sym]:.4f}')
        # print(f'Symbol: {sym}, MSE: {data_mse[sym]:.4f}, Returns MSE: {data_returns_mse[sym]:.4f}')

    # # 输出总体预测效果
    # print("\n总体预测效果:")
    # all_pred_returns = np.concatenate(list(predicted_returns_dict.values()))
    # all_actual_returns = np.concatenate(list(actual_returns_dict.values()))
    
    # overall_correlation = np.corrcoef(all_pred_returns, all_actual_returns)[0,1]
    # print(f"所有股票预测收益率与实际收益率的总体相关性: {overall_correlation:.4f}")
    
    # # 计算预测准确率（预测方向与实际方向一致的比例）
    # direction_accuracy = np.mean(np.sign(all_pred_returns) == np.sign(all_actual_returns))
    # print(f"预测方向准确率: {direction_accuracy:.4f}")

if __name__ == "__main__":
    main()