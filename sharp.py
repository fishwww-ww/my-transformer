import torch
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

# sharpe_ratio = calculate_sharpe_ratio(model, val_loader)
# print(f'Sharpe Ratio: {sharpe_ratio:.4f}')