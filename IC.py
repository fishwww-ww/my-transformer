import torch
import numpy as np
from scipy.stats import spearmanr

def calculate_ic(model, data_loader):
    model.eval()
    predicted_returns = []
    actual_returns = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # 将输入数据移动到与模型相同的设备上
            batch_x = batch_x.to(next(model.parameters()).device)
            batch_y = batch_y.to(next(model.parameters()).device)
            # 假设 batch_x 是输入特征，模型输出是预测的价格
            predicted_prices = model(batch_x).squeeze().cpu().numpy()
            # 计算预测的每日收益率
            predicted_daily_returns = np.diff(predicted_prices) / predicted_prices[:-1]
            predicted_returns.extend(predicted_daily_returns)

            # 计算实际的每日收益率
            actual_prices = batch_y.squeeze().cpu().numpy()
            actual_daily_returns = np.diff(actual_prices) / actual_prices[:-1]
            actual_returns.extend(actual_daily_returns)

    # 计算 Spearman 相关系数作为 IC 值
    ic_value, _ = spearmanr(predicted_returns, actual_returns)
    return ic_value

# 示例用法
# ic_value = calculate_ic(model, val_loader)
# print(f'Information Coefficient (IC): {ic_value:.4f}')