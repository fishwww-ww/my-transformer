import numpy as np
import torch

#各种指标
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def calculate_metrics(model, data_loader,target):
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # 将输入数据移动到与模型相同的设备上
            batch_x = batch_x.to(next(model.parameters()).device)
            # 获取模型预测
            predicted = model(batch_x).squeeze().cpu().numpy()
            # 获取真实值
            true = batch_y.squeeze().cpu().numpy()
            
            predictions.extend(predicted)
            true_values.extend(true)

    # 转换为numpy数组
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    match target:
        case 'rse':
            return RSE(predictions, true_values)
        case 'corr':
            return CORR(predictions, true_values)
        case 'mae':
            return MAE(predictions, true_values)
        case 'mse':
            return MSE(predictions, true_values)
        case 'rmse':
            return RMSE(predictions, true_values)
        case 'mape':
            return MAPE(predictions, true_values)
        case 'mspe':
            return MSPE(predictions, true_values)
        case _:
            raise ValueError(f"Invalid target: {target}")
