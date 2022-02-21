import math

import torch


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.mean(loss)

def masked_mape_loss(y_pred, y_true):
    mask = (y_true > math.exp(-7)).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true) * 100.0 / torch.abs(y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.mean(loss)

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.square(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.mean(loss)

def masked_rmse_loss(y_pred, y_true):
    return torch.sqrt(masked_mse_loss(y_pred, y_true))