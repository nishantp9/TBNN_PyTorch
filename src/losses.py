# =============================================================================
# File Name   : losses.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch.nn as nn

def loss_function(y_true, y_pred, loss_type):
    if loss_type == 'smooth_mae':
        loss = smooth_mae_loss(y_true, y_pred)
    elif loss_type == 'mae':
        loss = mae_loss(y_true, y_pred)
    elif loss_type == 'mse':
        loss = mse_loss(y_true, y_pred)
    return loss

def mse_loss(y_true, y_pred):
    loss = nn.MSELoss()
    out = loss(y_pred, y_true)
    return out

def mae_loss(y_true, y_pred):
    loss = nn.L1Loss()
    out = loss(y_pred, y_true)
    return out

def smooth_mae_loss(y_true, y_pred):
    loss = nn.SmoothL1Loss()
    out = loss(y_pred, y_true)
    return out