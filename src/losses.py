# =============================================================================
# File Name   : losses.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch.nn as nn
import torch

def loss_function(y_true, y_pred, loss_type, S=None):
    if loss_type == 'smooth_mae':
        loss = smooth_mae_loss(y_true, y_pred)
    elif loss_type == 'mae':
        loss = mae_loss(y_true, y_pred)
    elif loss_type == 'mse':
        loss = mse_loss(y_true, y_pred)
    elif loss_type == 'eig':
        loss = eig_loss(y_true, y_pred)
    elif loss_type == 'alignment':
        loss = alignment_loss(y_true, y_pred, S, pdf=True)
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


class QuantileLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        quantiles = torch.tensor([0,0.001,0.01,0.1,0.3,0.5,
                                    0.7,0.9,0.99,0.999,1])
        self.quantiles = quantiles

    def forward(self, true, pred):
        mse = torch.nn.MSELoss()
        quant = self.quantiles.to(true.device)
        q_true = torch.quantile(true, quant, dim=0)
        q_pred = torch.quantile(pred, quant, dim=0)
        loss = mse(q_true, q_pred)
        return loss

def alignment_loss(y_true, y_pred, S, pdf=False):
    loss = nn.MSELoss()
    bsz, ndim, _ = y_true.size()
    Lt, Vt = torch.linalg.eigh(y_true)
    Lp, Vp = torch.linalg.eigh(y_pred)
    _, Vs = torch.linalg.eigh(S)
    I = torch.eye(ndim, device=y_true.device).unsqueeze(0).repeat(bsz,1,1)

    # alignment pdf loss
    align_t = torch.bmm(torch.transpose(Vt,1,2), Vs)
    align_p = torch.bmm(torch.transpose(Vp,1,2), Vs)
    if pdf:
        qloss = QuantileLoss()
        out1 = qloss(align_t, align_p)
    else:
        out1 = loss(align_t, align_p)

    # Eigenvalue pdf loss
    qloss2 = QuantileLoss()
    out2 = qloss2(Lt, Lp)

    #out2 = loss(Lt, Lp)
    #out2 = loss(torch.norm(y_true, dim=(1,2)), torch.norm(y_pred, dim=(1,2)))
    #out2 = qloss2(torch.norm(y_true, dim=(1,2)), torch.norm(y_pred, dim=(1,2)))
    return out1 + out2

def eig_loss(y_true, y_pred):
    loss = nn.MSELoss()
    bsz, ndim, _ = y_true.size()
    Lt, Vt = torch.linalg.eigh(y_true)
    Lp, Vp = torch.linalg.eigh(y_pred)
    I = torch.eye(ndim, device=y_true.device).unsqueeze(0).repeat(bsz,1,1)

    # Eigenvector loss
    out1 = loss(I, torch.bmm(torch.transpose(Vt,1,2), Vp))

    # Eigenvalue loss
    out2 = loss(Lt, Lp)
    return out1 + out2
