# =============================================================================
# File Name   : utils.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def get_bases_invariants(X):
    I = torch.eye(3)
    S   = 0.5*(X + X.T)
    R   = 0.5*(X - X.T)
    S2  = S.mm(S)
    R2  = R.mm(R)
    T = {}; Lam = {}
    T[0] = S
    T[1] = S.mm(R) - R.mm(S)
    T[2] = S2 - 1./3. * I * S2.trace()
    T[3] = R2 - 1./3. * I * R2.trace()
    T[4] = R.mm(S2) - S2.mm(R)
    T[5] = R2.mm(S) + S.mm(R2) - 2./3. * I * (S.mm(R2)).trace()
    T[6] = (R.mm(S)).mm(R2) - (R2.mm(S)).mm(R)
    T[7] = (S.mm(R)).mm(S2) - (S2.mm(R)).mm(S)
    T[8] = R2.mm(S2) + S2.mm(R2) - 2./3. * I * (S2.mm(R2)).trace()
    T[9] = (R.mm(S2)).mm(R2) - (R2.mm(S2)).mm(R)
    Lam[0] = S2.trace()
    Lam[1] = R2.trace()
    Lam[2] = (S2.mm(S)).trace()
    Lam[3] = (R2.mm(S)).trace()
    Lam[4] = (R2.mm(S2)).trace()
    return T, Lam

def calculate_scale(X, strategy):
    if strategy == 'standard':
        return {
            'mean': X.mean(dim=0),
            'std':  X.std(dim=0),
        }
    elif strategy == 'minmax':
        return {
            'min': X.min(dim=0)[0],
            'max': X.max(dim=0)[0],
        }
    elif strategy == 'norm':
        shp = X.size()
        if len(shp) == 4:
            norm = torch.norm(X.view(*shp[:-2],-1), dim=-1).mean(dim=0).view(shp[1],1,1)
        elif len(shp) == 3:
            norm = torch.norm(X.view(shp[0],-1), dim=-1).mean(dim=0)
        else:
            norm = torch.norm(X.view(shp[0],-1), dim=0)
        return norm
    
def normalize(X, scale, strategy):
    epsilon = 1e-8
    if strategy == 'standard':
        return (X - scale['mean']) / (scale['std']+epsilon)
    elif strategy == 'minmax':
        return (X - scale['min']) / (scale['max']-scale['min']+epsilon)
    elif strategy == 'norm':
        return X / (scale+epsilon)

def unnormalize(X, scale, strategy):
    epsilon = 1e-8
    if strategy == 'standard':
        return X * (scale['std']+epsilon) + scale['mean']
    elif strategy == 'minmax':
        return X * (scale['max']-scale['min']+epsilon) + scale['min']
    elif strategy == 'norm':
        return X * (scale+epsilon)

def save_predictions(y_true, y_pred, save_dir):
    np.save(os.path.join(save_dir, 'y_true'), y_true)
    np.save(os.path.join(save_dir, 'y_pred'), y_pred)
    fig = plt.figure(figsize=(8,7))
    gs = fig.add_gridspec(3,3)
    for i in range(3):
        for j in range(3):
            fig.add_subplot(gs[i,j])
            plt.plot(y_true[:,i,j], y_true[:,i,j], '.k', label='true[{},{}]'.format(i,j))
            plt.plot(y_true[:,i,j], y_pred[:,i,j], '.r', label='pred[{},{}]'.format(i,j))
            plt.xlabel('true[{},{}]'.format(i,j))
            plt.grid()
            plt.legend()
            plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preds.png'))
    plt.close()