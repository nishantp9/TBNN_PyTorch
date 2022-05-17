# =============================================================================
# File Name   : utils.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def get_basis_invariants(X):
    BT = lambda x : torch.transpose(x, 1, 2)
    Btrace = lambda x : torch.einsum('aii->a', x).view(-1,1,1)
    size = X.size()
    bsz, n_dim = size[0], size[-1]
    I = torch.eye(n_dim).unsqueeze(0).repeat(bsz,1,1)
    S   = 0.5*(X + BT(X))
    R   = 0.5*(X - BT(X))
    S   = S - 1./n_dim * I * Btrace(S)
    S2  = S.bmm(S)
    R2  = R.bmm(R)
    T = {}; Lam = {}
    T[0] = S
    T[1] = S.bmm(R) - R.bmm(S)
    T[2] = S2 - 1./n_dim * I * Btrace(S2)
    T[3] = R2 - 1./n_dim * I * Btrace(R2)
    T[4] = R.bmm(S2) - S2.bmm(R)
    T[5] = R2.bmm(S) + S.bmm(R2) - 2./n_dim * I * Btrace(S.bmm(R2))
    T[6] = (R.bmm(S)).bmm(R2) - (R2.bmm(S)).bmm(R)
    T[7] = (S.bmm(R)).bmm(S2) - (S2.bmm(R)).bmm(S)
    T[8] = R2.bmm(S2) + S2.bmm(R2) - 2./n_dim * I * Btrace(S2.bmm(R2))
    T[9] = (R.bmm(S2)).bmm(R2) - (R2.bmm(S2)).bmm(R)
    Lam[0] = Btrace(S2)
    Lam[1] = Btrace(R2)
    Lam[2] = Btrace(S2.bmm(S))
    Lam[3] = Btrace(R2.bmm(S))
    Lam[4] = Btrace(R2.bmm(S2))
    return T, Lam

def clamp(X, n_std):
    cap = n_std * X.flatten().std()
    X = X.clamp(min=-cap, max=cap)
    return X

def remove_trace(X):
    n_dim = X.size()[-1]
    X -= torch.einsum('aii->a', X).view(-1,1,1) * torch.eye(n_dim)/n_dim
    return X

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
    elif strategy == 'none':
        return None

def normalize(X, scale, strategy):
    epsilon = 1e-14
    if strategy == 'standard':
        return (X - scale['mean']) / (scale['std']+epsilon)
    elif strategy == 'minmax':
        return (X - scale['min']) / (scale['max']-scale['min']+epsilon)
    elif strategy == 'norm':
        return X / (scale+epsilon)
    elif strategy == 'none':
        return X

def unnormalize(X, scale, strategy):
    epsilon = 1e-14
    if strategy == 'standard':
        return X * (scale['std']+epsilon) + scale['mean']
    elif strategy == 'minmax':
        return X * (scale['max']-scale['min']+epsilon) + scale['min']
    elif strategy == 'norm':
        return X * (scale+epsilon)
    elif strategy == 'none':
        return X

def save_predictions(y_true, y_pred, y_coef, save_dir):
    n_dim = y_true.shape[-1]
    np.save(os.path.join(save_dir, 'y_true'), y_true)
    np.save(os.path.join(save_dir, 'y_pred'), y_pred)
    np.save(os.path.join(save_dir, 'y_coef'), y_coef)

    y_coef_mean = y_coef.mean(axis=0)
    y_coef_std  = y_coef.std(axis=0)
    print('-'*90)
    print('Basis coefficients:')
    for i in range(y_coef.shape[1]):
        print('C_{}: mean = {:+1.3E}'.format(i, y_coef_mean[i]))
    print('-'*90)

    np.save(os.path.join(save_dir, 'y_coef_mean'), y_coef_mean)
    fig = plt.figure(figsize=(8,7))
    gs = fig.add_gridspec(n_dim,n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            fig.add_subplot(gs[i,j])
            plt.plot(y_true[:,i,j], y_true[:,i,j], '.k', label='true[{},{}]'.format(i,j))
            plt.plot(y_true[:,i,j], y_pred[:,i,j], '.r', label='pred[{},{}]'.format(i,j))
            plt.xlabel('true[{},{}]'.format(i,j))
            plt.grid()
            plt.legend()
            plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preds.png'))
    plt.close()

    fig = plt.figure(figsize=(20,12))
    gs = fig.add_gridspec(y_coef.shape[1]//3+1, 3)
    for i in range(y_coef.shape[1]):
        fig.add_subplot(gs[i//3,i%3])
        plt.title('mean = {:+1.3E}, std = {:1.3E}'.format(y_coef_mean[i], y_coef_std[i]))
        plt.hist(y_coef[:,i], density=True, bins=25)
        plt.xlabel('C{}'.format(i))
        plt.grid()
        plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coefs.png'))
    plt.close()