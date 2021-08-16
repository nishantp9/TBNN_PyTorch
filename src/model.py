# =============================================================================
# File Name   : TBNN.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F

class TBNN(nn.Module):
    def __init__(self, params):
        super(TBNN, self).__init__()
        self.n_lam = params.n_lam
        self.n_bases = params.n_bases
        hidden_dim = [100,50,25,50,100]
        layers = [nn.Linear(self.n_lam, hidden_dim[0])]
        for i in range(1,len(hidden_dim)):
            layers.append(nn.ReLU())
            #layers.append(nn.BatchNorm1d(hidden_dim[i-1]))
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], self.n_bases))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_lam, x_bases = x['lam'], x['bases']
        C =  self.net(x_lam)
        out = (C.view(*C.size(),1,1) * x_bases).sum(dim=1)
        return out