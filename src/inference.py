# =============================================================================
# File Name   : inference.py
# Created By  : Nishant Parashar
# Created Date: Tue May 17 2022
# =============================================================================

import torch
from args import parameters
from trainer import Trainer
from utils import *
from ops import seed
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')

def insert_trace(y, A):
    n_dim = A.size()[-1]
    y = remove_trace(y)
    y += torch.einsum('aii->a', -A.bmm(A)).view(-1,1,1) * torch.eye(n_dim)/n_dim
    return y

class TBNN_PH():
    def __init__(self, ckpt):
        self.params = parameters()
        self.params.operating_mode = 'load'
        self.params.ckpt_timestamp = ckpt
        # Fix Random Seeding
        seed(self.params)

        if self.params.use_cuda and torch.cuda.is_available():
            self.params.device = torch.device('cuda:0')
        else:
            self.params.device = torch.device('cpu')
        self.trainer = Trainer(self.params)
        self.trainer.load()
        self.trainer.model.eval()
        self.basis_scale = self.trainer.scale['basis']
        self.lam_scale   = self.trainer.scale['lam']
        self.out_scale   = self.trainer.scale['output']

    def __call__(self, A):
        basis_dict, invariant_dict  = get_basis_invariants(A)
        n_basis = len(basis_dict)
        n_lam   = len(invariant_dict)
        basis = torch.stack([basis_dict[i] for i in range(n_basis)], dim=1)
        invariants = torch.stack([invariant_dict[i].squeeze() for i in range(n_lam)], dim=1)
        x = {
            'basis': normalize(basis, self.basis_scale, self.params.normalizing_strategy_basis),
            'lam'  : normalize(invariants, self.lam_scale, self.params.normalizing_strategy_lam)
            }
        with torch.no_grad():
            y = self.trainer.predict(x)
        y = unnormalize(y, self.out_scale, self.params.normalizing_strategy_output)
        y = insert_trace(y, torch.from_numpy(A))
        y = y.cpu().detach().numpy()
        return y

if __name__ == '__main__':
    PHESS = TBNN_PH(ckpt='2022_05_17_19_31')
    A = torch.from_numpy(np.load('data/test/traceless_input.npy'))
    P = PHESS(A)