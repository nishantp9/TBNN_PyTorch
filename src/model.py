# =============================================================================
# File Name   : TBNN.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch.nn as nn

class TBNN(nn.Module):
    """Tensor Basis Neural Network (TBNN)
    Attributes
    ----------
    params : argparse parameters
    n_lam : int
        number of tensor invariants
    n_bases : int
        number of tensor bases
    net : PyTorch NN model
        input (lam) --> net --> coefficients
    Methods
    -------
    forward(x)
        return TBNN output after performing linear combination
        of tensor coefficients with the input bases tensors
        x --> net(x) --> coefficients --> dot with bases --> output
        usage: model(x)
    Note
    ----
    All other methods are assumed private and are meant for internal processing only
    """
    def __init__(self, params):
        super(TBNN, self).__init__()
        self.n_lam = params.n_lam
        self.n_bases = params.n_bases
        hidden_dim = params.hidden_layer_dims

        layers = [nn.Linear(self.n_lam, hidden_dim[0])]
        for i in range(1,len(hidden_dim)):
            #layers.append(nn.BatchNorm1d(hidden_dim[i-1]))
            layers.append(nn.Dropout(params.dropout))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        layers.append(nn.ReLU())
        # last layer of NN outputs coefficients
        layers.append(nn.Linear(hidden_dim[-1], self.n_bases))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """forward pass through TBNN
        Parameter
        ---------
        x : PyTorch Tensor
            input to TBNN
        Returns
        -------
        out : PyTorch Tensor
            output of the TBNN network after performing
            linear combination of coeffs. (output of self.net(x))
            with tensor bases
        """
        x_lam, x_bases = x['lam'], x['bases']

        # forward pass through NN to get output coefficients
        C =  self.net(x_lam)

        # Linear combination of x['bases'] with coefficients
        out = (C.view(*C.size(),1,1) * x_bases).sum(dim=1)
        return out