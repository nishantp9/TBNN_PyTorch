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
    n_basis : int
        number of tensor basis
    net : PyTorch NN model
        input (lam) --> net --> coefficients
    Methods
    -------
    forward(x)
        return TBNN output after performing linear combination
        of tensor coefficients with the input basis tensors
        x --> net(x) --> coefficients --> dot with basis --> output
        usage: 
            model = TBNN(params)
            out = model(x)
    """
    def __init__(self, params):
        super(TBNN, self).__init__()
        self.n_lam = params.n_lam
        self.n_basis = params.n_basis
        hidden_dim = params.hidden_layer_dims

        layers = []
        for dim1, dim2 in zip([self.n_lam]+hidden_dim, hidden_dim):
            layer = nn.Sequential (
                nn.Linear(dim1, dim2),
                # nn.BatchNorm1d(dim2),
                nn.Dropout(params.dropout),
                nn.ReLU(),
            )
            layers.append(layer)

        # last layer outputs the basis coefficients
        layers.append(nn.Linear(hidden_dim[-1], self.n_basis))
        self.net = nn.Sequential(*layers)
        #self.net.apply(init_weights)

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
            with tensor basis
        """
        x_lam, x_basis = x['lam'], x['basis']

        # forward pass through NN to get output coefficients
        C =  self.net(x_lam)

        # Linear combination of x['basis'] with coefficients
        out = (C.view(*C.size(),1,1) * x_basis).sum(dim=1)
        return {'output': out, 'coefficients': C}

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)