# =============================================================================
# File Name   : dataloader.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from utils import *
import pickle


class TensorDataset(Dataset):
    """PyTorch Dataset class for Tensor basis NN
    Computes Tensor Basis and Invariants of input Tensor and 
    generate input/output samples for feededing to Tensor Basis NN
    Input: Tracefree tensor of dimension 3
    Ouput: Basis and Invariants of Symmetric and AntiSymmetric part 
    of Input that have following properties: 1) trace = 0, 2) is symmetric
    Attributes
    ----------
    params : argparse parameters
    n_dim : int
        dimension of input/output tensors
    n_lam : int
        number of tensor invariants
    n_basis : int
        number of tensor basis
    invariants : PyTorch tensor
        independent invariants of input tensor
        dimension (N, n_basis, n_dim, n_dim)
    basis : PyTorch tensor
        integrity basis formed from sym and anti-sym parts of input tensor
        properties -- 1) symmetric, 2) tracefree
        dimension (N, n_lam)
    output : PyTorch tensor
        output tensor that is to be modelled as fn(input tensor)
        properties -- 1) symmetric, 2) tracefree
        dimension (N, n_dim, n_dim)
    scale : dict1 of {str:  dict2 or PyTorch tensor}
        dict1:
            key: str ('lam', 'basis', 'output')
            val: if normalizing_strategy is 'starndard' or 'minmax':
                    dict2 (dictionary of scaling tensors)
                 else: # 'norm'
                    PyTorch Tensor
                dict2:
                    key: str [('mean', 'std') or ('min', 'max')]
                    val: PyTorch Tensor
    transform : list of PyTorch transforms
    Methods
    -------
    __len__
        returns length of dataset
        usage: len(datset<class TensorDataset>)
    __item__
        dataset iterator to be used by PyTorch DataLoader
    Note
    ----
    Lam is used as an abbrv. for Invariants
    """
    def __init__(self, params, scale=None, transform=[]):
        self.params = params
        self.n_dim, self.n_basis, self.n_lam = params.n_dim, params.n_basis, params.n_lam
        self.normalizing_strategy = self._get_normalizing_strategy()
        self.invariants, self.basis = self._load_input_tensors()
        self.output = self._load_output_tensor()
        self.scale = self._get_scale() if scale is None else scale
        transform.append(Scaler(self.scale, self.normalizing_strategy))
        self.transform = transforms.Compose(transform)

    def _get_normalizing_strategy(self):
        """returns normalizing strategies for input/output
        Returns
        -------
        dict of {str: str}
            key: 'lam', 'basis', 'output'
            vals: normalizing strategy
        """
        return {
            'lam'   : self.params.normalizing_strategy_lam,
            'basis' : self.params.normalizing_strategy_basis,
            'output': self.params.normalizing_strategy_output
        }

    def _load_input_tensors(self):
        """returns invariants and basis of input tensor
        Returns
        -------
        invariants, basis: tuple of (PyTorch tensor, PyTorch tensor)
            dimensions:
                invariants (N, n_lam)
                basis      (N, n_basis, n_dim, n_dim)
        """
        inp_file = os.path.join(self.params.data_dir, self.params.inp_file)
        A = torch.from_numpy(np.load(inp_file).astype(self.params.precision))

        # get list of tuple of basis and invariant dictionaries
        # [(lam_dict, basis_dict), ...] for each input tensor
        tensor_list = [get_basis_invariants(X) for X in A]
        basis_dict = {
            i: torch.stack([X[0][i] for X in tensor_list])
            for i in range(self.n_basis)
        }
        invariant_dict = {
            i: torch.stack([X[1][i] for X in tensor_list])
            for i in range(self.n_lam)
        }
        basis = torch.stack([basis_dict[i] for i in range(self.n_basis)], dim=1)
        invariants = torch.stack([invariant_dict[i] for i in range(self.n_lam)], dim=1)
        return invariants, basis

    def _load_output_tensor(self):
        """returns output tensor
        Returns
        -------
        B : PyTorch tensor
            dimensions (N, n_dim, n_dim)
        """
        out_file = os.path.join(self.params.data_dir, self.params.out_file)
        B = torch.from_numpy(np.load(out_file).astype(self.params.precision))
        return B

    def _get_scale(self):
        """returns scale for inputs (lam, basis) and output
        Returns
        -------
        scale : dict1 of {str: dict2 or PyTorch tensor}
            dict1:
                key: str ('lam', 'basis', 'output')
                val: if normalizing_strategy is 'starndard' or 'minmax':
                        dict2 (dictionary of scaling tensors)
                    else: # 'norm'
                        PyTorch Tensor
                    dict2:
                        key: str [('mean', 'std') or ('min', 'max')]
                        val: PyTorch Tensor
        """
        return {
            x: calculate_scale(y, strategy=self.normalizing_strategy[x])
            for x,y in [('lam', self.invariants), ('basis', self.basis), ('output', self.output)]
        }

    def __len__(self):
        return self.basis.size()[0]

    def __getitem__(self, idx):
        sample = {
            'lam': self.invariants[idx],
            'basis': self.basis[idx], 
            'output': self.output[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

class Scaler(object):
    """
    Transform: Normalizes PyTorch tensors
    Normalizes both input and output sample
    """
    def __init__(self, scale, strategy):
        self.scale = scale
        self.strategy = strategy

    def __call__(self, sample):
        return {
            x: normalize(y, self.scale[x], self.strategy[x])
            for x,y in sample.items()
        }

class TensorDataloader(object):
    """Dataloader class consisting of dataloaders for train/val/test splits 
    Attributes
    ----------
    dataloader_dict : dict of {str: Pytorch Dataloader}
        dictionary of dataloaders 
        operating_mode=train: dataloaders for train, val splits
        operating_mode=load:  
            not resume_training: dataloaders for test split
            resume_training    : dataloaders for train, val splits
    scale : dict1 of {str:  dict2 or PyTorch tensor}
        dict1:
            key: str ('lam', 'basis', 'output')
            val: dict of scaling tensors or PyTorch Tensor if (strategy='norm')
    Methods
    -------
    No public methods
    """
    def __init__(self, params):
        self.params = params
        self.dataloader = {}
        if self.params.operating_mode == 'train':
            dataset = TensorDataset(self.params)
            self.scale = dataset.scale
            with open(os.path.join(self.params.root_path, 'scale.pkl'), 'wb') as handle:
                pickle.dump(self.scale, handle)
        else:
            with open(os.path.join(self.params.root_path, 'scale.pkl'), 'rb') as handle:
                self.scale = pickle.load(handle)
            dataset = TensorDataset(self.params, self.scale)
            self.scale = dataset.scale

        if (self.params.operating_mode == 'load') & (not self.params.resume_training):
            self.dataloader['test'] = self._dataloader(dataset, shuffle=False)
        else:
            train_len = int(0.8*len(dataset))
            val_len = len(dataset) - train_len
            dataset_split = torch.utils.data.random_split (
                dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(self.params.seed)
            )
            self.dataloader['train'] = self._dataloader(dataset_split[0], shuffle=True)
            self.dataloader['val']   = self._dataloader(dataset_split[1], shuffle=True)

    def _dataloader(self, dset, shuffle=False):
        """return PyToch Dataloader"""
        kwargs = {'num_workers': self.params.num_workers, 'pin_memory': self.params.use_cuda}
        dloader = DataLoader (
            dataset = dset, 
            batch_size = self.params.batch_size, 
            shuffle = shuffle,
            **kwargs
        )
        return dloader