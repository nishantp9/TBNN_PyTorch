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
    Computes Tensor Bases and Invariants of input Tensor and 
    generate input/output samples for feededing to Tensor Basis NN
    Input: Tracefree tensor of dimension 3
    Ouput: Bases and Invariants of Symmetric and AntiSymmetric part 
    of Input that have following properties: 1) trace = 0, 2) is symmetric
    Attributes
    ----------
    params : argparse parameters
    n_lam : int
        number of tensor invariants
    n_bases : int
        number of tensor bases
    invariants : PyTorch tensor
        independent invariants of input tensor
        dimension (N, n_bases, 3, 3)
    bases : PyTorch tensor
        integrity bases formed from sym and anti-sym parts of input tensor
        properties -- 1) symmetric, 2) tracefree
        dimension (N, n_lam)
    output : PyTorch tensor
        output tensor that is to be modelled as fn(input tensor)
        properties -- 1) symmetric, 2) tracefree
        dimension (N, 3, 3)
    scale : dict1 of {str:  dict2 or PyTorch tensor}
        dict1:
            key: str ('lam', 'bases', 'output')
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
        self.n_bases, self.n_lam = params.n_bases, params.n_lam
        self.invariants, self.bases = self._load_input_tensors()
        self.output = self._load_output_tensor()
        self.scale = self._get_scale() if scale is None else scale
        transform.append(Scaler(self.scale, self.params.normalizing_strategy))
        self.transform = transforms.Compose(transform)
    
    def _load_input_tensors(self):
        """returns invariants and bases of input tensor
        Returns
        -------
        invariants, bases: tuple of (PyTorch tensor, PyTorch tensor)
            dimensions:
                invariants (N, n_lam)
                bases      (N, n_bases, 3, 3)
        """
        inp_file = os.path.join(self.params.data_dir, self.params.inp_file)
        A = torch.from_numpy(np.load(inp_file).astype(self.params.precision))

        # get list of tuple of bases and invariant dictionaries
        # [(lam_dict, bases_dict), ...] for each input tensor
        tensor_list = [get_bases_invariants(X) for X in A]
        bases_dict = {
            i: torch.stack([X[0][i] for X in tensor_list])
            for i in range(self.n_bases)
        }
        invariant_dict = {
            i: torch.stack([X[1][i] for X in tensor_list])
            for i in range(self.n_lam)
        }
        bases = torch.stack([bases_dict[i] for i in range(self.n_bases)], dim=1)
        invariants = torch.stack([invariant_dict[i] for i in range(self.n_lam)], dim=1)
        return invariants, bases

    def _load_output_tensor(self):
        """returns output tensor
        Returns
        -------
        B : PyTorch tensor
            dimensions (N, 3, 3)
        """
        out_file = os.path.join(self.params.data_dir, self.params.out_file)
        B = torch.from_numpy(np.load(out_file).astype(self.params.precision))
        return B

    def _get_scale(self):
        """returns scale for inputs (lam, bases) and output
        Returns
        -------
        scale : dict1 of {str: dict2 or PyTorch tensor}
            dict1:
                key: str ('lam', 'bases', 'output')
                val: if normalizing_strategy is 'starndard' or 'minmax':
                        dict2 (dictionary of scaling tensors)
                    else: # 'norm'
                        PyTorch Tensor
                    dict2:
                        key: str [('mean', 'std') or ('min', 'max')]
                        val: PyTorch Tensor
        """
        return {
            x: calculate_scale(y, strategy=self.params.normalizing_strategy)
            for x,y in [('lam', self.invariants), ('bases', self.bases), ('output', self.output)]
        }

    def __len__(self):
        return self.bases.size()[0]

    def __getitem__(self, idx):
        sample = {
            'lam': self.invariants[idx],
            'bases': self.bases[idx], 
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
            x: normalize(y, self.scale[x], self.strategy)
            for x,y in sample.items()
        }

class TensorDataloader(object):
    """Dataloader class consisting of dataloaders for train/val/test splits 
    Attributes
    ----------
    dataloader_dict : dict of {str: Pytorch Dataloader}
        dictionary of dataloaders 
        operating_mode=train: dataloaders for train, val splits
        operating_mode=load:  dataloaders for test split
    scale : dict1 of {str:  dict2 or PyTorch tensor}
        dict1:
            key: str ('lam', 'bases', 'output')
            val: dict of scaling tensors or PyTorch Tensor if (strategy='norm')
    Methods
    -------
    No public methods
    """
    def __init__(self, params):
        self.params = params
        if self.params.operating_mode == 'train':
            dataset = TensorDataset(self.params)
            self.scale = dataset.scale
            with open(os.path.join(self.params.root_path, 'scale.pkl'), 'wb') as handle:
                pickle.dump(self.scale, handle)
            train_len = int(0.8*len(dataset))
            val_len = len(dataset) - train_len
            dataset_split = torch.utils.data.random_split(dataset, [train_len, val_len])
            self.dataloader = {}
            self.dataloader['train'] = self._dataloader(dataset_split[0], shuffle=True)
            self.dataloader['val']   = self._dataloader(dataset_split[1], shuffle=True)
        else:
            with open(os.path.join(self.params.root_path, 'scale.pkl'), 'rb') as handle:
                self.scale = pickle.load(handle)
            dataset = TensorDataset(self.params, self.scale)
            self.scale = dataset.scale
            self.dataloader = {}
            self.dataloader['test'] = self._dataloader(dataset, shuffle=False)

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