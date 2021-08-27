# =============================================================================
# File Name   : trainer.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import os
import torch
import torch.optim as optim
from losses import loss_function
from ops import get_log_paths
from dataloader import TensorDataloader
from model import TBNN
import matplotlib.pyplot as plt

from utils import unnormalize, save_predictions

class Trainer(object):
    """Trainer class for Tensor Basis Neural Network
    Attributes
    ----------
    params : argparse params
    root_path : str
        root directory for storing checkpoints, scale and predictions
    log_path : str
        directory for storing checkpoints "root_path/checkpoints/"
    tensor_data : Instance of <TensorDataloader> class
        contains dictionary of PyTorch dataloaders for <TensorDataset>
        PyTorch Dataset class
    scale : dict1 of {str:  dict2 or PyTorch tensor}
        dict1:
            key: str ('lam', 'basis', 'output')
            val: dict of scaling tensors or PyTorch Tensor if (strategy='norm')
    model : PyTorch Model (inherits torch.nn.module)
        Tensor Basis Neural Network (TBNN) model
    optimizer : PyTorch optimizer <torch.optim>
        optimizer for temporal model
    Parameters
    ----------
    params : argparse parameters
    Methods
    -------
    fit()
        fits/trains the model on the training data and save the best model
    predict(x)
        returns predictions from current state of the model
    load()
        loads the model from specified checkpoint
    save_predictions(split)
        generates prediction for provided split
    Note
    ----
    All other methods are assumed private and are meant for internal processing only
    """
    def __init__(self, params):
        self.params = params

        # Define logger
        if self.params.operating_mode == 'train':
            self.root_path, self.log_path  = get_log_paths(self.params.run)
        elif self.params.operating_mode == 'load':
            if self.params.ckpt_timestamp == None:
                raise Exception("please provide ckpt_timestamp")
            else:
                self.root_path, self.log_path = get_log_paths(self.params.run, self.params.ckpt_timestamp)
        print('-'*90)
        print('ROOT_PATH = {}'.format(self.root_path))
        print('-'*90)
        # get DataLoaders for train, val, test splits
        self.params.root_path = self.root_path
        self.tensor_data = TensorDataloader(self.params)
        self.scale = self.tensor_data.scale
        self.dataloader = self.tensor_data.dataloader
        self.model = TBNN(self.params)
        if self.params.precision == 'float64':
            self.model = self.model.double()
        self.model = self.model.to(self.params.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    def fit(self):
        """fits/trains PyTorch model on the training data and save checkpoints"""
        print('-'*90)
        print('Training model ...')
        print(self.model)
        print('-'*90)

        best_loss = float('inf')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,\
                            patience=self.params.lr_patience,\
                            factor=self.params.lr_reduce_factor,\
                            verbose=True, mode=self.params.lr_schedule_mode,\
                            cooldown=self.params.lr_cooldown,\
                            min_lr=self.params.min_lr)
        print('-'*90)
        train_loss = self._eval(self.dataloader['train'])
        val_loss   = self._eval(self.dataloader['val'])
        loss_history = {'train': [train_loss], 'val': [val_loss]}
        for epoch in range(self.params.epochs):
            train_loss = self._train()
            val_loss   = self._eval(self.dataloader['val'])

            loss_history['train'].append(train_loss)
            loss_history['val'].append(val_loss)
            print("Epoch: {:02d}, train_loss: {:1.4E}, val_loss: {:1.4E}"\
                    .format(epoch+1, train_loss, val_loss))

            if self.params.schedule_lr:
                lr_scheduler.step(val_loss)

            if epoch % self.params.save_interval == 0:
                self._plot_loss_history(loss_history)
                model_path = os.path.join(self.log_path, "net_{}.pth".format(epoch))
                torch.save(self.model, model_path)
                if val_loss < best_loss:
                    model_path = os.path.join(self.log_path, "net_best_so_far.pth")
                    torch.save(self.model, model_path)
                    best_loss = val_loss

    def _train(self):
        """Training pass for a single epoch
        Returns
        -------
        loss : float
            training loss for single epoch
        """
        self.model.train()
        n_batches = len(self.dataloader['train'])
        epoch_loss = 0.0
        for idx, sample in enumerate(self.dataloader['train']):
            print('batch: {}/{}'.format(idx+1,n_batches), end='\r')

            x_lam = sample['lam']
            x_basis = sample['basis']
            y_true = sample['output']
            x_lam = x_lam.to(self.params.device)
            x_basis = x_basis.to(self.params.device)
            x = {'basis': x_basis, 'lam': x_lam}
            y_true = y_true.to(self.params.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = loss_function(y_true, y_pred, self.params.loss_type)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().cpu().item() 

        epoch_loss /= n_batches
        return epoch_loss

    def _eval(self, dloader):
        """Evaluation pass for a single epoch
        Returns
        -------
        loss : float
            validation/test loss for single epoch
        """
        self.model.eval()
        n_batches = len(dloader)
        epoch_loss = 0.0
        for idx, sample in enumerate(dloader):
            x_lam = sample['lam']
            x_basis = sample['basis']
            y_true = sample['output']
            x_lam = x_lam.to(self.params.device)
            x_basis = x_basis.to(self.params.device)
            x = {'basis': x_basis, 'lam': x_lam}
            y_true = y_true.to(self.params.device)

            y_pred = self.model(x)

            loss = loss_function(y_true, y_pred, self.params.loss_type)
            epoch_loss += loss.detach().cpu().item() 

        epoch_loss /= n_batches
        return epoch_loss

    def predict(self, x):
        """Returns predictions from current state of model
        Parameters
        ----------
        x : Pytorch Tensor
        Returns
        -------
        Pytorch Tensor
            output from the model
        """
        y = self.model(x)
        return y

    def load(self):
        """Loads the best/latest checkpoint from log_path"""
        if self.log_path is None:
            raise Exception("Invalid ckpt_timestamp: no checkpoints exist")
        if self.params.ckpt == 'best':
            checkpoint_path = os.path.join(self.log_path, 'net_best_so_far.pth')
        elif self.params.ckpt == 'latest':
            filename_list = os.listdir(self.log_path)
            filepath_list = [os.path.join(self.log_path, fname) for fname in filename_list]
            checkpoint_path = sorted(filepath_list, key=os.path.getmtime)[-1]
        elif self.params.ckpt_epoch is not None:
            checkpoint_path = os.path.join(self.log_path, 'net_{}.pth'.format(self.params.ckpt_epoch))
        else:
            raise Exception("please provide trainer_params.ckpt_epoch")

        print('-'*90)
        print('Loading model from: <{}>'.format(checkpoint_path))
        self.model = torch.load(checkpoint_path)
        print('-'*90)

    def _plot_loss_history(self, loss):
        """plot training curves"""
        plt.semilogy(loss['train'], '-k', label='train')
        plt.semilogy(loss['val'], '-r', label='val')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.root_path, 'loss.png'))
        plt.close()

    def save_predictions(self, split='test'):
        """Saves predictions and prints metrics
        Generates predictions for provided train/val/test split
        Saved forecast as a netcdf file
        Parameters
        ----------
        split : str
            train/val/test split
        """
        if self.params.operating_mode == 'load':
            split = 'test'
        self.model.eval()
        print('Getting model predictions for {}-set ...'.format(split))
        dataloader = self.dataloader[split]
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []
            out_scale = self.scale['output']
            for idx, sample in enumerate(dataloader):
                x_lam = sample['lam']
                x_basis = sample['basis']
                y_true = sample['output']
                x_lam = x_lam.to(self.params.device)
                x_basis = x_basis.to(self.params.device)
                x = {'basis': x_basis, 'lam': x_lam}
                y_true = y_true.to(self.params.device)
                y_pred = self.predict(x)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
        y_true = torch.cat(y_true_list, dim=0).detach().cpu()
        y_pred = torch.cat(y_pred_list, dim=0).detach().cpu()
        y_true = unnormalize(y_true, out_scale, self.params.normalizing_strategy)
        y_pred = unnormalize(y_pred, out_scale, self.params.normalizing_strategy)
        save_path = os.path.join(self.params.root_path, 'Results', split)
        os.makedirs(save_path, exist_ok=True)
        save_predictions(y_true.numpy(), y_pred.numpy(), save_path)