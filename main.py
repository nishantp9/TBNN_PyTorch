# =============================================================================
# File Name   : main.py
# Created By  : Nishant Parashar
# Created Date: Mon August 14 2021
# =============================================================================

import os
import sys
sys.path.insert(0, 'src')
import torch
from args import parameters
from trainer import Trainer
from ops import seed

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    params = parameters()
    # Fix Random Seeding
    seed(params)

    if params.use_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')

    trainer = Trainer(params)

    if (params.operating_mode == 'train') | params.resume_training:
        trainer.fit()

    for split in params.save_splits:
        trainer.save_predictions(split)