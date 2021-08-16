# =============================================================================
# File Name   : ops.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import torch
import numpy as np
import random
import os
from datetime import datetime
import time

def seed(params):
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

def get_log_paths(run, ckpt_timestamp=None):
    init_time = ckpt_timestamp
    if init_time is None:
        time = datetime.now()
        init_time = time.strftime('%Y_%m_%d_%H_%M')
    root_path = os.path.join('runs', run, init_time)
    log_path = os.path.join(root_path, 'checkpoints')
    if ckpt_timestamp is None:
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
    else:
        if os.path.isdir(root_path) == False:
            raise Exception("Invalid ckpt_timestamp: {}".format(ckpt_timestamp)) 
    return root_path, log_path

def tic(message=''):
    """Returns current time for timing a task
    Parameters
    ----------
    message : str
        message to be printed
    Returns
    -------
        time.time()
            time at which task is initiated
    """
    print(message)
    return time.time()

def toc(t1, message=''):
    """Returns time taken for execution of a task
    Parameters
    ----------
    t1 : time.time()
        time at which task was initiated
    message : str
        message to be printed
    Returns
    -------
        time.time()
            time taken for execution of task initiated at t1
    """
    print(message + ' | time taken =  {:1.4f}s'.format(time.time()-t1))
    print('-'*90)