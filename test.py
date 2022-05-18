# =============================================================================
# File Name   : test.py
# Created By  : Nishant Parashar
# Created Date: Tue May 17 2022
# =============================================================================

import os
import sys
sys.path.insert(0, 'src')
import numpy as np
from inference import TBNN_PH

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    PHESS = TBNN_PH(ckpt='2022_05_17_19_31')
    A = np.load('data/test/traceless_input.npy')
    P = PHESS(A)