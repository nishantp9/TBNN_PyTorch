# =============================================================================
# File Name   : args.py
# Created By  : Nishant Parashar
# Created Date: August 14 2021
# =============================================================================

import argparse

def parameters():
    parser = argparse.ArgumentParser(description="Arguments for TBNN")

    parser.add_argument('--seed', type=int, default=1234,
                        help="seeding things")
    parser.add_argument('--operating_mode', type=str, default='train',
                        choices=['train','load'],
                        help='Train model or load pre-trained model')

    # whether to resume training and learning rate from specified checkpoint
    parser.add_argument('--resume_training',    dest='resume_training', action='store_true')
    parser.add_argument('--no-resume_training', dest='resume_training', action='store_false')
    parser.set_defaults(resume_training=False)
    parser.add_argument('--use_ckpt_lr',    dest='use_ckpt_lr', action='store_true')
    parser.add_argument('--no-use_ckpt_lr', dest='use_ckpt_lr', action='store_false')
    parser.set_defaults(use_ckpt_lr=True)

    parser.add_argument('--save_splits', nargs="+", default=['val'],# 'train'],
                        help="Splits for which predictions to be saved")
    parser.add_argument('--n_dim', type=int, default=3, choices=[2, 3],
                        help='dimension of input/output tensors (2D or 3D)')
    parser.add_argument('--n_basis', type=int, default=10,
                        help='number of integrated basis tensors')
    parser.add_argument('--n_lam', type=int, default=5,
                        help='number of invariants')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/train', #'data/test'
                        help='path of data directory')
    parser.add_argument('--inp_file', type=str, default='traceless_input.npy',
                        help='input file name')
    parser.add_argument('--out_file', type=str, default='traceless_sym_output.npy',
                        help='output file name')

    parser.add_argument('--clamp_input',    dest='clamp_input', action='store_true')
    parser.add_argument('--no-clamp_input', dest='clamp_input', action='store_false')
    parser.set_defaults(clamp_input=False)
    parser.add_argument('--clamp_output',    dest='clamp_output', action='store_true')
    parser.add_argument('--no-clamp_output', dest='clamp_output', action='store_false')
    parser.set_defaults(clamp_output=False)
    parser.add_argument('--clamp_std', type=int, default=3,
                        help='number of std deviations to be clamped')

    parser.add_argument('--normalizing_strategy_basis', type=str, default='norm',
                        choices=['standard', 'minmax', 'norm', 'none'],
                        help='normalizing strategy for tensor basis')
    parser.add_argument('--normalizing_strategy_lam', type=str, default='standard',
                        choices=['standard', 'minmax', 'norm', 'none'],
                        help='normalizing strategy for tensor invariants')
    parser.add_argument('--normalizing_strategy_output', type=str, default='norm',
                        choices=['standard', 'minmax', 'norm', 'none'],
                        help='normalizing strategy for output tensor')

    parser.add_argument('--precision', type=str, default='float32',
                        help="precision of all tensors and model parameters")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size for training")
    parser.add_argument('--use_cuda',    dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)

    parser.add_argument('--device', type=str, default='cpu',
                        help="device name -- 'cpu' / 'cuda:{}' ")

    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of parallel workers")

    parser.add_argument('--run', type=str, default='NN', help="name for the current run")

    parser.add_argument('--ckpt_timestamp', type=str, default=None,
                        help="timestamp of model to be loaded")
    parser.add_argument('--ckpt', type=str, default='best',
                        help="which checkpoint to load", choices=['best', 'latest', 'epoch'])
    parser.add_argument('--ckpt_epoch', type=int, default=None, help="ckpt epoch to load")

    # Learning rate arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="learning rate for training")
    parser.add_argument('--lr_schedule_mode', type=str, default='min',
                        help="mode for lr scheduler. min | max")
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help="factor by which to reduce the learning rate")
    parser.add_argument('--lr_cooldown', type=int, default=5,
                        help="number of epochs for lr reduction")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="minimun lr for learning rate scheduler")
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help="#epochs to wait for early stopping")

    # Training loop arguments
    parser.add_argument('--epochs', type=int, default=501,
                        help="number of training epochs")

    parser.add_argument('--schedule_lr', dest='schedule_lr', action='store_true',
                        help="whether to schedule decay of learning rate")
    parser.add_argument('--no-schedule_lr', dest='schedule_lr', action='store_true')
    parser.set_defaults(schedule_lr=True)

    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'mae', 'smooth_mae'],
                        help="type of loss to be used for training")

    # Logging specific arguments
    parser.add_argument('--save_interval', type=int, default=10,
                        help="running test loop after every test_interval epochs")

    # Model Arguments
    parser.add_argument('--hidden_layer_dims', nargs="+", default=[100,100,100,100],
                        help="dimension of hidden layers of neural network")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="dropout rate")

    opt = parser.parse_args()
    return opt