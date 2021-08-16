# TBNN_PyTorch
PyTorch implementation of Tensor Basis Neural Network
https://doi.org/10.1017/jfm.2016.615

## Dataset
Input data:
* Velocity Gradient Tensor (VGT) for incompressible flow
* Dimension: N x 3 x 3
* Data directory: data/train/ & data/test/
* File format .npy
* Input tensors are expected to be traceless (Incompressible VGT)

## Output data / true labels:
* Any output tensor that is modelled as fn(VGT) and has following properties:
  * trace = 0
  * is symmetric 
* Dimension: N x 3 x 3
* File format .npy
* Data directory: data/train/ & data/test/

## Run instructions
python3 main.py --data_dir data/train/ --inp_file ## --normalization_strategy ## --batch_size ## --epochs ## --save_interval ##

## Save predictions for held-out test set
python3 main.py --data_dir data/test/ --inp_file ## --normalization_strategy ## --ckpt_timestamp ## --ckpt best

### NOTE:
* Check src/args.py for default arguments
* Model checkpoints and results are saved in runs/NN/
