# TBNN_PyTorch
PyTorch implementation of Tensor Basis Neural Network
* https://doi.org/10.1017/jfm.2016.615
* https://doi.org/10.1103/PhysRevFluids.5.114604

## Conda Installation
```
conda create -n tbnn
conda activate tbnn
conda install numpy
conda install matplotlib
conda install scipy
conda update --all
```
##### LINUX
`conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 -c pytorch`
##### MAC
`conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch`


## Dataset
http://turbulence.pha.jhu.edu/Forced_isotropic_turbulence.aspx
### Input data:
* Velocity Gradient Tensor (VGT) for incompressible flow
* Dimension: `N x 3 x 3`
* Data directory: `data/train/` & `data/test/`
* File format `.npy`
* Input tensors are expected to be traceless (Incompressible VGT)

### Output data / true labels:
* Any output tensor that is modelled as fn(VGT) and has following properties:
  * `trace = 0`
  * is symmetric 
* Dimension: `N x 3 x 3`
* File format `.npy`
* Data directory: `data/train/` & `data/test/`

## Run instructions
### Template
`python3 main.py --data_dir data/train/ --inp_file ## --out_file ## --normalization_strategy ## --batch_size ## --epochs ## --save_interval ##`

### Example
* generate random sample dataset (unphysical & non-turbulence, for illustration purposes only): 
  * `python3 data/sample.py`

`python3 main.py --data_dir data/train/ --inp_file traceless_input.npy --out_file traceless_sym_output.npy --normalization_strategy standard --batch_size 32 --epochs 501 --save_interval 20`

## Save predictions for held-out test set
`python3 main.py --data_dir data/test/ --inp_file ## --out_file ## --normalization_strategy ## --ckpt_timestamp ## --ckpt best`

### NOTE:
* Check `src/args.py` for default arguments
* Training curves are generated after certain # epochs (defined as `--save_interval` in `src/args.py`)
* Training curves are saved in `runs/NN/'timestamp/loss.png'`
* Model checkpoints and results are saved in `runs/NN/'timestamp'/checkpoint` & `runs/NN/'timestamp'/'split'`
