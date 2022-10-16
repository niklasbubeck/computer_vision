# PointPWC Scene Flow

## Overview

This package contains the PointPWC network to estimate the scene flow between two given pointclouds. You can find different models with all their logs and configs in the folder "pretrained_models". Please make sure to check out the inference.py on how to use them.  

The network is based on the following assumptions:
 - A pointcloud was sampled before and has a size of 8192 points
 - When making inference, you should use the same sampling strategy as when you trained
 - The amout of weights was reduced from 7millions to 3.3millions
 


## Installation

First of all please make sure you are using conda 11 and have linked it to your path correctly.

To install an appropriate environment please make sure to build a conda environment using the given ../environments/pwc.yaml file. When building pointnet2 you probably will get an error as torch is not installed yet. You can install the correct torch version using "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113". Then build pointnet2 manually by running "python setup.py install" in the pointnet2 folder. After this just install the missing dependencies.

### External Dependencies

All external dependencies can be installed using pip or conda.

- [torch]: library providing deep learning utils
- [open3D]: library providing visualization and 3D data processing
- [pytorch-lightning]: library providing deep learning utils


## Usage
An example how to use it is delivered in the inference.py file.


## TODOS
* One could introduce the upsampling for the intensity values. 
* Include other sampling strategies/ground-removals etc... 

## Known Problems 

 
[open3D]: http://www.open3d.org/
[torch]: https://pytorch.org/
[pytorch-lightning]: https://www.pytorchlightning.ai/