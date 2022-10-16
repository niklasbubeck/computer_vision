# Pointcloud Sampler

## Overview

This small package contains the pointcloud sampler. It's responsible to provide different, configurable sampling strategies to influence the pointcloud. It holds both, downsampling as well as upsampling.

The algorithm is based on the following assumptions:
 - A pointcloud has a label for each point in it
 - A Voxel aims to estimate a representative value for all points within the voxel 
 - It should provide adaptive sampling corresponding to different labels 
 - Sampling strategy should be configured to suit best 8192 points


## Installation

#### External Dependencies

All external dependencies can be installed using pip or conda.

- [torch]: library providing deep learning utils
- [open3D]: library providing visualization and 3D data processing



## Usage

An example how to use it is delivered in the \__main__() method of the file.


## Static Config Files

All static config files used by the sampler are located in ./config

* **semantic-kitti.yaml:** Is the file originally delivered by the semanticKitti dataset. It contains the mapping from numbers to text, as well as color-mapping, general content distribution, as well as the splits for train/val/test set.
*  **voxel_sizes_kitti.yaml:** This file contains the configuration of voxel sizes. Each label is given its own voxel size for the downsampling part. 
*  **semantic-nuscenes.yaml:** Similarly, this file contains the mapping between text, and numbers of the nuscene labels. 
*  **voxel_sizes_nuscenes.yaml:** Also, this class holds the corresponding voxel size to a label.

## TODOS
* One could introduce the upsampling for the intensity values. 
* Include other sampling strategies/ground-removals etc... 

## Known Problems 
* KD-Tree is not a good method if one only wants to know the shortest distance (Fixed)
 
[open3D]: http://www.open3d.org/
[torch]: https://pytorch.org/
