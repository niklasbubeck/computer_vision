# ss22 4d Panoptic Segmentation

The aim of this project lies in estimating semantic labels and instance ids for each point in a pointcloud sequence, and set up a time dependency, such that instances stay the same over time. 

# Getting Started 
## EfficientLPS Environment 
- need cuda 11, 10.2 doesn't work
- conda env create -n efficientLPS_env --file=environment.yml
- conda activate efficientLPS_env
- conda install -c conda-forge pycocotools
- pip install -r requirements.txt

### build efficientnet
- cd efficientNet
- python setup.py develop

### build efficientlps
- cd ..
- python setup.py develop


- pip install numpy==1.20.3

Run for a validation:
- ./tools/dist_test.sh EfficientLPS_pretrained/config/efficientLPS_multigpu_sample.py EfficientLPS_pretrained/model/model.pth  1 --eval panoptic

## PointPWC Environment 

First of all please make sure you are using conda 11 and have linked it to your path correctly.

To install an appropriate environment please make sure to build a conda environment using the given ../environments/pwc.yaml file:
- conda env create -f ../environments/pwc.yaml

When building pointnet2 you probably will get an error as torch is not installed yet. You can install the correct torch version using: 
- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113. 
- Then build pointnet2 manually by running "python setup.py install" in the pointnet2 folder. After this just install the missing dependencies.



## Train the point_pwc network
With the pwc environment

You can simply do the training by running the point_pwc/trainer.py file. The refinement.py file lets you load a model to continou training from. 

## Generate the scene flow inference dataset 
With the pwc environment

You can generate the scene flow inference dataset by running the inference.py script. This will make inference on the 08 sequence, save all the data and later on evaluate the sequence. 

## Make panoptic inference
Switch to the efficientPS_env environment, cd to the EfficientLPS folder and run:
- ./tools/dist_test.sh EfficientLPS_pretrained/config/efficientLPS_multigpu_sample.py EfficientLPS_pretrained/model/model.pth  1 --eval panoptic

This will create the panoptic predictions in the EfficientLPS/tmpDir folder. 

## Run the IoU matching 
In the efficientPS_env ennvironment 

You can now run the iou_matching.py script to make the iou matching. The data in tmpDir/08/orig will be overwritten accordingly and are then the new predicitons. 

## Visualize results
In the efficientPS_env environment

Rename the new predictions which are yet labeled as "orig" as "predictions" and put them in a folder structure like "dataDit/sequences/08/predictions" 

Then cd to semantic-kitti-api and run: 
- ./visualize.py --sequence 08 --dataset path/to/semanticKitti/dataset --predictions /dataDir -di True --ignore_safety True


