from pyexpat import model
from data_module import ProcessedKittiDataset, KittiDataModule, get_default_device, to_device
from torchsummary import summary 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pwc_pointconv import PointConvSceneFlowPWC8192selfglobalPointConv
#from original_pwc import PointConvSceneFlowPWC8192selfglobalPointConv
import os
import traceback
import time

experiment = "Kitti_and_NuScenes"
data_module_params = {"dataset": "ProcessedNuScenesAndKittiDataset", #ProcessedNuScenesAndKittiDataset, ProcessedNuScenesDataset, ProcessedKittiDataset
                      "data_dir_kitti":"/media/niklas/Extreme SSD/semanticKitti/dataset",
                      "data_dir_nuscenes":"/media/niklas/Extreme SSD/data/sets/nuscenes",
                      "nuscenes_version":"v1.0-trainval",
                      "kitti_sequences": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], #leave out 8 for validation
                      "nuscenes_sequences": 25000,
                      "remove_classes_kitti": ["outlier", "unlabeled", "road", "parking", "other-ground", "building", "sidewalk", "vegetation", "terrain", "other-object"], # only relevant for preproc=adaptive
                      "remove_classes_nuscenes": ["noise", "static_object_bicycle_rack", "flat_driveable_surface","flat_sidewalk", "flat_other", "flat_terrain", "static_other", "static_manmade", "static_vegetation", "vehicle_ego"],
                      "split_in_pct": 0.8,
                      "batch_size":3, 
                      "num_workers":8,
                      "preproc": "adaptive", ## linear or adaptive
                      "randomize": False}

model_module_params = {
    "learning_rate": 0.000005,
    }

save_dir = "./"

def train():
    time_stamp = str(time.time())
    folder_name = "NN-Output" + experiment + "_" + time_stamp 

    data_module = KittiDataModule(data_module_params)
    model = PointConvSceneFlowPWC8192selfglobalPointConv(model_module_params)

    device = get_default_device()
    model = to_device(model, device)

    logger = TensorBoardLogger(save_dir, name='logs', version=folder_name)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(save_dir,"checkpoints",folder_name), filename='PWC-{epoch:04d}-{Overall_Val:.8f}', monitor="Overall_Val", verbose=True, save_last=True, auto_insert_metric_name=True)
    early_stopping = EarlyStopping(monitor="Overall_Val", min_delta=0.0001, patience=50, verbose=True, mode="min", strict=True)
    
    nbr_batches = len(data_module.train_dataloader())
    trainer = Trainer(max_epochs=-1, accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint, early_stopping], log_every_n_steps=nbr_batches, fast_dev_run=False)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    try: 
        train()
    except: 
        print(traceback.format_exc())
