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
import argparse

experiment = "Adapted"
data_module_params = {"dataset": "ProcessedNuScenesAndKittiDataset", #ProcessedNuScenesAndKittiDataset, ProcessedNuScenesDataset, ProcessedKittiDataset
                      "data_dir_kitti":"/home/niklas/Documents/datasets/semanticKitti/dataset/",
                      "data_dir_nuscenes":"/media/niklas/Extreme SSD/data/sets/nuscenes",
                      "nuscenes_version":"v1.0-trainval",
                      "kitti_sequences": [7],
                      "nuscenes_sequences": 500,
                      "remove_classes_kitti": ["unlabeled", "road", "parking", "other-ground", "building", "vegetation", "terrain", "other-object"], # only relevant for preproc=adaptive
                      "remove_classes_nuscenes":["noise", "static_object_bicycle_rack", "flat_driveable_surface", "flat_other", "flat_terrain", "static_other", "static_manmade", "static_vegetation", "vehicle_ego"],
                      "split_in_pct": 0.8,
                      "batch_size":3, 
                      "num_workers":8,
                      "preproc": "adaptive", ## linear or adaptive
                      "randomize": False}

model_module_params = {
    "learning_rate": 0.000005,
    }

save_dir = "./"

def train(model_path):
    time_stamp = str(time.time())
    folder_name = "NN-Output" + experiment + "_" + time_stamp 

    data_module = KittiDataModule(data_module_params)
    model = PointConvSceneFlowPWC8192selfglobalPointConv(model_module_params)
    model.load_from_checkpoint(model_path)

    device = get_default_device()
    model = to_device(model, device)

    logger = TensorBoardLogger(save_dir, name='logs', version=folder_name)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(save_dir,"checkpoints",folder_name), filename='PWC-{epoch:04d}-{Overall_Val:.8f}', monitor="Overall_Val", verbose=True, save_last=True, auto_insert_metric_name=True)
    early_stopping = EarlyStopping(monitor="Overall_Val", min_delta=0.0001, patience=50, verbose=True, mode="min", strict=True)
    
    nbr_batches = len(data_module.train_dataloader())
    trainer = Trainer(max_epochs=-1, accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint, early_stopping], log_every_n_steps=nbr_batches, fast_dev_run=False)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            description=__doc__)
    argparser.add_argument(
        '-m', '--model',
        default="pretrained_models/semantic_kitti_forth_back_model/checkpoints/NN-OutputAdapted_1661258818.0553877/PWC-epoch=0003-Overall_Val=513.57562256.ckpt",
        help='Path to the model ckpt')
    args = argparser.parse_args()
    try: 
        train(args.model)
    except: 
        print(traceback.format_exc())
