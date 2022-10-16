from data_module import CarlaDataModule, get_default_device, to_device
from unet_module import UNet, compute_max_depth
from torchsummary import summary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import traceback
import time
# "/usr/prakt/s0099/niklas-intellisys-ss2022/week4/_out/pickles/"
experiment = "Unnormalized-small-set"
data_module_params = {"data_dir":"/storage/remote/atcremers1/s0099/pickles/stationary-small/", 
                      "split_in_pct": 0.8,
                      "batch_size":4, 
                      "num_workers":4}



model_module_params = {"learning_rate":0.00001}
save_dir = "/usr/prakt/s0099/niklas-intellisys-ss2022/week4"

def train():
    time_stamp = str(time.time())
    out = compute_max_depth(512)
    folder_name = "NN-Output" + time_stamp + "_" + experiment

    carla_dm = CarlaDataModule(data_module_params)
    model_module_params["mean"], model_module_params["std"] = carla_dm.mean, carla_dm.std
    model = UNet(
             model_module_params,
             in_channels=2,
             out_channels=3,
             n_blocks= 3,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)

    device = get_default_device()
    model = to_device(model, device)
    sumry = summary(model, (2, 512, 512))
    print(sumry)

    logger = TensorBoardLogger(save_dir, name='logs', version=folder_name)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(save_dir,"checkpoints",folder_name), filename='U-Net-{epoch:04d}-{MAE_Val:.8f}', monitor="MAE_Val", verbose=True, save_last=True, auto_insert_metric_name=True)
    early_stopping = EarlyStopping(monitor="MAE_Val", min_delta=0.00000001, patience=50, verbose=True, mode="min", strict=True)
    
    nbr_batches = len(carla_dm.train_dataloader())
    trainer = Trainer(max_epochs=-1, accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint, early_stopping], log_every_n_steps=nbr_batches, fast_dev_run=False)
    trainer.fit(model, carla_dm)

if __name__ == '__main__':
    try: 
        train()
    except: 
        print(traceback.format_exc())