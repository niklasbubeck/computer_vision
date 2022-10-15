from data_module import CarlaDataModule
from resnet_module import ResNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import datetime

# TODO : Edit global paths
experiment = "DS1_ResNet"
data_module_params = {"label_file":"_out/curve_to_steering.csv", 
                      "data_dir":"",
                      "split_in_pct":0.8,
                      "batch_size":32, 
                      "num_workers":4}
model_module_params = {"learning_rate":0.0001}
save_dir = ""

def train():
    time_stamp = str(datetime.datetime.now())
    time_stamp, _ = time_stamp.rsplit(".")
    time_stamp = time_stamp.replace("-","_")
    time_stamp = time_stamp.replace(" ","_")
    time_stamp = time_stamp.replace(":","")
    folder_name = time_stamp + "_" + experiment

    carla_dm = CarlaDataModule(data_module_params)
    model_module_params["mean"], model_module_params["std"] = carla_dm.mean, carla_dm.std
    model = ResNet(model_module_params)

    logger = TensorBoardLogger(save_dir, name='logs', version=folder_name)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(save_dir,"checkpoints",folder_name), filename='ResNet-{epoch:04d}-{MAE_Val:.8f}', monitor="MAE_Train", verbose=True, save_last=True, auto_insert_metric_name=True)
    early_stopping = EarlyStopping(monitor="MAE_Train", min_delta=0.00000001, patience=50, verbose=True, mode="min", strict=True)
    
    nbr_batches = len(carla_dm.train_dataloader())
    trainer = Trainer(max_epochs=-1, accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint, early_stopping], log_every_n_steps=nbr_batches, fast_dev_run=False)
    trainer.fit(model, carla_dm)

if __name__ == '__main__':
    train()