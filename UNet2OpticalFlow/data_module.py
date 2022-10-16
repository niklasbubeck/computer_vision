from matplotlib.pyplot import close
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np 
import os 
import csv
import cv2
import pytorch_lightning as pl
import pickle
import torchvision.transforms.functional as transform
from torch.utils.data import random_split
import traceback



class CarlaDataset(Dataset):
    """Carla dataset."""

    def __init__(self, data_dir):
        """
        Args:
            data_dir (string): Directory of the dataset containing the pickles
        """
        self.data_dir = data_dir

        self.pickle_paths = [os.path.join(data_dir, file)
                       for file in sorted(os.listdir(data_dir))]
       
        self.length = len(self.pickle_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        object = []
        with (open(self.pickle_paths[idx], "rb")) as openfile:
                while True:
                    try:
                        object.append(pickle.load(openfile))
                    except EOFError:
                        # print(traceback.format_exc())
                        break
                close(self.pickle_paths[idx])

        gray1 = transform.to_tensor(object[0]['gray1'])
        gray2 = transform.to_tensor(object[0]['gray2'])
        labels = transform.to_tensor(object[0]['gt_label'])
        image = torch.cat((gray1, gray2), 0)
        return image, labels

def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader(DataLoader):
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        super().__init__(dl)
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)

class CarlaDataModule(pl.LightningDataModule):

    def __init__(self, data_module_params):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_module_params["data_dir"]
        self.split_in_pct = data_module_params["split_in_pct"]
        self.batch_size = data_module_params["batch_size"]
        self.num_workers = data_module_params["num_workers"]

        dataset = CarlaDataset(self.data_dir)
        train_size = int(len(dataset)* self.split_in_pct) 
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset,[train_size, val_size])

        self.mean, self.std = self.compute_mean_std()

    def train_dataloader(self):
        return DeviceDataLoader(DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True), get_default_device())

    def val_dataloader(self):
        return DeviceDataLoader(DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True), get_default_device())

    def compute_mean_std(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        loader = self.train_dataloader()
        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        mean /= nb_samples
        std /= nb_samples
        mean = mean.cpu().data.numpy().tolist()
        std = std.cpu().data.numpy().tolist()
        return mean, std 



if __name__ == '__main__':
    pass