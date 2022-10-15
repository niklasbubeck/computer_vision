from cProfile import label
import torch 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as transform
import numpy as np 
import os 
import csv
import cv2
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class CarlaDataset(Dataset): 
    def __init__(self, label_file, img_dir, transform=None, target_transform=None) -> None:
        # super().__init__()
        self.labels = pd.read_csv(label_file, usecols=["path", "steer", "throttle"])
        self.img_dir = img_dir
        self.transfrom = transform
        self.target_transform = target_transform



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels.iloc[index,0])
        head_tail = os.path.split(img_path)
        
        rgb_mid_path = os.path.join(head_tail[0], "rgb_mid" + head_tail[1][3::])
        rgb_left_path = os.path.join(head_tail[0], "rgb_left" + head_tail[1][3::])
        rgb_right_path = os.path.join(head_tail[0], "rgb_right" + head_tail[1][3::])
        depth_path = os.path.join(head_tail[0], "depth" + head_tail[1][3::])


        image_mid = Image.open(rgb_mid_path).convert("L")
        image_mid = transform.to_tensor(image_mid)

        image_left = Image.open(rgb_left_path).convert("L")
        image_left = transform.to_tensor(image_left)

        image_right = Image.open(rgb_right_path).convert("L")
        image_right = transform.to_tensor(image_right)

        depth_image = Image.open(depth_path)
        depth_image = transform.to_tensor(depth_image)

        image = torch.cat((image_mid, image_left, image_right, depth_image), 0)

        steer = self.labels.iloc[index, 1]
        throttle = self.labels.iloc[index, 2]
        label = np.array([steer, throttle])
        if self.transfrom: 
            image = self.transfrom(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, np.float32(label)

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


def get_sampling_weights(label_file, n_bins):
    labels = pd.read_csv(label_file, usecols=["steer", "throttle"])
    labels = labels.to_numpy()
    
    throttle, steer = labels[:,1], labels[:,0]
    
    nbr_samples_throttle, bins_throttle = np.histogram(throttle, bins=n_bins)
    nbr_samples_steer, bins_steer = np.histogram(steer, bins=n_bins)


    plt.hist(steer, bins=5, range=[-1, 4], histtype='step',edgecolor='r',linewidth=3)
    nbr_samples_steer += np.histogram(steer[6:], bins=n_bins)[0]

    plt.bar(bins_steer[:-1],nbr_samples_steer,width=1)

    plt.show()

    nbr_samples_throttle = np.repeat(np.expand_dims(nbr_samples_throttle, axis=0), repeats=len(throttle), axis=0)
    nbr_samples_steer = np.repeat(np.expand_dims(nbr_samples_steer, axis=0), repeats=len(steer), axis=0)

    bins_throttle[0], bins_throttle[-1] = -np.inf, np.inf
    bins_steer[0], bins_steer[-1] = -np.inf, np.inf
    
    mask_throttle = (bins_throttle[:-1][None,:]<=throttle[:,None])*(throttle[:,None]<bins_throttle[1:][None,:])
    mask_steer = (bins_steer[:-1][None,:]<=steer[:,None])*(steer[:,None]<bins_steer[1:][None,:])
    
    nbr_samples_throttle = nbr_samples_throttle[mask_throttle]
    nbr_samples_steer = nbr_samples_steer[mask_steer]

    sampling_weights = 1/(nbr_samples_throttle*nbr_samples_steer)

    return sampling_weights

def create_csv_file(dataset):
    lists = []
    for data, moin in tqdm(dataset):
        lists.append(moin)
    array = np.array(lists)
    df = pd.DataFrame({"steer": array[:, 0], "throttle": array[:, 1]})
    df.to_csv("_out/train.csv")

class CarlaDataModule(pl.LightningDataModule):

    def __init__(self, data_module_params):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_module_params["data_dir"]
        self.label_file = data_module_params["label_file"]
        self.split_in_pct = data_module_params["split_in_pct"]
        self.batch_size = data_module_params["batch_size"]
        self.num_workers = data_module_params["num_workers"]

        dataset = CarlaDataset(self.label_file, self.data_dir)
        print(dataset.__len__())
        train_size = int(dataset.__len__()* self.split_in_pct) 
        val_size = dataset.__len__() - train_size
        print("Split: Train = %d, Val = %d" % (train_size, val_size))
        self.train_data, self.val_data = random_split(dataset,[train_size, val_size])
        create_csv_file(self.train_data)
        # create_csv_file(self.val_data)

        sampling_weights = get_sampling_weights(label_file="_out/train.csv", n_bins=10)
        self.sampler = torch.utils.data.WeightedRandomSampler(weights=sampling_weights, num_samples=len(sampling_weights), replacement=True)
        self.mean, self.std = self.compute_mean_std()

    def train_dataloader(self):
        return DeviceDataLoader(DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.sampler ,num_workers=self.num_workers), get_default_device())

    def val_dataloader(self):
        return DeviceDataLoader(DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), get_default_device())

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
        mean = mean.cpu().numpy().tolist()
        std = std.cpu().numpy().tolist()
        return mean, std 

if __name__ == '__main__':
    pass