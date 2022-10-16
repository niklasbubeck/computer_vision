#from matplotlib.pyplot import close
from asyncio import constants
from random import random
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np 
import os 
import pytorch_lightning as pl
import torchvision.transforms.functional as transform
from torch.utils.data import random_split
import numpy as np 
from sampler.src.sampler import Sampler
import torch.nn.functional as F 

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud 
import os

class ProcessedNuScenesAndKittiDataset(Dataset):
    """
    @brief: The dataset that combines the NuScenes and the Kitti dataset. 

    @classvariables: None 

    @version: 0.01
    """

    def __init__(self, data_dir_kitti:str, data_dir_nuscenes:str, version:str, remove_classes_nuscenes:list=[], remove_classes_kitti:list=[], kitti_sequences:list=[], nuscenes_sequences:int=None, preproc:str="linear", randomize:bool=False) -> None:
        """
        @brief: Constructor of the ProcessedNuScenesAndKittiDataset class

        @type data_dir_kitti: str
        @param data_dir_kitti: String to the root data directory of the kitti dataset (usually /semanticKitti/dataset)

        @type data_dir_nuscenes: str
        @param data_dir_nuscenes: String to the root data directory of the NuScenes dataset

        @type version: str
        @param version: String defining the version of the NuScenes that should be used 

        @type remove_classes_nuscenes: list[int or str] 
        @param remove_classes_nuscenes: The classes that will be removed during sampling NuScenes data

        @type remove_classes_kitti: list[int or str] 
        @param remove_classes_kitti: The classes that will be removed during sampling SemanticKitti data

        @type kitti_sequences: list[int]
        @param kitti_sequences: What sequences of the semanticKitti data will be considered for the trainval split

        @type nuscenes_sequences: int 
        @param nuscenes_sequences: what amount of the trainval split will be used of the nuscenes data

        @type preproc: str
        @param preproc: Defines the sampling strategy (either linear or adaptive)

        @type randomize: bool 
        @param randomize: Rather the first 8192 points of a sample point cloud will be used, or if its gonna be randomized 
        """
        ## General 
        self.remove_classes_nuscenes = remove_classes_nuscenes
        self.remove_classes_kitti = remove_classes_kitti
        self.randomize = randomize
        self.preproc = preproc

        ## Prepare Nuscenes List
        self.nusc = NuScenes(version=version, dataroot=data_dir_nuscenes)
        dir = data_dir_nuscenes + "/samples/LIDAR_TOP"
        _, _, files = next(os.walk(dir))
        self.length = len(files)
        nulist = None
        if nuscenes_sequences != None:
            nulist = [str(elem) for elem in list(range(nuscenes_sequences))]
        else:    
            nulist = [str(elem) for elem in list(range(self.length))]
        print("Using %d many sample from nuscenes" % len(nulist))
        ## Prepare Kitti List
        paths = []
        for seq in kitti_sequences:
            path = os.path.join(data_dir_kitti, "sequences/%02d/velodyne" % int(seq)) # /storage/remote/atcremers61/s0100/ps4d/dataset/sequences/seq
            seq_paths = [os.path.join(path, file)
                        for file in sorted(os.listdir(path))]
            paths.append(seq_paths)
        paths = [item for sublist in paths for item in sublist]
        print("Using %d many samples from semanticKitti" % len(paths))
        ## Add all paths/sample_numbers

        self.all_data = paths + nulist
        print("Length of overall dataset: ", len(self.all_data))
        print(self.all_data)

    def __len__(self):
        """
        @brief: Defines the length of the data (batch)
        """
        return len(self.all_data) - 1

    def __getitem__(self, idx):
        """
        @brief: Constructs the batch given the idx
        
        @type idx: int
        @param idx: The index to use for constructin the batch
        """

        ## Use data from semanticKitti
        if len(self.all_data[idx]) > 10 and len(self.all_data[idx +1]) > 10:
            ## Compare to random pointcloud in range (-2, 2)
            path1 = self.all_data[idx]
            path2 = self.all_data[idx + 1]
            number_of_points = 8192
            ## Extract pointcloud 
            pc1 = np.fromfile(path1, dtype=np.float32).reshape((-1,4))[:,0:4]
            pc2 = np.fromfile(path2, dtype=np.float32).reshape((-1,4))[:,0:4]
            ## Extract labels 
            labels1 = np.fromfile(path1.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
            labels2 = np.fromfile(path2.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
            
            try:
                if self.preproc == "adaptive":
                
                    ds1 = Sampler(pc1, labels1, remove_classes=self.remove_classes_kitti)
                    ds2 = Sampler(pc2, labels2, remove_classes=self.remove_classes_kitti)
                    

                    voxel1, _, _ = ds1.adaptive_voxelization()
                    voxel2, _, _ = ds2.adaptive_voxelization()

                    

                elif self.preproc == "linear":
                    ds1 = Sampler(pc1, np.asarray([]), remove_classes=self.remove_classes_kitti)
                    ds2 = Sampler(pc2, np.asarray([]), remove_classes=self.remove_classes_kitti)

                    voxel1 = ds1.linear_voxelization(0.3) # voxel size
                    voxel2 = ds2.linear_voxelization(0.3) # voxel size



                if self.randomize:
                    idx1 = np.random.permutation(voxel1.shape[0])[: number_of_points]
                    idx2 = np.random.permutation(voxel2.shape[0])[: number_of_points]
                    voxel1 = voxel1[idx1, :]
                    voxel2 = voxel2[idx2, :]

                else:
                    voxel1 = voxel1[0:number_of_points, :]
                    voxel2 = voxel2[0:number_of_points, :]
            except:
                print("Something went wrong get other index")
                pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.all_data)-1))
                return pc1, pc2

            if voxel1.shape[0] < 200 or voxel2.shape[0] < 200:
                pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.all_data)-1))
                return pc1, pc2

            if voxel1.shape[0] < number_of_points:
                voxel1 = np.pad(voxel1, ((0,number_of_points-voxel1.shape[0]), (0,0)), mode="constant")
            
            if voxel2.shape[0] < number_of_points:
                voxel2 = np.pad(voxel2, ((0,number_of_points-voxel2.shape[0]), (0,0)), mode="constant")
            
            pc1 = torch.from_numpy(voxel1).float()
            pc2 = torch.from_numpy(voxel2).float()

            return pc1, pc2
        
        ## we have Nuscene Data
        elif len(self.all_data[idx]) < 10 and len(self.all_data[idx +1]) < 10:
            sample1 = self.nusc.sample[int(self.all_data[idx])]
            sample2 = self.nusc.sample[int(self.all_data[idx])]
            token1 = sample1["data"]["LIDAR_TOP"]
            token2 = sample2["data"]["LIDAR_TOP"]

            pointcloud_filename1 = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', token1)["filename"])
            label_filename1 = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", token1)["filename"])

            pointcloud_filename2 = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', token2)["filename"])
            label_filename2 = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", token2)["filename"])

            pc1 = LidarPointCloud.from_file(pointcloud_filename1).points.T[:, 0:4]
            pc2 = LidarPointCloud.from_file(pointcloud_filename2).points.T[:, 0:4]
            labels1 = np.fromfile(label_filename1, dtype=np.uint8).reshape((-1,1))
            labels2 = np.fromfile(label_filename2, dtype=np.uint8).reshape((-1,1))

            number_of_points = 8192
            try:
                if self.preproc == "adaptive":
            
                    ds1 = Sampler(pc1, labels1, remove_classes=self.remove_classes_nuscenes, dataset="nuscenes")
                    ds2 = Sampler(pc2, labels2, remove_classes=self.remove_classes_nuscenes, dataset="nuscenes")
                    

                    voxel1, _, _ = ds1.adaptive_voxelization()
                    voxel2, _, _ = ds2.adaptive_voxelization()

                    

                elif self.preproc == "linear":
                    ds1 = Sampler(pc1, np.asarray([]), remove_classes=self.remove_classes_nuscenes, dataset="nuscenes")
                    ds2 = Sampler(pc2, np.asarray([]), remove_classes=self.remove_classes_nuscenes, dataset="nuscenes")

                    voxel1 = ds1.linear_voxelization(0.3) # voxel size
                    voxel2 = ds2.linear_voxelization(0.3) # voxel size



                if self.randomize:
                    idx1 = np.random.permutation(voxel1.shape[0])[: number_of_points]
                    idx2 = np.random.permutation(voxel2.shape[0])[: number_of_points]
                    voxel1 = voxel1[idx1, :]
                    voxel2 = voxel2[idx2, :]

                else:
                    voxel1 = voxel1[0:number_of_points, :]
                    voxel2 = voxel2[0:number_of_points, :]
            except Exception as e:
                print("Something went wrong get other index")
                print(e)
                pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.all_data)-1))
                return pc1, pc2

            if voxel1.shape[0] < 200 or voxel2.shape[0] < 200:
                pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.all_data)-1))
                return pc1, pc2

            if voxel1.shape[0] < number_of_points:
                voxel1 = np.pad(voxel1, ((0,number_of_points-voxel1.shape[0]), (0,0)), mode="constant")
            
            if voxel2.shape[0] < number_of_points:
                voxel2 = np.pad(voxel2, ((0,number_of_points-voxel2.shape[0]), (0,0)), mode="constant")
            
            pc1 = torch.from_numpy(voxel1).float()
            pc2 = torch.from_numpy(voxel2).float()
            return pc1, pc2
        ## we have mix of both --> skip
        else:
            pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.all_data)-1))
            return pc1, pc2

class ProcessedNuScenesDataset(Dataset):
    """
    @brief: The dataset for the NuScenes data. 

    @classvariables: None 

    @version: 0.01
    """
    
    def __init__(self, data_dir, version, remove_classes:list=[], preproc:str="linear", randomize:bool=False) -> None:
        """
        @brief: Constructor of the ProcessedNuScenesDataset class

        @type data_dir: str
        @param data_dir: String to the root data directory of the NuScenes dataset

        @type version: str
        @param version: String defining the version of the NuScenes that should be used 

        @type remove_classes: list[int or str] 
        @param remove_classes: The classes that will be removed during sampling NuScenes data

        @type preproc: str
        @param preproc: Defines the sampling strategy (either linear or adaptive)

        @type randomize: bool 
        @param randomize: Rather the first 8192 points of a sample point cloud will be used, or if its gonna be randomized 
        """
        
        self.nusc = NuScenes(version=version, dataroot=data_dir)
        ## Get number of samples
        self.data_dir = data_dir 
        self.remove_classes = remove_classes
        self.randomize = randomize
        self.preproc = preproc
        dir = data_dir + "/samples/LIDAR_TOP"
        print(dir)
        _, _, files = next(os.walk(dir))
        self.length = len(files)
        print(self.length)

    def __len__(self):
        """
        @brief: Defines the length of the data (batch)
        """
        return self.length - 1

    def __getitem__(self, idx):
        """
        @brief: Constructs the batch given the idx
        
        @type idx: int
        @param idx: The index to use for constructin the batch
        """
        sample1 = self.nusc.sample[idx]
        sample2 = self.nusc.sample[idx + 1]
        token1 = sample1["data"]["LIDAR_TOP"]
        token2 = sample2["data"]["LIDAR_TOP"]

        pointcloud_filename1 = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', token1)["filename"])
        label_filename1 = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", token1)["filename"])

        pointcloud_filename2 = os.path.join(self.nusc.dataroot, self.nusc.get('sample_data', token2)["filename"])
        label_filename2 = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", token2)["filename"])

        pc1 = LidarPointCloud.from_file(pointcloud_filename1).points.T[:, 0:4]
        pc2 = LidarPointCloud.from_file(pointcloud_filename2).points.T[:, 0:4]
        labels1 = np.fromfile(label_filename1, dtype=np.uint8).reshape((-1,1))
        labels2 = np.fromfile(label_filename2, dtype=np.uint8).reshape((-1,1))

        number_of_points = 8192
        try:
            if self.preproc == "adaptive":
        
                ds1 = Sampler(pc1, labels1, remove_classes=self.remove_classes, dataset="nuscenes")
                ds2 = Sampler(pc2, labels2, remove_classes=self.remove_classes, dataset="nuscenes")
                

                voxel1, _, _ = ds1.adaptive_voxelization()
                voxel2, _, _ = ds2.adaptive_voxelization()

                

            elif self.preproc == "linear":
                ds1 = Sampler(pc1, np.asarray([]), remove_classes=self.remove_classes, dataset="nuscenes")
                ds2 = Sampler(pc2, np.asarray([]), remove_classes=self.remove_classes, dataset="nuscenes")

                voxel1 = ds1.linear_voxelization(0.3) # voxel size
                voxel2 = ds2.linear_voxelization(0.3) # voxel size



            if self.randomize:
                idx1 = np.random.permutation(voxel1.shape[0])[: number_of_points]
                idx2 = np.random.permutation(voxel2.shape[0])[: number_of_points]
                voxel1 = voxel1[idx1, :]
                voxel2 = voxel2[idx2, :]

            else:
                voxel1 = voxel1[0:number_of_points, :]
                voxel2 = voxel2[0:number_of_points, :]
        except:
            print("Something went wrong get other index")
            pc1, pc2 = self.__getitem__(np.random.randint(0, self.length)-1)
            return pc1, pc2

        if voxel1.shape[0] < 200 or voxel2.shape[0] < 200:
            pc1, pc2 = self.__getitem__(np.random.randint(0, self.length)-1)
            return pc1, pc2

        print("Voxel Shape1: ", voxel1.shape)

        if voxel1.shape[0] < number_of_points:
            voxel1 = np.pad(voxel1, ((0,number_of_points-voxel1.shape[0]), (0,0)), mode="constant")
        
        if voxel2.shape[0] < number_of_points:
            voxel2 = np.pad(voxel2, ((0,number_of_points-voxel2.shape[0]), (0,0)), mode="constant")
        
        pc1 = torch.from_numpy(voxel1).float()
        pc2 = torch.from_numpy(voxel2).float()
        return pc1, pc2



class ProcessedKittiDataset(Dataset):
    """
    @brief: The dataset for the NuScenes data. 

    @classvariables: None 

    @version: 0.01
    """

    def __init__(self, data_dir, sequences, remove_classes=[], preproc:str="linear", randomize:bool=False) -> None:
        """
        @brief: Constructor of the ProcessedKittiDataset class

        @type data_dir: str
        @param data_dir: String to the root data directory of the kitti dataset (usually /semanticKitti/dataset)

        @type remove_classes: list[int or str] 
        @param remove_classes: The classes that will be removed during sampling SemanticKitti data

        @type sequences: list[int]
        @param sequences: What sequences of the semanticKitti data will be considered for the trainval split

        @type preproc: str
        @param preproc: Defines the sampling strategy (either linear or adaptive)

        @type randomize: bool 
        @param randomize: Rather the first 8192 points of a sample point cloud will be used, or if its gonna be randomized 
        """
        self.data_dir = data_dir 
        self.remove_classes = remove_classes
        self.randomize = randomize
        self.preproc = preproc
        self.paths = []
        for seq in sequences:
            path = os.path.join(data_dir, "sequences/%02d/velodyne" % int(seq)) # /storage/remote/atcremers61/s0100/ps4d/dataset/sequences/seq
            seq_paths = [os.path.join(path, file)
                        for file in sorted(os.listdir(path))]
            self.paths.append(seq_paths)
        self.paths = [item for sublist in self.paths for item in sublist]
        self.paths = self.paths + self.paths[::-1]
        ## Comment out to set certain frames for overfitting and only give one scene!!!
        # self.paths = self.paths[65:75]
        print(self.paths)

    def __len__(self):
        """
        @brief: Defines the length of the data (batch)
        """
        ## -1 to not overshoot the range because second pointcloud is idx +1
        return len(self.paths) - 1

    def __getitem__(self, idx):
        """
        @brief: Constructs the batch given the idx
        
        @type idx: int
        @param idx: The index to use for constructin the batch
        """
        ## Compare to random pointcloud in range (-2, 2)
        path1 = self.paths[idx]
        path2 = self.paths[idx + 1]
        number_of_points = 8192
        ## Extract pointcloud 
        pc1 = np.fromfile(path1, dtype=np.float32).reshape((-1,4))[:,0:4]
        pc2 = np.fromfile(path2, dtype=np.float32).reshape((-1,4))[:,0:4]
        ## Extract labels 
        labels1 = np.fromfile(path1.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
        labels2 = np.fromfile(path2.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
        
        try:
            if self.preproc == "adaptive":
            
                ds1 = Sampler(pc1, labels1, remove_classes=self.remove_classes)
                ds2 = Sampler(pc2, labels2, remove_classes=self.remove_classes)
                

                voxel1, _, _ = ds1.adaptive_voxelization()
                voxel2, _, _ = ds2.adaptive_voxelization()

                

            elif self.preproc == "linear":
                ds1 = Sampler(pc1, np.asarray([]), remove_classes=self.remove_classes)
                ds2 = Sampler(pc2, np.asarray([]), remove_classes=self.remove_classes)

                voxel1 = ds1.linear_voxelization(0.3) # voxel size
                voxel2 = ds2.linear_voxelization(0.3) # voxel size



            if self.randomize:
                idx1 = np.random.permutation(voxel1.shape[0])[: number_of_points]
                idx2 = np.random.permutation(voxel2.shape[0])[: number_of_points]
                voxel1 = voxel1[idx1, :]
                voxel2 = voxel2[idx2, :]

            else:
                voxel1 = voxel1[0:number_of_points, :]
                voxel2 = voxel2[0:number_of_points, :]
        except:
            print("Something went wrong get other index")
            pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.paths)-1))
            return pc1, pc2

        if voxel1.shape[0] < 200 or voxel2.shape[0] < 200:
            pc1, pc2 = self.__getitem__(np.random.randint(0, len(self.paths)-1))
            return pc1, pc2

        if voxel1.shape[0] < number_of_points:
            voxel1 = np.pad(voxel1, ((0,number_of_points-voxel1.shape[0]), (0,0)), mode="constant")
        
        if voxel2.shape[0] < number_of_points:
            voxel2 = np.pad(voxel2, ((0,number_of_points-voxel2.shape[0]), (0,0)), mode="constant")
        
        pc1 = torch.from_numpy(voxel1).float()
        pc2 = torch.from_numpy(voxel2).float()

        return pc1, pc2


def get_default_device():
    """ 
    @brief: Set Device to GPU if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    """
    @brief: Move data to the device
    """
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader(DataLoader):
    """ 
    @brief: Wrap a dataloader to move data automatically to a device
    """
    
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

class KittiDataModule(pl.LightningDataModule):
    """
    @brief: The DataModule that loads the dataset 

    @classvariables: None 

    @version: 0.01
    """
    def __init__(self, data_module_params):
        """
        @brief: Constructor of the ProcessedNuScenesDataset class

        @data_module_params: dict
        @data_module_params: dictionary of parameters used for loading.
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = data_module_params["dataset"]
        self.data_dir_kitti = data_module_params["data_dir_kitti"]
        self.data_dir_nuscenes = data_module_params["data_dir_nuscenes"]
        self.nuscenes_version = data_module_params["nuscenes_version"]
        self.kitti_sequences = data_module_params["kitti_sequences"]
        self.nuscenes_sequences = data_module_params["nuscenes_sequences"]
        self.split_in_pct = data_module_params["split_in_pct"]
        self.batch_size = data_module_params["batch_size"]
        self.num_workers = data_module_params["num_workers"]
        self.remove_classes_kitti = data_module_params["remove_classes_kitti"]
        self.remove_classes_nuscenes = data_module_params["remove_classes_nuscenes"]
        self.preproc = data_module_params["preproc"]
        self.randomize = data_module_params["randomize"]

        dataset = None
        if self.dataset == "ProcessedNuScenesAndKittiDataset":
            dataset = ProcessedNuScenesAndKittiDataset(self.data_dir_kitti, self.data_dir_nuscenes, self.nuscenes_version, self.remove_classes_nuscenes, self.remove_classes_kitti, self.kitti_sequences, self.nuscenes_sequences, self.preproc, self.randomize)
        elif self.dataset == "ProcessedNuScenesDataset":
            dataset = ProcessedNuScenesDataset(self.data_dir_nuscenes, self.nuscenes_version, self.remove_classes_nuscenes, self.preproc, self.randomize)
        elif self.dataset == "ProcessedKittiDataset":
            dataset = ProcessedKittiDataset(self.data_dir_kitti, self.kitti_sequences, self.remove_classes_kitti, self.preproc, self.randomize)
        
        train_size = int(len(dataset)* self.split_in_pct) 
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset,[train_size, val_size])


    def train_dataloader(self):
        """
        @brief: dataloader for the training
        """
        return DeviceDataLoader(DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True), get_default_device())

    def val_dataloader(self):
        """
        @brief: dataloader for the validation
        """
        return DeviceDataLoader(DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True), get_default_device())



if __name__ == '__main__':
    pass
