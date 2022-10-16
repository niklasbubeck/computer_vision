# from cProfile import label
# from nbformat import write
import numpy as np 
import time 
import argparse
import open3d as o3d 
import yaml
import os
from scipy import spatial
from tqdm import tqdm
import sys 
sys.setrecursionlimit(10000)




class Sampler():
    """
    @brief: The Sampler class provides methods to downsample original pointclouds of size [n, 4] to a 
            downscaled representation of [m, 3] as also to upsample back to the original size. 

    @classvariables: None 

    @version: 0.01 -> sampling with intensities is only available for downsampling....we probably dont even need it at all
    """

    def __init__(self, pointcloud: np.ndarray, labels:np.ndarray, vis:bool=False, write:bool=False, remove_classes:list = [], with_intens:bool = False, dataset:str="kitti") -> None:
        """
        @brief: Constructor of the Sampler method

        @type pointcloud: ndarray [n, 4]
        @param pointcloud: The original pointcloud with position [n, 0:3] and intensity values [n, 3]

        @type labels: ndarray [n, 1]
        @param labels: The semantic labels of the corresponding pointcloud

        @type vis: bool
        @param vis: Determine if intermediate steps like voxelized pointclouds will be visualized

        @type write: bool 
        @param write: Determine if intermediate results like voxelized pointclouds will be saved

        @type remove_classes: list
        @param remove_classes: Contains the classes the will be neglected for the sampling

        @type with_intens: bool 
        @param with_intens: Determines if the intensity values will also be sampled (Under constraction for upsampling)
        """
        
        ## Init the args
        self.pc = pointcloud[:, 0:3]
        self.its = pointcloud[:, 3]
        self.pc_labels = labels
        self.vis = vis
        self.write = write
        self.with_intens = with_intens
        self.remove_classes = remove_classes

        ## Init timing, and configuration params
        self.time = time.time()
        if dataset == "kitti":
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs/voxel_sizes_kitti.yaml")
            with open(path, "r") as file:
                self.scale_params = yaml.safe_load(file)
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs/semantic-kitti.yaml")
            with open(path, "r") as file:
                self.labels = yaml.safe_load(file)["labels"]
            self.label_dep_pc = self._get_label_dep_pointclouds()
        elif dataset == "nuscenes":
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs/voxel_sizes_nuscenes.yaml")
            with open(path, "r") as file:
                self.scale_params = yaml.safe_load(file)
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../configs/semantic-nuscenes.yaml")
            with open(path, "r") as file:
                self.labels = yaml.safe_load(file)["labels"]
            self.label_dep_pc = self._get_label_dep_pointclouds()

        ## Cache downsampling result for upsampling
        self.downsampled = None

    def _get_label_dep_pointclouds(self) -> dict:
        """
        @brief: splits the pointcloud by its classes and returns the pointcloud for a corresponding class within a dict e.g. {"ground": ndarray, ...}

        @type dict: dict
        @return dict: The dictionary that includes class based pointclouds as value with the corresponding class name as key 
        """
        dict = {}
        for item in self.labels.items():
            if item[0] in self.remove_classes or item[1] in self.remove_classes:
                continue
            globals()[f"{item[1]}"] = self.pc[np.where(self.pc_labels == item[0])[0], :]
            dict[f"{item[1]}"] = globals()[f"{item[1]}"]

        return dict

    def linear_voxelization(self, sample_size:float) -> np.ndarray:
        """
        @brief: makes a linear voxelization independent of the classes 

        @type sample_size: float
        @param sample_size: The size to sample for. Will create voxels with w, h, l = sample_size

        @type voxelized_pc: ndarray
        @param voxelized_pc: The voxelized pointcloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pc)

        voxelized_pc = np.asarray(pcd.voxel_down_sample(voxel_size = sample_size).points)

        if self.write:
            o3d.io.write_point_cloud("_output/pointclouds/frame_%s.ply" % self.time, voxelized_pc)

        if self.vis: 
            o3d.visualization.draw_geometries(voxelized_pc)

        self.downsampled = voxelized_pc
        return voxelized_pc

    def height_based_ground_removal(self, height:float=0):
        pass

    def adaptive_voxelization(self) -> np.ndarray:
        """
        @type voxel_pc: ndarray(float)
        @return voxel_pc: Pointcloud only containing position [n, 3]

        @type labels_pc: ndarray(str)
        @return labels_pc: Labels to the corresponding point [n, 1]

        @type intensities_pc: ndarray(float)
        @return intensities_pc: Intensity values [n, 1]
        """
        pcds = []
        np_pcds = None
        counter=0
        for item in self.label_dep_pc.items():
            if item[1].shape[0] == 0:
                continue
            counter += 1   
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(item[1])
            voxelized_pcd = pcd.voxel_down_sample(voxel_size=self.scale_params[item[0]])
            pcds.append(voxelized_pcd)
            if counter == 1:
                np_pcds = np.concatenate((np.asarray(voxelized_pcd.points), np.full((np.asarray(voxelized_pcd.points).shape[0], 1), item[0])), axis=1)
                continue
            conc = np.concatenate((np.asarray(voxelized_pcd.points), np.full((np.asarray(voxelized_pcd.points).shape[0], 1), item[0])), axis=1)
            np_pcds = np.concatenate((np_pcds, np.asarray(conc)), axis=0)


        
        if self.with_intens:
            np_pcds = self._get_intensity_sampling(np_pcds)
        
        else:
            zeros = np.zeros((np_pcds.shape[0], 1))
            np_pcds = np.concatenate((np_pcds, zeros), axis=1)

        if self.vis: 
            o3d.visualization.draw_geometries(pcds)
        
        if self.write: 
            print("Save Pointclouds")
             ## make directories
            try:
                os.makedirs("_output/pointclouds/")
            except Exception as e:
                print(e)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcds[:,0:3])
            o3d.io.write_point_cloud("_output/pointclouds/adaptive_voxel_frame_%s.ply" % self.time, pcd)
            
        self.downsampled = np_pcds[:, 0:4]
        voxel_pc = np_pcds[:, 0:3].astype(float)
        labels_pc = np_pcds[:, 3].reshape(np_pcds.shape[0], 1)
        intensities_pc = np_pcds[:, 4].astype(float).reshape(np_pcds.shape[0], 1)

        return voxel_pc, labels_pc, intensities_pc

    def _get_intensity_sampling(self, sampled_pc:np.ndarray) -> np.ndarray:
        """
        @brief: Samples the intensity values for a voxel 

        @type sampled_pc: ndarray(float)
        @param sampled_pc: The downsampled pointcloud [n, 4]

        @type res: ndarray(str)
        @return res: Array containing the positions [n, 0:3] the corresponding labels [n, 3] and the sampled intensities [n, 4]
        """
        intensities_in_voxel = np.zeros((sampled_pc.shape[0], 2))
        for i, its in enumerate(tqdm(self.its)):
            # pick the orig point of the intensity and its label
            orig_point = self.pc[i]
            orig_label = self._label_number_to_text(self.pc_labels[i][0])
            
            # Mask points with the correct label
            masked = sampled_pc[np.where(sampled_pc[:,3] == orig_label), :]
            ## When class got removed skip it 
            if masked.shape[1] == 0:
                continue
            
            ## Calculate Nearest Neighbor of same class with a KDtree 
            distance, index = spatial.KDTree(masked[0, :, 0:3].astype(np.float)).query(orig_point, p=1)
            intensities_in_voxel[index][0] += its 
            intensities_in_voxel[index][1] += 1
        
        ## Transform not a number due to zero division to 0
        mean = np.nan_to_num(intensities_in_voxel[:, 0] / intensities_in_voxel[:,1]).reshape(sampled_pc.shape[0], 1)
        
        res = np.concatenate((sampled_pc, mean), axis=1)
        return res


    def _label_number_to_text(self, number:int) -> str:
        """
        @brief: Transfer the label number to the corresponding text label 

        @type number: int
        @param number: the number to transfer to text 

        @type up_flow: str 
        @return text_label: the label as text in str
        """
        text_label = self.labels.get(number)
        return text_label

    def upsample(self, flow:np.ndarray, p:int=1, padded=False) -> np.ndarray: 
        """
        @brief: Upsampling the downsampled flow array [m, 3] by finding the representative nearest neighbor with same class and apply representive flow. 
                Thus each point in the original pointcloud [n, 3] gets a flow value. The more classes are removed, the more sparse the matrix gets.

        @type flow: ndarray(float)
        @param flow: The flow of the downsampled pointcloud [m, 3]

        @type p: int 
        @param p: The value of Norm to be applied (L1 = 1, L2 = 2 ....)

        @type up_flow: ndarray(float) [n, 3]
        @param up_flow: The upsampled flow array [n, 3]
        """
        ## Init arrays
        up_flow = np.zeros(self.pc.shape)
        downsampled = self.downsampled
       
        ## For each point in the original pc compare to the nearest voxel representation of the same class and apply the same flow
        for i, point in enumerate(tqdm(self.pc)):
            label = self._label_number_to_text(self.pc_labels[i][0])
            idx = np.where(downsampled[:,3] == label)
            masked = downsampled[idx]
            ## When class got removed skip it 
            if masked.shape[0] == 0:
                continue
            point_scaled = np.repeat(point.reshape(-1,3), masked.shape[0], axis=0)
            l1_dist = np.linalg.norm(point_scaled - masked[:, :3].astype(np.float), axis=1, ord=1)
            index = np.argmin(l1_dist)

            up_flow[i][0] = flow[idx[0][index]][0]
            up_flow[i][1] = flow[idx[0][index]][1]
            up_flow[i][2] = flow[idx[0][index]][2]
        return up_flow

if __name__ == '__main__':

    # Args
    parser = argparse.ArgumentParser(description='sample points')
    parser.add_argument('--seq', type=int, default=2, help='used sequence')
    parser.add_argument('--frame', type=int, default=105, help='used fram')
    args = parser.parse_args()

    print("\n-----Start Sampler Example:-----\n sequence:%s \n frame:%s \n" % (args.seq, args.frame))

    sequence = args.seq
    sequence_text = f'{sequence:02d}'
    scannum = args.frame
    scannum_next = scannum + 1
    scannum_text = f'{scannum:06d}'
    scannum_next_text =   f'{scannum_next:06d}'

    filename = f'/media/niklas/Extreme SSD/semanticKitti/dataset/sequences/{sequence_text}/velodyne/{scannum_text}.bin'
    filename2 = f'/media/niklas/Extreme SSD/semanticKitti/dataset/sequences/{sequence_text}/velodyne/{scannum_next_text}.bin'

    ## Extract pointcloud 
    pc1 = np.fromfile(filename,dtype=np.float32).reshape((-1,4))
    pc2 = np.fromfile(filename2, dtype=np.float32).reshape((-1, 4))

    ## Extract labels 
    labels1 = np.fromfile(filename.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
    labels2 = np.fromfile(filename.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
    print("Original PC shape: ", pc1.shape)

    ## Init Sampler
    remove_classes = ["unlabeled", "outlier", "road", "parking", "sidewalk", "other-ground", "building", "fence", "other-structure", "lane-marking", "vegetation", "trunk", "terrain", "pole", "traffic-sign", "other-object"]
    ds = Sampler(pc1, labels1, vis=True, remove_classes=remove_classes)
    voxel1, _, _ = ds.adaptive_voxelization()
    print("Downsampled Shape: ", voxel1.shape)

    ## Make random flow
    flow = np.random.rand(voxel1.shape[0], voxel1.shape[1])

    ## Upsample again
    upsample = ds.upsample(flow, 1)
    print("Upsampled Shape: ", upsample.shape)

    print(upsample)
    np.savetxt("test.txt", upsample)



