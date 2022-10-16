import imp
from pwc_pointconv import PointConvSceneFlowPWC8192selfglobalPointConv
#from original_pwc import PointConvSceneFlowPWC8192selfglobalPointConv
import numpy as np
import open3d as o3d 
import time
import torch
from sampler.src.sampler import Sampler
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud 
from loss import computeChamfer
import os
import argparse
import traceback
from tqdm import tqdm

def rigid_transformation(data_dir, sequence, frames):
    """
    @brief: applies the rigid transformation to the pointcloud 

    @type data_dir: str
    @param data_dir: Data root directory

    @type sequence: int
    @param sequence: Number of the sequence

    @type frames: list[int,int]
    @param frames: the two frames to use for transformation

    @type new_coords: np.ndarray
    @return new_coords: pointcloud with applied rigid transformation
    """
    ## Get Stuff from dataset
    sequence_text = f'{sequence:02d}'
    frame1 = f'{frames[0]:06d}'
    frame2 = f'{frames[1]:06d}'
    filename =  os.path.join(data_dir , f'sequences/{sequence_text}/velodyne/{frame1}.bin')
    filename2 = os.path.join(data_dir, f'sequences/{sequence_text}/velodyne/{frame2}.bin')
    calib_filename = os.path.join(data_dir, f'sequences/{sequence_text}/calib.txt')
    poses_filename = os.path.join(data_dir, f'sequences/{sequence_text}/poses.txt')
    pc1 = np.fromfile(filename,dtype=np.float32).reshape(-1,4)
    pc2 = np.fromfile(filename2, dtype=np.float32).reshape((-1, 4))

    ## Get calibration
    calib = {}
    calib_file = open(calib_filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()

    ## Get Poses
    poses_file = open(poses_filename)
    poses = []
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in poses_file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    pose1 = poses[frames[0]]
    pose2 = poses[frames[1]]

    ## Apply the transformation
    # to global coords
    hpoints = np.hstack((pc1[:, :3], np.ones_like(pc1[:, :1])))
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose1.T, axis=1)[:, :3]
    # to next frame coords
    new_coords = new_points - pose2[:3, 3]
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose2[:3, :3], axis=1)
    
    return new_coords

def flow_inference(data_dir, output, seq, frame, model, write=False):
    """
    @brief: Does the scene flow inference as well as applying the rigid transformation and the flow to the pointcloud 

    @type data_dir: str
    @param data_dir: Data root directory

    @type seq: int
    @param seq: Number of the sequence

    @type frame: int
    @param frame: the frame of the sequence

    @type model: str
    @param model: path the model ckpt to use 

    @type write: bool
    @param write: decides if to generate the ply files of the pointclouds

    @type rigid: np.ndarray
    @return rigid: pointcloud with applied rigid and flow transformation
    """

    sequence_text = f'{seq:02d}'
    frame1 = f'{frame:06d}'
    frame2 = f'{frame+1 :06d}'
    filename =  os.path.join(data_dir , f'sequences/{sequence_text}/velodyne/{frame1}.bin')
    filename2 = os.path.join(data_dir, f'sequences/{sequence_text}/velodyne/{frame2}.bin')

    ## Extract pointcloud 
    pc1_load = np.fromfile(filename,dtype=np.float32).reshape((-1,4))
    pc2_load = np.fromfile(filename2, dtype=np.float32).reshape((-1, 4))

    ## Extract labels 
    labels1 = np.fromfile(filename.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
    labels2 = np.fromfile(filename2.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) & 0xFFFF
    # print("Original PC1 shape: ", pc1_load.shape)
    # print("Original PC2 shape: ", pc2_load.shape)

    # print("Original Labels1 shape: ", labels1.shape)
    # print("Original Labels2 shape: ", labels2.shape)
    ## Init Samplers
    remove_classes = ["outlier", "unlabeled", "road", "parking", "other-ground", "building", "sidewalk", "vegetation", "terrain", "other-object"]
    ds1 = Sampler(pc1_load, labels1, remove_classes=remove_classes)
    ds2 = Sampler(pc2_load, labels2, remove_classes=remove_classes)
    
    # ## For Nuscenses
    # nusc = NuScenes(version="v1.0-trainval", dataroot="/media/niklas/Extreme SSD/data/sets/nuscenes", verbose=True)
    # sample1 = nusc.sample[380]
    # sample2 = nusc.sample[380 + 1]
    # token1 = sample1["data"]["LIDAR_TOP"]
    # token2 = sample2["data"]["LIDAR_TOP"]

    # pointcloud_filename1 = os.path.join(nusc.dataroot, nusc.get('sample_data', token1)["filename"])
    # label_filename1 = os.path.join(nusc.dataroot, nusc.get("lidarseg", token1)["filename"])

    # pointcloud_filename2 = os.path.join(nusc.dataroot, nusc.get('sample_data', token2)["filename"])
    # label_filename2 = os.path.join(nusc.dataroot, nusc.get("lidarseg", token2)["filename"])

    # pc1_load = LidarPointCloud.from_file(pointcloud_filename1).points.T[:, 0:4]
    # pc2_load = LidarPointCloud.from_file(pointcloud_filename2).points.T[:, 0:4]
    # print("Orig Pointcloud1 Shape: ", pc1_load.shape)
    # print("Orig Pointcloud2 Shape: ", pc2_load.shape)
    
    # labels1 = np.fromfile(label_filename1, dtype=np.uint8).reshape((-1,1))
    # labels2 = np.fromfile(label_filename2, dtype=np.uint8).reshape((-1,1))
    
    # remove_classes = ["noise", "static_object_bicycle_rack", "flat_driveable_surface","flat_sidewalk", "flat_other", "flat_terrain", "static_other", "static_manmade", "static_vegetation", "vehicle_ego"]
    # ds1 = Sampler(pc1_load, labels1, remove_classes=remove_classes, dataset="nuscenes")
    # ds2 = Sampler(pc2_load, labels2, remove_classes=remove_classes, dataset="nuscenes")

    ## Voxelize the pointclouds
    voxel1, _, _ = ds1.adaptive_voxelization()
    voxel2, _, _ = ds2.adaptive_voxelization()

    # print("Voxel1 Shape: ", voxel1.shape)
    # print("Voxel2 Shape: ", voxel2.shape)

    ## Pad to a general size of 8192
    number_of_points = 8192
    if voxel1.shape[0] < number_of_points:
        voxel1 = np.pad(voxel1, ((0,number_of_points-voxel1.shape[0]), (0,0)), mode="constant")
        
    if voxel2.shape[0] < number_of_points:
        voxel2 = np.pad(voxel2, ((0,number_of_points-voxel2.shape[0]), (0,0)), mode="constant")

    # print("Voxel1 Padded: ", voxel1.shape)
    # print("Voxel2 Padded: ", voxel2.shape)
    test1 = voxel1
    test2 = voxel2

    ## Prepare for torch
    voxel1 = torch.from_numpy(voxel1).unsqueeze(0).cuda().float()
    voxel2 = torch.from_numpy(voxel2).unsqueeze(0).cuda().float()

    ## Make inference
    path = model
    model_module_params = {"learning_rate": 0.000005}
    model = PointConvSceneFlowPWC8192selfglobalPointConv.load_from_checkpoint(path).cuda()
    flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2 = model.predict(voxel1, voxel2)
    print("Inference Done")

    ## The Flow applied on the 8192 pointcloud
    pc_3 = np.transpose((pc1[0].cpu().detach().numpy() + flows[0].cpu().detach().numpy()).squeeze(0))

    ## Prepare for generating .ply
    ## The original Pointcloud1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1_load[:, :3])

    ## The original Pointcloud2
    pcd2 = o3d.geometry.PointCloud()    
    pcd2.points = o3d.utility.Vector3dVector(pc2_load[:, :3])

    ## Only the Flow for the masked areas
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pc_3)

    ## Upsample the flow, and apply masked upsampled to the original of Pointcloud1
    upsampled_flow = ds1.upsample(flows[0].cpu().detach().numpy().squeeze(0).T)
    upsampled_transformed = pc1_load[:, 0:3] + upsampled_flow
    print("Upsampled transformed Shape: ", upsampled_transformed.shape)

    # Get mask of where we have flow
    mask = np.where(upsampled_flow.any(axis=1))[0]
    # Make the rigid transformation
    rigid = rigid_transformation(data_dir, 8, [frame, frame+1])
    
    
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(rigid)
    
    ## Replace the rigid with our flow 
    rigid[mask] = upsampled_transformed[mask]
    # print("Rigid Shape: ", rigid.shape)

    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(rigid)

    print(pc2_load[:,:3].shape)
    print(rigid.shape)

    dist1, dist2 = computeChamfer(pc2[0], pc1[0] + flows[0])
    chamfer_loss = (dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()) / (2 * mask.shape[0])
    chamfer_loss = chamfer_loss.cpu().detach().numpy()
    print(chamfer_loss)

    if not os.path.isdir("../_output/pointclouds"):
        os.makedirs("../_output/pointclouds")

    if write:
        ts = str(time.time())
        o3d.io.write_point_cloud("../_output/pointclouds/pc1_%s.ply" % ts, pcd1)
        o3d.io.write_point_cloud("../_output/pointclouds/pc2_%s.ply" % ts, pcd2)
        o3d.io.write_point_cloud("../_output/pointclouds/flow8192_%s.ply" % ts, pcd3)
        o3d.io.write_point_cloud("../_output/pointclouds/rigid_%s.ply" % ts, pcd4)
        o3d.io.write_point_cloud("../_output/pointclouds/rigid_and_flow_%s.ply" % ts, pcd5)
        print("Created Pointclouds")

    ## write binary
    if not os.path.isdir(os.path.join(output, "sequences/08/velodyne")):
        os.makedirs(os.path.join(output, "sequences/08/velodyne"))
        os.makedirs(os.path.join(output, "sequences/08/labels"))
        os.makedirs(os.path.join(output, "sequences/08/sizes"))

    
    
    # path_to_write = "/media/niklas/Extreme SSD/rigid_flow_dataset/sequences/08/velodyne/%s_%s.%s" % (frame1, frame2, "bin")
    # all_points = np.vstack((rigid, rigid))
    # print(all_points.shape)
    # all_points.astype("float32").tofile(path_to_write)
    # path_to_write = "/media/niklas/Extreme SSD/rigid_flow_dataset/sequences/08/sizes/%s_%s.%s" % (frame1, frame2, "size")
    # sizes = np.array([rigid.shape[0], rigid.shape[0]])
    # sizes.astype("float32").tofile(path_to_write)
    # path_to_write = "/media/niklas/Extreme SSD/rigid_flow_dataset/sequences/08/labels/%s_%s.%s" % (frame1, frame2, "label")
    # labels = np.vstack((labels1, labels1))
    # print(labels.shape)
    # labels.astype("int32").tofile(path_to_write)

    if frame == 0:
        path_to_write = os.path.join(output, "sequences/08/velodyne/%s_orig.%s" % (frame1,  "bin"))
        pc1_load[:,:3].astype("float32").tofile(path_to_write)

        path_to_write = os.path.join(output, "sequences/08/labels/%s_orig.%s" % (frame1,  "label"))
        labels1 = np.fromfile(filename.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
        labels1.astype("int32").tofile(path_to_write)

    path_to_write = os.path.join(output, "sequences/08/velodyne/%s_prime.%s" % (frame2,  "bin"))
    rigid.astype("float32").tofile(path_to_write)
    
    path_to_write = os.path.join(output, "sequences/08/velodyne/%s_orig.%s" % (frame2, "bin"))
    pc2_load[:,:3].astype("float32").tofile(path_to_write)
    

    path_to_write = os.path.join(output, "sequences/08/labels/%s_prime.%s" % (frame2, "label"))
    labels1 = np.fromfile(filename.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
    labels1.astype("int32").tofile(path_to_write)


    path_to_write = os.path.join(output, "sequences/08/labels/%s_orig.%s" % (frame2, "label"))
    labels2 = np.fromfile(filename2.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
    labels2.astype("int32").tofile(path_to_write)
    

    return rigid, chamfer_loss

if __name__ == "__main__":
    try:
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '-dd', '--data_dir',
            metavar="P",
            default='/media/niklas/Extreme SSD/semanticKitti/dataset',
            help='path to the semantic kitti dataset')
        argparser.add_argument(
            '-seq', '--sequence',
            metavar='S',
            default=8,
            help='Which sequence to use')
        argparser.add_argument(
            '-f', '--frame',
            metavar='E',
            default=26,
            help='the first frame')
        argparser.add_argument(
            '-o', '--output',
            default="../_output/flow_dataset",
            help='where to output the flow dataset')
        argparser.add_argument(
            '-m', '--model',
            default="pretrained_models/semantic_kitti_and_nuscenes_model/checkpoints/NN-OutputKitti_and_NuScenes_1662110673.7257555/PWC-epoch=0017-Overall_Val=160.81707764.ckpt",
            help='Path to the model ckpt')
        argparser.add_argument(
            '-w', '--write',
            default=True,
            help='If pointclouds will be safed')
        args = argparser.parse_args()
        
        path = os.path.join(args.data_dir, f'sequences/{args.sequence:02d}/velodyne/')
        _, _, files = next(os.walk(path))
        loss_list = []
        for i in tqdm(range(len(files))):
            print("Current Iteration: ", i)
            rigid, chamfer_loss = flow_inference(args.data_dir, args.output, args.sequence, i, args.model, args.write)
            loss_list.append(chamfer_loss)

        print("+++++ Evaluation +++++")
        print("Mean Scaled Chamfer Loss: ", np.mean(np.array(loss_list)))
        print("ACC (0.05): ", sum(i < 0.05 for i in loss_list) / len(loss_list))
        print("ACC (0.1): ", sum(i < 0.1 for i in loss_list) / len(loss_list))

    except Exception as e: 
        print(traceback.format_exc())