#from pytorch3d.io import IO
import numpy as np
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Pointclouds
import torch
import os 


"""This is an exemplary script on how one can do the rigid body alignement"""


def main():
    
    #raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
    sequence = 2
    sequence_text = f'{sequence:02d}'
    scannum = 70
    scannum_next = scannum + 1
    scannum_text = f'{scannum:06d}'
    scannum_next_text =   f'{scannum_next:06d}'
    filename = f'/home/niklas/Documents/datasets/semanticKitti/dataset/sequences/{sequence_text}/velodyne/{scannum_text}.bin'
    filename2 = f'/home/niklas/Documents/datasets/semanticKitti/dataset/sequences/{sequence_text}/velodyne/{scannum_next_text}.bin'
    calib_filename = f'/home/niklas/Documents/datasets/semanticKitti/dataset/sequences/{sequence_text}/calib.txt'
    poses_filename = f'/home/niklas/Documents/datasets/semanticKitti/dataset/sequences/{sequence_text}/poses.txt'
    raw_data = np.fromfile(filename,dtype=np.float32).reshape(-1,4)
    next_raw_data = np.fromfile(filename2, dtype=np.float32).reshape((-1, 4))
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


    pose0 = poses[scannum]
    p_origin = np.zeros((1, 4))
    p_origin[0, 3] = 1

    p0 = p_origin.dot(pose0.T)[:, :3]
    p0 = np.squeeze(p0)
    points = raw_data[:, :3]
    print('First Scan Shape', points.shape)
    points_tensor = torch.FloatTensor(points).unsqueeze(0)
    point_cloud = Pointclouds(points=points_tensor)

    points_next = next_raw_data[:, :3]
    print('Next Scan Shape', points_next.shape)
    points_next_tensor = torch.FloatTensor(points_next).unsqueeze(0)
    point_cloud_next = Pointclouds(points=points_next_tensor)
    point_cloud_dict = {"Original": point_cloud,
    "Next PC": point_cloud_next,
     }
    fig = plot_scene({
    "Dataset": point_cloud_dict
}, point_cloud_max_points=300000)
    #fig.show()

    cur_pose = poses[scannum_next]
     # to global coor
    hpoints = np.hstack((next_raw_data[:, :3], np.ones_like(next_raw_data[:, :1])))
    new_points = np.sum(np.expand_dims(hpoints, 2) * cur_pose.T, axis=1)[:, :3]
    # to first frame coor
    new_coords = new_points - pose0[:3, 3]
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    points_next = new_coords
    merged_pts = np.vstack((points, points_next))
    merged_pts =  torch.FloatTensor(new_coords).unsqueeze(0)

    point_cloud_merged = Pointclouds(points=merged_pts)
    #point_cloud_dict = {"Merged PC": point_cloud_merged,}
    point_cloud_dict['Merged'] = point_cloud_merged
    fig = plot_scene({
    "Alignment Result": point_cloud_dict
}, point_cloud_max_points=300000)
    fig.show()    
if __name__ == '__main__':
    main()
