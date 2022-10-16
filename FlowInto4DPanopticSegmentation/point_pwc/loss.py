import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from pointconv_util import SceneFlowEstimatorPointConv
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import argparse

def get_cycle_loss_pytorch(pc1:torch.FloatTensor, flow12:torch.FloatTensor, pc2:torch.FloatTensor, flow21:torch.FloatTensor, knn_loss_weight:float=1, cycle_loss_weight:float=1) -> float:
    """
    @brief estimates the cycle loss. It applies the flowfield forwards and backwards and estimates the offset. 
           This ensures that the network does not develop a bias towards directions.

    @type pc1: torch.FloatTensor [S, B, N, 3]
    @param pc1: The pointcloud of the first frame 

    @type flow12: torch.FloatTensor [S, B, N, 3]
    @param flow12: The forward flow from the first to the second frame

    @type pc2: torch.FloatTensor [S, B, M, 3] 
    @param pc2: The pointcloud of the second frame

    @type flow21: torch.FloatTensor [S, B, M, 3]
    @param flow21: The backward flow from the second to the first frame

    @type knn_loss_weight: int 
    @key knn_loss_weight: 1 
    @param knn_loss_weight: The weighting factor of the knn loss 

    @type cycle_loss_weight: int 
    @key cycle_loss_weight: 1 
    @param cycle_loss_weight: The weighting factor of the cycle loss 

    @type loss: float
    @return loss: float
    """


    scale = len(flow12[0])
    overall_loss = 0
    alpha = [0.02, 0.04, 0.08, 0.16]
    for i in range(scale):
        ## make forward prediction pc2_hat (in optimal case min||pc2 - pc2_hat|| --> 0)
        fwd_pred = pc1[i] + flow12[i] 

        ## Get the grouped pc 
        sqrdist = square_distance(pc2[i], fwd_pred)
        _, kidx = torch.topk(sqrdist, 1, dim=-1, largest=False, sorted=False) ## only check one NN as in the paper
        grouped_pc = index_points_group(pc2[i], kidx)
        grouped_pc = torch.squeeze(grouped_pc, axis=2)

        ## make back prediction
        pred_fb = (fwd_pred + grouped_pc) / 2
        back_pred = flow21[i] + pred_fb

        ## estimate knn l2 loss weight
        knn_l2_loss = alpha[i] * knn_loss_weight*torch.mean(
            torch.sum((fwd_pred - grouped_pc) * (fwd_pred - grouped_pc), axis=2) / 2.0)
        
        cycle_l2_loss = alpha[i] * cycle_loss_weight*torch.mean(
            torch.sum((back_pred - pc1[i]) * (back_pred - pc1[i]), axis=2) / 2.0)

        overall_loss += knn_l2_loss + cycle_l2_loss
    return overall_loss


def curvature(pc:torch.FloatTensor) -> torch.FloatTensor:
    """
    @brief: estimates the curvature of the given pointcloud by looking at the closest 10 neighbors. 

    @type pc: torch.FloatTensor, device='cuda:0'
    @param pc: A pointcloud tensor on the gpu

    @type pc_curvature: torch.FloatTensor [B, N, 3]
    @return pc_curvature: The curvature tensor with a curvature value for 
                          each point based on its 10 nearest neighbors
    """
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx) # [B, N, K, C] -> batch, number, neighbors, axes
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1:torch.FloatTensor, pc2:torch.FloatTensor) -> torch.FloatTensor:
    '''
    @brief: The Chamfer distance is computed by summing the squared distances
            between nearest neighbor correspondences of two point clouds
    
    @type pc1: torch.tensor [B, 3, N]
    @param pc1: The first pointcloud

    @type pc2: torch.tensor [B, 3, M]
    @param pc2: The second pointlcoud

    @type dist1: torch.FloatTensor [1, N]
    @return dist1: The distance tensor for pointlcoud1

    @type dist2: torch.FloatTensor [1, M]
    @return dist2: The distance tensor for pointcloud2


    '''
    ## Change axis to [B, N, 3]
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M


    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc:torch.FloatTensor, warped_pc:torch.FloatTensor) -> torch.FloatTensor:
    """
    @brief: estimates the curvature of the given warped pointcloud by looking 
            at the closest 10 neighbors of the original pointcloud. 

    @type pc: torch.FloatTensor, device='cuda:0'
    @param pc: A pointcloud tensor on the gpu

    @type warped_pc: torch.FloatTensor, device='cuda:0'
    @param warped_pc: The warped pointcloud to check the curvature for

    @type pc_curvature: torch.FloatTensor [B, N, 3]
    @return pc_curvature: The curvature tensor with a curvature value for 
                          each point based on its 10 nearest neighbors
    """
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1:torch.FloatTensor, pred_flow:torch.FloatTensor) -> torch.FloatTensor:
    """
    @brief: estimates the curvature of the given warped pointcloud by looking 
            at the closest 10 neighbors of the original pointcloud. 

    @type pc: torch.FloatTensor  [B, 3, N]
    @param pc: A pointcloud tensor

    @type pred_flow: torch.FloatTensor [B, 3, N]
    @param pred_flow: The predicted flow tensor

    @type diff_flow: torch.FloatTensor [B, N, 3]
    @return diff_flow: the difference/smoothness of the flow prediction
    """

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1:torch.FloatTensor, pc2:torch.FloatTensor, pc2_curvature:torch.FloatTensor) -> torch.FloatTensor:
    """
    @brief: Interpolates the curvature from pointcloud1 to pointcloud2

    @type pc_1: torch.FloatTensor  [B, 3, N]
    @param pc_1: First pointcloud tensor

    @type pc_2: torch.FloatTensor  [B, 3, M]
    @param pc_2: The second pointcloud tensor

    @type pred_flow: torch.FloatTensor [B, 3, M]
    @param pred_flow: The curvature tensor of the second pointcloud

    @type inter_pc2_curvature: torch.FloatTensor [B, N, 3]
    @return inter_pc2_curvature: the interpolated pointcloud curvature tensor
    """

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False) # get 5 NN of the other pointcloud
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    # normalize over the neighbors
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm
    # sum over the weighted neighbors
    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1:torch.FloatTensor, pc2:torch.FloatTensor, pred_flows:torch.FloatTensor):
    """
    @brief: 

    @type pc1: tensor.FloatTensor [S, B, 3, N]
    @param pc1: Pointcloud1 

    @type pc2: tensor.FloatTensor [S, B, 3, M]
    @param pc2: Pointcloud 2 

    @type pred_flows: tensor.FloatTensor 
    @param pre_flows: The multiscalt flow prediction
    """
    ## Init the weight for the losses
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0

    #num of scale
    num_scale = len(pred_flows)

    ## Define multiscale weights
    alpha = [0.02, 0.04, 0.08, 0.16]
    
    ## Init Loss matrices 
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N

        #compute curvature
        cur_pc2_curvature = curvature(cur_pc2)

        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

        #smoothness
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss

    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss


if __name__ == "__main__":
    """Check out some loss caluclations and ensure they work"""

    # Args
    parser = argparse.ArgumentParser(description='sample points')
    parser.add_argument('--seq', type=int, default=2, help='used sequence')
    parser.add_argument('--frame', type=int, default=70, help='used fram')
    args = parser.parse_args()

    print("\n-----Start Loss Example:-----\n sequence:%s \n frame:%s \n" % (args.seq, args.frame))

    sequence = args.seq
    sequence_text = f'{sequence:02d}'
    scannum = args.frame
    scannum_next = scannum + 1
    scannum_text = f'{scannum:06d}'
    scannum_next_text =   f'{scannum_next:06d}'

    ## Get two example pointclouds
    filename = f'/storage/remote/atcremers61/s0100/ps4d/dataset/sequences/{sequence_text}/velodyne/{scannum_text}.bin'
    filename2 = f'/storage/remote/atcremers61/s0100/ps4d/dataset/sequences/{sequence_text}/velodyne/{scannum_next_text}.bin'

    ## Extract pointcloud 
    pc1 = np.fromfile(filename,dtype=np.float32).reshape((-1, 4))[:, 0:3]
    pc2 = np.fromfile(filename2, dtype=np.float32).reshape((-1, 4))[:, 0:3]

    print(pc1.shape)

    ## Get some points 
    number_of_points = 8192
        
    idx1 = np.random.permutation(pc1.shape[0])[0:8192]
    idx2 = np.random.permutation(pc2.shape[0])[0:9000]
    pc1 = pc1[idx1, :]
    pc2 = pc2[idx2, :]
    ## Adapt to tensor format
    pc1 = torch.from_numpy(pc1).unsqueeze(0).permute(0,2,1)
    pc2 = torch.from_numpy(pc2).unsqueeze(0).permute(0,2,1)
   
    dist1, dist2 = computeChamfer(pc1, pc2)
    print(dist1.shape)
    print(dist2.shape)
    curv = curvature(pc1.cuda())


