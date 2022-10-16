
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from point_pwc.pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from point_pwc.pointconv_util import SceneFlowEstimatorPointConv
from point_pwc.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time

from point_pwc.loss import multiScaleChamferSmoothCurvature, get_cycle_loss_pytorch
scale = 1.0

class PointConvSceneFlowPWC8192selfglobalPointConv(nn.Module):
    def __init__(self):
        super(PointConvSceneFlowPWC8192selfglobalPointConv, self).__init__()
        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 16)
        self.level0_1 = Conv1d(16, 16)
        #self.cost0 = PointConvFlow(flow_nei, 32 + 32 + 32 + 32 + 3, [32, 32])
        self.cost0 = PointConvFlow(flow_nei, 96 + 3, [16, 16])
        #self.flow0 = SceneFlowEstimatorPointConv(32 + 64, 32)
        self.flow0 = SceneFlowEstimatorPointConv(16 + 64, 16)
        #self.level0_2 = Conv1d(32, 64)
        self.level0_2 = Conv1d(16, 32)
        
        #l2: 512
        #self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.level2 = PointConvD(1536, feat_nei, 32 + 3, 64)
        #self.cost2 = PointConvFlow(flow_nei, 128 + 64 + 128 + 64 + 3, [128, 128])
        self.cost2 = PointConvFlow(flow_nei, 195, [64, 64])
        #self.flow2 = SceneFlowEstimatorPointConv(128 + 64, 64)
        self.flow2 = SceneFlowEstimatorPointConv(64 + 64, 64)
        self.level2_0 = Conv1d(64, 64)
        self.level2_1 = Conv1d(64, 128)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 128 + 3, 128)
        #self.cost3 = PointConvFlow(flow_nei, 256 + 64 + 256 + 64 + 3, [256, 256])
        self.cost3 = PointConvFlow(flow_nei, 323, [128, 128])
        self.flow3 = SceneFlowEstimatorPointConv(128, 128, flow_ch=0)
        self.level3_0 = Conv1d(128, 128)
        self.level3_1 = Conv1d(128, 128)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 128 + 3, 128)

        #deconv
        self.deconv4_3 = Conv1d(128, 32)
        self.deconv3_2 = Conv1d(128, 32)
        self.deconv2_1 = Conv1d(64, 32)
        self.deconv1_0 = Conv1d(32, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N
        feat1_l0 = self.level0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0(color2)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

       

        #l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l0, feat1_l0_1)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l0, feat2_l0_1)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)

        #l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        #l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        #l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cost3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)


        feat1_l2_1 = self.upsample(pc1_l0, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l0, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)


        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

       


        #c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        #c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

       

       
        c_feat1_l0 = torch.cat([feat1_l0, feat1_l2_1], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l2_1], dim = 1)

        #l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2)

       
        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l2, self.scale * flow2)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l2, feat2)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0)

        flows = [flow0, flow2, flow3]
        pc1 = [pc1_l0,  pc1_l2, pc1_l3]
        pc2 = [pc2_l0,  pc2_l2, pc2_l3]
        fps_pc1_idxs = [ fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2


if __name__ == "__main__":

    import time

    num_points = 8192
    pc1 = torch.rand(1, num_points, 3)
    pc2 = torch.rand(1, num_points, 3)
    xyz1 = pc1.cuda()
    xyz2 = pc2.cuda()
    color1 = pc1.cuda()
    color2 = pc2.cuda()

    #gt_flow = torch.rand(1, num_points, 3).cuda()
    mask1 = torch.ones(1, num_points, dtype = torch.bool).cuda()
    model = PointConvSceneFlowPWC8192selfglobalPointConv().cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #Original parameter count: 7722284 
    print('Total params:', pytorch_total_params)
    print('Total trainable params:', pytorch_total_params_trainable)
    model.eval()
    for _ in range(1):
        with torch.no_grad():
            flows12, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2 = model(xyz1, xyz2, color1, color2)
            flows21, fps_pc2_idxs, fps_pc1_idxs, pc2, pc1 = model(xyz2, xyz1, color2, color1)
            torch.cuda.synchronize()

    #loss = multiScaleLoss(flows, gt_flow, fps_pc1_idxs)

    self_loss_pwc = multiScaleChamferSmoothCurvature(pc1, pc2, flows12)
    self_loss_jgwtf = get_cycle_loss_pytorch(pc1[0], flows12[0], pc2[0], flows21[0])
   # print(flows[0].shape, loss)
    print(self_loss_pwc[0], self_loss_jgwtf)
    
