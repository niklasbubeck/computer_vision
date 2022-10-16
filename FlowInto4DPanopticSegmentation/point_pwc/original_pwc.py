"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from point_pwc.pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from point_pwc.pointconv_util import SceneFlowEstimatorPointConv
from point_pwc.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time
import pytorch_lightning as pl
from point_pwc.loss import multiScaleChamferSmoothCurvature, get_cycle_loss_pytorch


scale = 1.0

class PointConvSceneFlowPWC8192selfglobalPointConv(pl.LightningModule):
    def __init__(self, model_module_params):
        super(PointConvSceneFlowPWC8192selfglobalPointConv, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = model_module_params["learning_rate"]

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        self.cost0 = PointConvFlow(flow_nei, 32 + 32 + 32 + 32 + 3, [32, 32])
        self.flow0 = SceneFlowEstimatorPointConv(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cost1 = PointConvFlow(flow_nei, 64 + 32 + 64 + 32 + 3, [64, 64])
        self.flow1 = SceneFlowEstimatorPointConv(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cost2 = PointConvFlow(flow_nei, 128 + 64 + 128 + 64 + 3, [128, 128])
        self.flow2 = SceneFlowEstimatorPointConv(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cost3 = PointConvFlow(flow_nei, 256 + 64 + 256 + 64 + 3, [256, 256])
        self.flow3 = SceneFlowEstimatorPointConv(256, 256, flow_ch=0)
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        #deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

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

        #l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)

        #l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
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

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        #l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2)

        #l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cost1, up_flow1)

        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cost0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2

    def training_step(self, batch, batch_idx):
        xyz1, xyz2 = batch
        flows12, fps_pc1_idxs, fps_pc2_idxs, pc11, pc21 = self(xyz1, xyz2, xyz1, xyz2)
        # flows21, fps_pc1_idxs, fps_pc2_idxs, pc12, pc22 = self(xyz2, xyz1, xyz2, xyz1)
        # cycle_loss = get_cycle_loss_pytorch(pc11, flows12, pc21, flows21, 0.1, 0.1) + get_cycle_loss_pytorch(pc12, flows21, pc22, flows12, 0.1, 0.1)
        total_loss_f, chamfer, curvature, smooth = multiScaleChamferSmoothCurvature(pc11, pc21, flows12)  
        # total_loss_b, _, _, _ = multiScaleChamferSmoothCurvature(pc12, pc22, flows21)
        # total_loss = total_loss_b + total_loss_f
        loss = total_loss_f
        logs = {"Loss": {"Train_overall":loss, "Train_total": total_loss_f, "Train_chamfer": chamfer, "Train_curvature": curvature, "Train_smooth":smooth}, "Overall_Train":loss}
        self.log_dict(logs, prog_bar=False, logger=True, on_epoch=True, on_step=False, reduce_fx='mean')
        return loss

    def validation_step(self, batch, batch_idx):
        xyz1, xyz2 = batch
        flows12, fps_pc1_idxs, fps_pc2_idxs, pc11, pc21 = self(xyz1, xyz2, xyz1, xyz2)
        # flows21, fps_pc1_idxs, fps_pc2_idxs, pc12, pc22 = self(xyz2, xyz1, xyz2, xyz1)
        # cycle_loss = get_cycle_loss_pytorch(pc11, flows12, pc21, flows21, 0.1, 0.1) + get_cycle_loss_pytorch(pc12, flows21, pc22, flows12, 0.1, 0.1)
        total_loss_f, chamfer, curvature, smooth = multiScaleChamferSmoothCurvature(pc11, pc21, flows12)  
        # total_loss_b, _, _, _ = multiScaleChamferSmoothCurvature(pc12, pc22, flows21)
        # total_loss = total_loss_b + total_loss_f
        loss = total_loss_f
        logs = {"Loss": {"Val_overall":loss,"Val_total": total_loss_f, "Val_chamfer": chamfer, "Val_curvature": curvature, "Val_smooth": smooth}, "Overall_Val":loss}
        self.log_dict(logs, prog_bar=False, logger=True, on_epoch=True, on_step=False, reduce_fx='mean')
        return loss

    def predict(self, xyz1, xyz2):
        self.eval()
        with torch.no_grad():
            flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2 = self(xyz1, xyz2, xyz1, xyz2)
        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'