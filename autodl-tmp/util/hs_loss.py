# util/hs_loss.py (最终修正版 - 使用 reshape)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    """
    光谱角映射损失函数 (Spectral Angle Mapper Loss)
    """

    def __init__(self, eps=1e-8):
        super(SAMLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        计算SAM损失
        """
        # ####################################################################
        # ## 核心修改点：使用 .reshape() 替代 .view() 以增加稳健性 ##
        # ####################################################################
        # 将图像展平为像素列表
        y_pred_flat = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)  # (B, C, H*W)
        y_true_flat = y_true.reshape(y_true.shape[0], y_true.shape[1], -1)  # (B, C, H*W)

        # 沿通道维度计算点积
        dot_product = torch.sum(y_pred_flat * y_true_flat, dim=1)

        # 计算每个向量的模长
        norm_pred = torch.linalg.norm(y_pred_flat, dim=1)
        norm_true = torch.linalg.norm(y_true_flat, dim=1)

        # 计算余弦角并处理数值不稳定的情况
        cos_angle = dot_product / (norm_pred * norm_true + self.eps)
        cos_angle = torch.clamp(cos_angle, -1, 1)

        # 计算角度（弧度）
        angle = torch.acos(cos_angle)

        # 返回所有像素角度的平均值
        return torch.mean(angle)


class CombinedLoss(nn.Module):
    """
    一个结合了多种损失函数的混合损失函数
    """

    def __init__(self, l1_weight=1.0, sam_weight=None):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.sam_loss = SAMLoss()
        self.l1_weight = l1_weight
        # self.sam_weight = sam_weight
        # print(f"初始化混合损失: L1 权重 = {l1_weight}, SAM 权重 = {sam_weight}")
        print(f"初始化损失: L1 权重 = {l1_weight}")

    def forward(self, y_pred, y_true):
        loss_l1 = self.l1_loss(y_pred, y_true)
        # loss_sam = self.sam_loss(y_pred, y_true)

        # total_loss = self.l1_weight * loss_l1 + self.sam_weight * loss_sam
        total_loss = self.l1_weight * loss_l1
        
        # 返回总损失以及各个分量，方便日志记录
        return total_loss, loss_l1, torch.tensor(0.0, device=y_pred.device)