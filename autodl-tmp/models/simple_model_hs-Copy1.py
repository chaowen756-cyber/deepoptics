# models/simple_model_hs.py

import torch
import torch.nn as nn

from models.outputs_container import OutputsContainer
from nets.unet import UNet


class SimpleModelHS(nn.Module):
    """
    一个为高光谱数据修改过的简单U-Net解码器模型。
    """

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse

        # ####################################################################
        # ## 核心修改点 1: 从 hparams 获取通道数，而不是硬编码 ##
        # ####################################################################
        hs_channels = hparams.hs_channels  # 获取高光谱通道数
        depth_ch = 1  # 深度图通道数固定为1

        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = hparams.model_base_ch

        # 计算输入层的通道数
        # 输入包括：模糊的HS图像(hs_channels) + 物理模型的伪逆卷(hs_channels * n_depths)
        preinv_input_ch = hs_channels * n_depths + hs_channels
        
        
        base_input_layers = nn.Sequential(
            # --- 修改开始 ---
            # 立即将通道数从 preinv_input_ch (493) 降到 base_ch (32)
            nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            # 保持U-Net结构需要的另一个卷积块 (输入和输出通道都是 base_ch)
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            # --- 修改结束 ---
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

#         base_input_layers = nn.Sequential(
#             nn.Conv2d(preinv_input_ch, preinv_input_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(preinv_input_ch),
#             nn.ReLU(),
#             nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(base_ch),
#             nn.ReLU(),
#         )


        if self.preinverse:
            input_layers = base_input_layers
        else:
            # 如果不使用伪逆，则输入只有模糊的HS图像
            input_layers = nn.Sequential(
                nn.Conv2d(hs_channels, preinv_input_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(preinv_input_ch),
                nn.ReLU(),
                base_input_layers,
            )

        # ####################################################################
        # ## 核心修改点 2: 修改输出层的通道数 ##
        # ####################################################################
        # 输出包括：重建的HS图像(hs_channels) + 深度图(depth_ch)
        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, hs_channels + depth_ch, kernel_size=1, bias=True)
        )

#         self.decoder = nn.Sequential(
#             input_layers,
#             UNet(
#                 # UNet的结构保持不变，只处理特征通道
#                 channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
#                 n_layers=n_layers,
#             ),
#             output_layers,
#         )
        # U-Net 的主体 (没有最后的输出层)
        self.unet_body = UNet(
            channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
            n_layers=n_layers,
        )

        # --- 关键修复：创建两个独立的“头” ---
        
        # 头 1: 专门用于深度
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid() 
        )
        
        # 头 2: 专门用于高光谱图像
        self.image_head = nn.Sequential(
            nn.Conv2d(base_ch, hs_channels, kernel_size=1, bias=True),
            nn.Sigmoid() 
        )
        # --- 修复结束 ---
        
        self.decoder = nn.Sequential(
            input_layers,
            self.unet_body  # <-- 注意：这里只保留 unet_body
        )
        

        # 权重初始化部分保持不变
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, pinv_volumes, *args, **kargs):
        b_sz, c_sz, d_sz, h_sz, w_sz = pinv_volumes.shape

        if self.preinverse:
            # 将模糊图像和伪逆结果在深度维度上拼接
            inputs = torch.cat([captimgs.unsqueeze(2), pinv_volumes], dim=2)
        else:
            inputs = captimgs.unsqueeze(2)
       
        # 将输入reshape成 (B, C*D, H, W) 的形式送入2D CNN
        # 并通过sigmoid函数将输出限制在 [0, 1] 范围
#         est = torch.sigmoid(self.decoder(inputs.reshape(b_sz, -1, h_sz, w_sz)))

#         # ####################################################################
#         # ## 核心修改点 3: 分割输出的逻辑保持不变，但意义已更新 ##
#         # ####################################################################
#         # est[:, :-1] 现在是 (B, hs_channels, H, W) 的高光谱图像
#         est_images = est[:, :-1]
#         # est[:, [-1]] 依然是 (B, 1, H, W) 的深度图
#         est_depthmaps = est[:, [-1]]
        # 1. 得到共享的特征
        features = self.decoder(inputs.reshape(b_sz, -1, h_sz, w_sz))
        
        # --- 关键修复：将特征分别送入两个“头” ---
        est_depthmaps = self.depth_head(features)
        est_images = self.image_head(features)
        # --- 修复结束 ---

        outputs = OutputsContainer(
            est_images=est_images,
            est_depthmaps=est_depthmaps,
        )
        return outputs