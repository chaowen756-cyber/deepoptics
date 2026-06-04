# # models/simple_model_hs.py

# import torch
# import torch.nn as nn

# from models.outputs_container import OutputsContainer
# from nets.unet import UNet


# class SimpleModelHS(nn.Module):
#     """
#     一个为高光谱数据修改过的简单U-Net解码器模型。
#     """

#     def __init__(self, hparams, *args, **kargs):
#         super().__init__()
#         self.preinverse = hparams.preinverse

#         # ####################################################################
#         # ## 核心修改点 1: 从 hparams 获取通道数，而不是硬编码 ##
#         # ####################################################################
#         hs_channels = hparams.hs_channels  # 获取高光谱通道数
#         depth_ch = 1  # 深度图通道数固定为1

#         n_layers = 4
#         n_depths = hparams.n_depths
#         base_ch = hparams.model_base_ch

#         # 计算输入层的通道数
#         # 输入包括：模糊的HS图像(hs_channels) + 物理模型的伪逆卷(hs_channels * n_depths)
#         preinv_input_ch = hs_channels * n_depths + hs_channels
        
        
#         base_input_layers = nn.Sequential(
#             # --- 修改开始 ---
#             # 立即将通道数从 preinv_input_ch (493) 降到 base_ch (32)
#             nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(base_ch),
#             nn.ReLU(),
#             # 保持U-Net结构需要的另一个卷积块 (输入和输出通道都是 base_ch)
#             nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
#             # --- 修改结束 ---
#             nn.BatchNorm2d(base_ch),
#             nn.ReLU(),
#         )

# #         base_input_layers = nn.Sequential(
# #             nn.Conv2d(preinv_input_ch, preinv_input_ch, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(preinv_input_ch),
# #             nn.ReLU(),
# #             nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(base_ch),
# #             nn.ReLU(),
# #         )


#         if self.preinverse:
#             input_layers = base_input_layers
#         else:
#             # 不使用伪逆，直接处理hs_channels
#             input_layers = nn.Sequential(
#                 nn.Conv2d(hs_channels, base_ch, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(base_ch),
#                 nn.ReLU(),
#                 nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(base_ch),
#                 nn.ReLU(),
#             )

#         # ####################################################################
#         # ## 核心修改点 2: 修改输出层的通道数 ##
#         # ####################################################################
#         # 输出包括：重建的HS图像(hs_channels) + 深度图(depth_ch)
#         output_layers = nn.Sequential(
#             nn.Conv2d(base_ch, hs_channels + depth_ch, kernel_size=1, bias=True)
#         )

# #         self.decoder = nn.Sequential(
# #             input_layers,
# #             UNet(
# #                 # UNet的结构保持不变，只处理特征通道
# #                 channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
# #                 n_layers=n_layers,
# #             ),
# #             output_layers,
# #         )
#         # U-Net 的主体 (没有最后的输出层)
#         self.unet_body = UNet(
#             channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
#             n_layers=n_layers,
#         )

#         # --- 关键修复：创建两个独立的“头” ---
        
#         # 头 1: 专门用于深度
#         self.depth_head = nn.Sequential(
#             nn.Conv2d(base_ch, 1, kernel_size=1, bias=True),
#             nn.Sigmoid() 
#         )
        
#         # 头 2: 专门用于高光谱图像
#         self.image_head = nn.Sequential(
#             nn.Conv2d(base_ch, hs_channels, kernel_size=1, bias=True),
#             nn.Sigmoid() 
#         )
#         # --- 修复结束 ---
        
#         self.decoder = nn.Sequential(
#             input_layers,
#             self.unet_body  # <-- 注意：这里只保留 unet_body
#         )
        

#         # 权重初始化部分保持不变
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, captimgs, pinv_volumes, *args, **kwargs):
#         b_sz = captimgs.shape[0]
#         h_sz, w_sz = captimgs.shape[2], captimgs.shape[3]
#         if self.preinverse:
#         # 根据论文：在通道维度拼接
#             if pinv_volumes.ndim == 4:
#             # (B, n_depths, H, W)
#                 pinv_reshaped = pinv_volumes
#             elif pinv_volumes.ndim == 5:
#             # (B, hs_channels, n_depths, H, W) → flatten to 4D
#                 _, c, d, _, _ = pinv_volumes.shape
#                 pinv_reshaped = pinv_volumes.reshape(b_sz, c*d, h_sz, w_sz)
#             else:
#                 raise ValueError(f"pinv_volumes 维度错误: {pinv_volumes.shape}")
        
#             # 论文中说的 channel-wise concatenation
#             inputs = torch.cat([captimgs, pinv_reshaped], dim=1)
#         else:
#             inputs = captimgs
#         # 通过共享特征提取器
#         features = self.decoder(inputs)  # (B, 32, H, W)
#         # 通过独立的输出头（改进点！）
#         est_images = self.image_head(features)      # (B, 29, H, W)
#         est_depthmaps = self.depth_head(features)   # (B, 1, H, W)
#         if est_depthmaps.ndim == 4 and est_depthmaps.shape[1] == 1:
#             est_depthmaps = est_depthmaps.squeeze(1)  # (B, 1, H, W) → (B, H, W)
#         outputs = OutputsContainer(
#             est_images=est_images,
#             est_depthmaps=est_depthmaps,
#         )
#         return outputs


# models/simple_model_hs.py

import torch
import torch.nn as nn

from models.outputs_container import OutputsContainer
# 假设你将上一步生成的双头Unet保存为了 nets.dual_head_unet
# 如果文件名不同，请修改这里的引用
from nets.unet import DualHeadUNet 

class SimpleModelHS(nn.Module):
    """
    修改后的SimpleModelHS，集成了符合结构图的双头U-Net (DualHeadUNet)。
    """

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse

        # ================= 参数配置 =================
        hs_channels = hparams.hs_channels  # 高光谱通道数 (25)
        depth_ch = 1  # 深度图通道数 (1)
        base_ch = hparams.model_base_ch # 基础通道数 (32)
        n_depths = hparams.n_depths
        
        # ================= 1. 适配层 (Stem) =================
        # 计算输入数据的总通道数
        # 输入包括：模糊的HS图像 + (可选)物理模型的伪逆卷
        preinv_input_ch = hs_channels * n_depths + hs_channels
        
        # base_input_layers 充当了结构图中最左侧的 "Input Sensor" 到 第一层特征的映射
        # 它负责将 493 维的厚数据 压缩成 32 维，作为 DualHeadUNet 的输入
        self.base_input_layers = nn.Sequential(
            nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        if self.preinverse:
            self.input_adapter = self.base_input_layers
        else:
            # 如果不使用伪逆，输入仅为 hs_channels，也需要映射到 base_ch
            self.input_adapter = nn.Sequential(
                nn.Conv2d(hs_channels, base_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(),
                nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(),
            )

        # ================= 2. 核心骨干 (DualHeadUNet) =================
        # 实例化刚才设计的双头 U-Net
        # 注意：这里的 in_channels 设置为 base_ch (32)，衔接上面的适配层输出
        # 我们假设 DualHeadUNet 已经支持传入 in_channels 参数
        self.backbone = DualHeadUNet(
            norm_layer=nn.BatchNorm2d
            # 如果你的 DualHeadUNet __init__ 还没有支持传参，
            # 请确保修改 DualHeadUNet 的 enc_channels[0] 为 base_ch (32)
        )
        
        # 强制修正 backbone 的第一层，以防 DualHeadUNet 默认是 3通道输入
        # 结构图中 Encoder 第一层是: Input -> 32。
        # 这里我们的 input_adapter 已经输出了 32，所以 backbone 的第一层应该是 32 -> 32
        # 或者，我们可以认为 input_adapter 就是结构图的第一层卷积。
        # 为了严谨对接，我们检查 backbone 的第一层输入通道：
        if hasattr(self.backbone, 'enc_channels'):
            # 确保 Encoder 通道列表是以 base_ch 开头
            self.backbone.enc_channels[0] = base_ch 
            # 重新初始化第一层下采样块 (覆盖默认的 3->32)
            self.backbone.downblocks[0] = self.backbone.downblocks[0].__class__(
                base_ch, 
                self.backbone.enc_channels[1], # 32 -> 32 (或其他配置)
                norm_layer=nn.BatchNorm2d
             )

        # ================= 3. 激活函数 =================
        # 之前的 DualHeadUNet 输出层没有激活函数，这里添加 Sigmoid 用于图像/深度重建
        self.sigmoid = nn.Sigmoid()

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, pinv_volumes, *args, **kwargs):
        b_sz = captimgs.shape[0]
        h_sz, w_sz = captimgs.shape[2], captimgs.shape[3]

        # --- 1. 数据预处理与拼接 ---
        if self.preinverse:
            if pinv_volumes.ndim == 4:
                pinv_reshaped = pinv_volumes
            elif pinv_volumes.ndim == 5:
                _, c, d, _, _ = pinv_volumes.shape
                pinv_reshaped = pinv_volumes.reshape(b_sz, c*d, h_sz, w_sz)
            else:
                raise ValueError(f"pinv_volumes 维度错误: {pinv_volumes.shape}")
            
            inputs = torch.cat([captimgs, pinv_reshaped], dim=1)
        else:
            inputs = captimgs

        # --- 2. 适配层 (降维到 base_ch) ---
        # Output: [Batch, 32, H, W]
        x = self.input_adapter(inputs)

        # --- 3. 双头 U-Net 处理 ---
        # DualHeadUNet 直接返回两个张量：depth_logits, hs_logits
        depth_logits, hs_logits = self.backbone(x)

        # --- 4. 激活与格式化 ---
        # 应用 Sigmoid 限制输出在 [0, 1] 之间
        est_depthmaps = self.sigmoid(depth_logits)
        est_images = self.sigmoid(hs_logits)

        # 调整深度图维度 (B, 1, H, W) -> (B, H, W)
        if est_depthmaps.ndim == 4 and est_depthmaps.shape[1] == 1:
            est_depthmaps = est_depthmaps.squeeze(1)

        # --- 5. 封装输出 ---
        outputs = OutputsContainer(
            est_images=est_images,
            est_depthmaps=est_depthmaps,
        )
        return outputs