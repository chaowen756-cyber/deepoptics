# generic_transform.py

from typing import Tuple
import torch
import torch.nn as nn
import kornia.augmentation as K


class GenericRandomTransform(nn.Module):
    """
    一个通用的变换模块，可以对两个张量（如高光谱图像和深度图）
    施加完全相同的几何变换。
    """

    def __init__(self, size: Tuple[int, int], randcrop: bool, augment: bool, image_channels: int):
        """
        构造函数。

        Args:
            size (Tuple[int, int]): 裁剪后的目标尺寸 (height, width)。
            randcrop (bool): 是否进行随机裁剪 (True) 或中心裁剪 (False)。
            augment (bool): 是否进行随机翻转。
            image_channels (int): 第一个输入张量（例如高光谱图像）的通道数。
        """
        super().__init__()
        if randcrop:
            self.crop = K.RandomCrop(size)
        else:
            self.crop = K.CenterCrop(size)

        self.flip = nn.Sequential(
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5)
        )
        self.augment = augment
        self.image_channels = image_channels
#         self.crop = K.CenterCrop(size)
        
#         self.flip = nn.Sequential(
#             K.RandomVerticalFlip(p=0.5),
#             K.RandomHorizontalFlip(p=0.5)
#         )
#         self.augment = augment
#         self.image_channels = image_channels

    def forward(self, image_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行变换的前向传播。
        """
        # 1. 核心步骤：在通道维度上拼接，以确保变换一致 0-》1
        combined_input = torch.cat([image_tensor, depth_tensor], dim=1)

        # 2. 应用裁剪
        combined_input = self.crop(combined_input)

        # 3. 如果启用增强，则应用翻转
        if self.augment:
            combined_input = self.flip(combined_input)

        # 4. 在通道维度分割回原始部分
        image_out = combined_input[:, :self.image_channels, :, :]
        depth_out = combined_input[:, self.image_channels:, :, :]
#         # 调试信息
#         print(f"[Transform] Input: img={image_tensor.shape}, depth={depth_tensor.shape}")
#         print(f"[Transform] Combined: {combined_input.shape}")
#         print(f"[Transform] Output: img={image_out.shape}, depth={depth_out.shape}")
#         print(f"[Transform] Depth range: [{depth_out.min():.4f}, {depth_out.max():.4f}]")

        return image_out, depth_out
# import torch
# import torch.nn as nn
# import kornia.augmentation as K
# import random
# from typing import Tuple

# class GenericRandomTransform(nn.Module):
#     """
#     一个通用的变换模块，可以对两个张量（如高光谱图像和深度图）
#     施加完全相同的几何变换。
#     集成“智能锚点裁剪”，保证裁剪区域包含有效深度信息。
#     """

#     def __init__(self, size: Tuple[int, int], randcrop: bool, augment: bool, image_channels: int):
#         """
#         构造函数。
#         Args:
#             size (Tuple[int, int]): 裁剪后的目标尺寸 (height, width)。
#             randcrop (bool): 是否进行随机裁剪 (True) 或中心裁剪 (False)。
#             augment (bool): 是否进行随机翻转。
#             image_channels (int): 第一个输入张量（例如高光谱图像）的通道数。
#         """
#         super().__init__()
#         self.size = size
#         self.randcrop = randcrop
        
#         # 如果不是随机裁剪，保留 Kornia 的 CenterCrop 用于验证集
#         if not randcrop:
#             self.crop = K.CenterCrop(size)
#         else:
#             # 训练集我们将使用自定义的智能裁剪，不需要 Kornia 的 RandomCrop
#             self.crop = None 

#         self.flip = nn.Sequential(
#             K.RandomVerticalFlip(p=0.5),
#             K.RandomHorizontalFlip(p=0.5)
#         )
#         self.augment = augment
#         self.image_channels = image_channels

#     def _smart_slice(self, tensor: torch.Tensor, depth_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         内部辅助函数：计算基于深度图有效像素的裁剪坐标，并执行切片。
#         不再盲目随机，而是“指哪打哪”。
#         """
#         # 获取维度 (Batch, Channel, Height, Width)
#         B, C, H, W = tensor.shape
#         crop_h, crop_w = self.size
        
#         # 如果原图比裁剪尺寸小，直接返回（或者你可以选择Resize，这里默认原图够大）
#         if H < crop_h or W < crop_w:
#             return tensor

#         # 1. 寻找锚点 (Anchor)
#         # 取 Batch 中的第一个样本的深度图进行判断 (通常 Dataset 里的 B=1)
#         # depth_tensor 形状通常是 (B, 1, H, W)，取 [0, 0] 变成 (H, W)
#         d_map = depth_tensor[0, 0] 
        
#         # 找到所有大于 1e-6 的有效像素坐标
#         valid_indices = torch.nonzero(d_map > 1e-6, as_tuple=False) # shape: [N, 2] -> (y, x)

#         if len(valid_indices) == 0:
#             # === 情况 A: 全黑/全背景 ===
#             # 退化为中心裁剪
#             top = (H - crop_h) // 2
#             left = (W - crop_w) // 2
#         else:
#             # === 情况 B: 有效 ===
#             # 随机选一个“锚点”
#             idx = random.randint(0, len(valid_indices) - 1)
#             anchor_y, anchor_x = valid_indices[idx]

#             # 2. 反算裁剪框左上角 (Top, Left) 的合法范围
#             # 逻辑：裁剪框必须包含锚点，且不能超出图像边界
            
#             # Top 的范围
#             min_top = max(0, anchor_y - crop_h + 1)
#             max_top = min(H - crop_h, anchor_y)
#             # 保护措施：防止 max < min
#             top = random.randint(min_top, max(min_top, max_top))

#             # Left 的范围
#             min_left = max(0, anchor_x - crop_w + 1)
#             max_left = min(W - crop_w, anchor_x)
#             left = random.randint(min_left, max(min_left, max_left))

#         # 3. 执行切片 (Slicing) - 这种方式比 K.crop 更快
#         # [Batch, Channel, Top:Bottom, Left:Right]
#         return tensor[..., top:top+crop_h, left:left+crop_w]

#     def forward(self, image_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         执行变换的前向传播。
#         """
#         # 1. 核心步骤：在通道维度上拼接，以确保变换一致
#         combined_input = torch.cat([image_tensor, depth_tensor], dim=1)

#         # 2. 应用裁剪 (修改逻辑)
#         if self.randcrop:
#             # 使用自定义的智能裁剪，传入 combined_input 和 原始 depth_tensor 用于定位
#             combined_input = self._smart_slice(combined_input, depth_tensor)
#         else:
#             # 验证集保持中心裁剪
#             combined_input = self.crop(combined_input)

#         # 3. 如果启用增强，则应用翻转 (逻辑不变)
#         if self.augment:
#             combined_input = self.flip(combined_input)

#         # 4. 在通道维度分割回原始部分
#         image_out = combined_input[:, :self.image_channels, :, :]
#         depth_out = combined_input[:, self.image_channels:, :, :]

#         # 调试信息 (保留你的代码习惯)
#         # print(f"[Transform] Depth range: [{depth_out.min():.4f}, {depth_out.max():.4f}]")

#         return image_out, depth_out