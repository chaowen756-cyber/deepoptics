# import os
# import re
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from typing import Tuple, List
# from kornia.augmentation import CenterCrop
# from .generic_transform import GenericRandomTransform
# from util.helper import metric_to_ips

# # 导入专用库来读取.exr文件
# import OpenEXR
# import Imath


# def read_exr(file_path):
#     """
#     使用OpenEXR库读取.exr文件并返回一个NumPy数组。
#     这个版本修复了通道名称的问题，使其更具通用性。
#     """
#     if not OpenEXR.isOpenExrFile(file_path):
#         raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

#     exr_file = OpenEXR.InputFile(file_path)
#     header = exr_file.header()

#     dw = header['dataWindow']
#     width = dw.max.x - dw.min.x + 1
#     height = dw.max.y - dw.min.y + 1

#     channels_info = header['channels']

#     # ####################################################################
#     # ## 核心修改点：不再假设通道名为'R' ##
#     # ####################################################################
#     # 1. 获取所有通道的名称
#     channel_names = sorted(channels_info.keys())
#     if not channel_names:
#         raise ValueError(f"EXR文件没有任何通道: {file_path}")

#     # 2. 以第一个通道为准，判断数据类型
#     first_channel_type = channels_info[channel_names[0]]
#     if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
#         dtype = np.float32
#     elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
#         dtype = np.float16
#     else:
#         raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

#     # 读取所有通道的字节数据
#     all_channels_bytes = exr_file.channels(channel_names)

#     # 将字节数据转换为NumPy数组并堆叠
#     np_channels = []
#     for i, name in enumerate(channel_names):
#         channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
#         channel_data = channel_data.reshape(height, width)
#         np_channels.append(channel_data)

#     image_np = np.stack(np_channels, axis=-1)
#     # 新增：读取后立即校验数据是否全零
#     if (image_np == 0).all():
#         raise ValueError(f"❌ EXR文件 {file_path} 读取后全为零！")

#     return image_np



# # 确保您导入了 read_exr, GenericRandomTransform, 和 CenterCrop
# # from util.exr import read_exr 
# # from datasets.transforms import GenericRandomTransform, CenterCrop

# class HyperspectralDepthDataset(Dataset):
#     # --- 1. 修改 __init__ 签名 ---
#     # 我们添加了 min_depth 和 max_depth 参数
#     def __init__(self, base_dir: str, scene_folders: List[str], image_size: Tuple[int, int], hs_channels: int,
#                  is_training: bool = True, randcrop: bool = False, augment: bool = False,
#                  min_depth: float = 0.1, max_depth: float = 2.0): # <--- 关键修改
        
#         super().__init__()
#         self.is_training = is_training

#         if isinstance(image_size, int):
#             image_size = (image_size, image_size)

#         self.transform = GenericRandomTransform(image_size, randcrop, augment, hs_channels)
#         self.centercrop = CenterCrop(image_size)

#         self.min_depth = min_depth
#         self.max_depth = max_depth

#         self.sample_pairs = []
#         for folder_name in scene_folders:
#             match = re.search(r'\d+', folder_name)
#             if not match: continue
#             scene_num = match.group(0).zfill(2)

#             hs_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_hs.exr')
#             depth_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_depth_map.exr')

#             if os.path.exists(hs_path) and os.path.exists(depth_path):
#                 self.sample_pairs.append({'hs_path': hs_path, 'depth_path': depth_path, 'id': f'scene_{scene_num}'})
#             else:
#                 print(f"警告: 在 '{folder_name}' 中找不到完整的数据对，已跳过。")
#                 print(f"  - 正在检查: {hs_path}")
#                 print(f"  - 正在检查: {depth_path}")

#     def __len__(self):
#         return len(self.sample_pairs)

#     def __getitem__(self, idx):
#         sample = self.sample_pairs[idx]
#         hs_path = sample['hs_path']
#         depth_path = sample['depth_path']
#         sample_id = sample['id']
        
#         try:
#             hs_image = read_exr(hs_path)
#             depth_map = read_exr(depth_path) # <--- 加载原始值 (例如 1103.8)
#         except Exception as e:
#             raise IOError(f"无法使用OpenEXR库加载文件。高光谱: {hs_path}, 深度: {depth_path}\n原始错误: {e}")

#         hs_image = hs_image.astype(np.float32)
#         depth_map = depth_map.astype(np.float32)

# #         if depth_map.ndim == 2:
# #             depth_map = depth_map[..., None] claude修改
#         if depth_map.ndim == 3:
#             depth_map = depth_map.squeeze(-1)  # (H, W, 1) → (H, W)
    
#         assert depth_map.ndim == 2, f"深度图应该是 2D，得到 {depth_map.shape}"
        
#          # 你的 .exr 文件存储的是 mm（例如 1103.8 mm = 1.1038 m）
#         depth_map = depth_map / 1000.0
#         max_hs = 0.0
#         min_hs = 2.0
        
#         min_depth = depth_map.min()
#         max_depth = depth_map.max()
        
#         SANITY_THRESHOLD = 10000.0 
        
#         if max_hs > SANITY_THRESHOLD:
#             # 策略：不使用统计学的 P99.9，而是寻找"除去异常值后的真实最大值"
#             # 也就是你说的"第二大的值"（或者正常范围内的最大值）
            
#             # 1. 找出所有"正常"的像素 (小于阈值的像素)
#             valid_mask = hs_image < SANITY_THRESHOLD
            
#             # 2. 如果存在正常像素，取它们中的最大值
#             # 对于 Scene 12，这会忽略掉 3e22，找到 4.09
#             if np.any(valid_mask):
#                 real_max = np.max(hs_image[valid_mask])
#             else:
#                 # 极少数情况：如果整张图都坏了(全大于10000)，就设为1.0
#                 real_max = 1.0
            
#             # 3. 执行截断
#             # 将那个唯一的 3e22 强行拉回到 4.09
#             hs_image = np.clip(hs_image, min_hs, real_max)
            
#             # 4. 关键：更新 max_hs
#             # 这样后续归一化时的分母就是 4.09，而不是 3e22
#             max_hs = real_max
#         # 高光谱归一化
#         if max_hs != min_hs:
#             hs_image = (hs_image - min_hs) / (max_hs - min_hs)
    
#         # 步骤 6：归一化深度图到 [0, 1]
#         # ============================================================
#         # 裁剪到有效范围
# #         depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
    
#         # 归一化到 [0, 1]
#         depth_map = (depth_map - min_depth) / (max_depth - min_depth)
#         # depth_map = metric_to_ips(depth_map, self.min_depth, self.max_depth)
    
        
#         hs_tensor = torch.from_numpy(hs_image).permute(2, 0, 1).float()
#         hs_tensor = hs_tensor.unsqueeze(0) 
#         depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)


#         if self.is_training:
#             hs_tensor, depth_tensor = self.transform(hs_tensor, depth_tensor)
#             if (hs_tensor == 0).all():
#                 print(f"⚠️ {sample_id} 的HS图像在transform后全零！")
#                 print(f"  transform前 range: [{hs_before_transform.min():.4f}, {hs_before_transform.max():.4f}]")
#         else:
#             hs_tensor = self.centercrop(hs_tensor)
#             depth_tensor = self.centercrop(depth_tensor)

#         hs_tensor = hs_tensor.squeeze(0) 
#         depth_tensor = depth_tensor.squeeze(0).squeeze(0)
        
        
#         # 现在 hs_tensor 是 [29, H, W]
#         # depth_tensor 是 [H, W]
#         # 这两个都是正确的！
        

#         result = {
#             'id': sample_id,
#             'hs_image': hs_tensor,
#             'depth_map': depth_tensor 
#         }

#         return result

# 2026.1.21修改版 增加mask版本

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List
from kornia.augmentation import CenterCrop
# 假设 transform 能够处理多通道输入 (C, H, W)，如果你的 GenericRandomTransform 
# 内部写死了只处理单通道深度图，可能需要微调，但通常 Kornia/TorchVision 都能处理多通道。
from .generic_transform import GenericRandomTransform
from util.helper import metric_to_ips

import OpenEXR
import Imath

def read_exr(file_path):
    """
    使用OpenEXR库读取.exr文件并返回一个NumPy数组。
    """
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())
    
    if not channel_names:
        raise ValueError(f"EXR文件没有任何通道: {file_path}")

    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)

    np_channels = []
    for i, name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    
    # 稍微放宽全0检查，避免浮点数精度极低的情况误判，但通常全0就是0
    if np.allclose(image_np, 0):
        raise ValueError(f"❌ EXR文件 {file_path} 读取后全为零！")

    return image_np


class HyperspectralDepthDataset(Dataset):
    def __init__(self, base_dir: str, scene_folders: List[str], image_size: Tuple[int, int], hs_channels: int,
                 is_training: bool = True, randcrop: bool = False, augment: bool = False,
                 min_depth: float = 0.4, max_depth: float = 2.0): # <--- 1. 修改默认值为物理范围
        
        super().__init__()
        self.is_training = is_training

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.transform = GenericRandomTransform(image_size, randcrop, augment, hs_channels)
        self.centercrop = CenterCrop(image_size)

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.sample_pairs = []
        for folder_name in scene_folders:
            match = re.search(r'\d+', folder_name)
            if not match: continue
            scene_num = match.group(0).zfill(2)

            hs_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_hs.exr')
            depth_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_depth_map.exr')

            if os.path.exists(hs_path) and os.path.exists(depth_path):
                self.sample_pairs.append({'hs_path': hs_path, 'depth_path': depth_path, 'id': f'scene_{scene_num}'})
            else:
                # 仅在调试时打印，避免刷屏
                pass

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        sample = self.sample_pairs[idx]
        hs_path = sample['hs_path']
        depth_path = sample['depth_path']
        sample_id = sample['id']
        
        try:
            hs_image = read_exr(hs_path)
            depth_map = read_exr(depth_path) 
        except Exception as e:
            raise IOError(f"无法读取文件: {sample_id} \n错误: {e}")

        hs_image = hs_image.astype(np.float32)
        depth_map = depth_map.astype(np.float32)

        # 确保深度图是 (H, W)
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        
        # --- 深度单位转换 (mm -> m) ---
        depth_map = depth_map / 1000.0

        # ============================================================
        # 步骤 A: 生成 Mask (在归一化造成负数之前)
        # ============================================================
        # 【IPS 迁移】背景是0，物体是0.4~2.0。
        # 任何低于 min_depth 的都是背景，标记为 0；有效物体标记为 1。
        # 这个 Mask 稍后会在 Loss 计算时用到，确保网络不学背景的虚假值。
        valid_mask = (depth_map > self.min_depth - 1e-3).astype(np.float32)

        # ============================================================
        # 步骤 B: 高光谱图像处理 (保持你原有的去异常值逻辑)
        # ============================================================
        SANITY_THRESHOLD = 10000.0 
        min_hs = 0.0 # 假设最小光强是0
        
        # 你的逻辑：如果最大值异常大，寻找正常的第二大值
        if np.max(hs_image) > SANITY_THRESHOLD:
            valid_pixels = hs_image < SANITY_THRESHOLD
            if np.any(valid_pixels):
                real_max = np.max(hs_image[valid_pixels])
            else:
                real_max = 1.0
            # 截断异常值
            hs_image = np.clip(hs_image, min_hs, real_max)
            max_hs = real_max
        else:
            max_hs = np.max(hs_image)

        # HS 归一化
        if max_hs > min_hs:
            hs_image = (hs_image - min_hs) / (max_hs - min_hs)
        
        # ============================================================
        # 步骤 C: 深度图归一化 (【IPS 体系】使用逆深度均匀化)
        # ============================================================
        # 【迁移到 IPS】线性深度分布在远场会导致 PSF 差异很小。
        # 改用逆深度 (Inverse Perspective Sampling) 使深度分辨率均匀。
        #
        # 数学：
        #   物理深度 d ∈ [0.4m, 2.0m]
        #   逆深度 d_inv = 1/d ∈ [0.5, 2.5]
        #   归一化: d_norm = (1/d - 1/d_max) / (1/d_min - 1/d_max)
        #        = (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)
        #
        # 边界条件：
        #   d=min_depth(0.4m): d_norm → 0.0（最近处）
        #   d=max_depth(2.0m): d_norm → 1.0（最远处）
        #   背景 d=0: 设为 1.0（或任意值，Mask 会排除）
        #
        # 使用 util.helper.metric_to_ips() 函数进行转换
        # ============================================================
        
        # 将 depth_map 转为 torch.Tensor 以使用 metric_to_ips
        depth_tensor = torch.from_numpy(depth_map).float()
        
        # 创建背景 mask（depth < min_depth 的像素）
        valid_mask_bool = depth_tensor >= self.min_depth - 1e-3
        
        # 对有效前景像素使用 metric_to_ips 函数进行 IPS 归一化
        # ⚠️  关键：背景像素设为 min_depth (而不是 max_depth)
        # 原因：metric_to_ips(min_depth) = 0.0，这样 matting 函数能正确识别背景
        #      matting 通过 (depthmap > 1e-6) 检测，IPS=0.0 会被正确排除
        #      如果设为 max_depth → IPS=1.0 → 会被错误分配到 Layer 7
        depth_safe = torch.where(valid_mask_bool, depth_tensor, torch.tensor(self.min_depth))
        
        # 使用已有的 metric_to_ips 函数
        ips_depth_tensor = metric_to_ips(depth_safe, self.min_depth, self.max_depth)
        
        # 确保在 [0, 1] 范围内，并转回 numpy
        depth_map = torch.clamp(ips_depth_tensor, 0.0, 1.0).numpy() 

        # ============================================================
        # 步骤 D: 转 Tensor 并处理 Transform
        # ============================================================
        # HS: [H, W, C] -> [C, H, W]
        hs_tensor = torch.from_numpy(hs_image).permute(2, 0, 1).float()
        
        # Depth & Mask: [H, W] -> [1, H, W]
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()

        # 增加 batch 维度以适配 transform: [1, C, H, W]
        hs_tensor = hs_tensor.unsqueeze(0)
        depth_tensor = depth_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        if self.is_training:
            # --- 关键技巧：拼接 Depth 和 Mask ---
            # 为了保证 crop/flip 时 Mask 和 Depth 保持一致，我们将它们拼接在 Channel 维度
            # concat 后形状: [1, 2, H, W] (假设Depth是单通道)
            depth_mask_cat = torch.cat([depth_tensor, mask_tensor], dim=1)

            # 一起送入 transform
            hs_tensor, depth_mask_cat = self.transform(hs_tensor, depth_mask_cat)

            # 拆分回 Depth 和 Mask
            # 假设 GenericRandomTransform 保持了通道数不变
            depth_tensor = depth_mask_cat[:, 0:1, :, :] # 取第0个通道
            mask_tensor = depth_mask_cat[:, 1:2, :, :]  # 取第1个通道
            
            # 由于插值(Bilinear等)可能会导致 Mask 边缘出现小数，建议二值化回来
            # 或者如果你用的是 Nearest 插值则不需要，但在 Dataset 里加这行比较保险
            mask_tensor = (mask_tensor > 0.5).float()

        else:
            hs_tensor = self.centercrop(hs_tensor)
            depth_tensor = self.centercrop(depth_tensor)
            mask_tensor = self.centercrop(mask_tensor)

        # 移除 batch 维度
        hs_tensor = hs_tensor.squeeze(0)          # [C, H, W]
        depth_tensor = depth_tensor.squeeze(0)    # [1, H, W]
        mask_tensor = mask_tensor.squeeze(0)      # [1, H, W]
        
        # 最终移除 depth/mask 的 channel 维度，变成 [H, W] (如果你的Loss函数需要[H,W]格式)
        # 如果你的模型输入需要 [1, H, W]，请注释掉下面两行
        depth_tensor = depth_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)

        result = {
            'id': sample_id,
            'hs_image': hs_tensor,
            'depth_map': depth_tensor,
            'mask': mask_tensor  # <--- 新增返回 Mask
        }

        return result