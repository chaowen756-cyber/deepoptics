# import os
# import glob
# import argparse
# import numpy as np
# import torch
# import OpenEXR
# import Imath
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # 引入项目模块
# from snapshotdepth_hs import SnapshotDepthHS

# # ==============================================================================
# #                               核心功能函数
# # ==============================================================================

# def read_exr(file_path):
#     if not OpenEXR.isOpenExrFile(file_path):
#         raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

#     exr_file = OpenEXR.InputFile(file_path)
#     header = exr_file.header()
#     dw = header['dataWindow']
#     width = dw.max.x - dw.min.x + 1
#     height = dw.max.y - dw.min.y + 1
#     channels_info = header['channels']
#     channel_names = sorted(channels_info.keys())

#     # 判断数据类型
#     first_channel_type = channels_info[channel_names[0]]
#     if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
#         dtype = np.float32
#     elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
#         dtype = np.float16
#     else:
#         raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

#     all_channels_bytes = exr_file.channels(channel_names)
#     np_channels = []
#     for i, name in enumerate(channel_names):
#         channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
#         channel_data = channel_data.reshape(height, width)
#         np_channels.append(channel_data)

#     image_np = np.stack(np_channels, axis=-1)
#     return image_np.astype(np.float32)

# def normalize_data(data):
#     d_min = data.min()
#     d_max = data.max()
#     if d_max == d_min:
#         return np.zeros_like(data), d_min, d_max
#     norm_data = (data - d_min) / (d_max - d_min)
#     return norm_data, d_min, d_max

# def calculate_psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0: return 100
#     return 10 * np.log10(1.0 / mse)

# def calculate_mae(img1, img2):
#     return np.mean(np.abs(img1 - img2))

# def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
#     hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
#     bands = [min(b, hs_np.shape[2]-1) for b in bands]
#     rgb = hs_np[..., bands]
#     p_min = np.percentile(rgb, 1)
#     p_max = np.percentile(rgb, 99)
#     if p_max - p_min < 1e-6: p_max = p_min + 1.0
#     rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
#     plt.imsave(save_path, rgb)

# def visualize_depth(depth_tensor, save_path):
#     depth_np = depth_tensor.cpu().numpy()
#     vmin = np.percentile(depth_np, 1)
#     vmax = np.percentile(depth_np, 99)
#     plt.imsave(save_path, depth_np, cmap='inferno', vmin=vmin, vmax=vmax)

# # ==============================================================================
# #                 生成梯形权重掩膜 (Trapezoidal Mask)
# # ==============================================================================
# def get_trapezoid_mask(h, w, overlap, device):
#     """
#     生成一个梯形权重的掩膜。
#     """
#     def get_1d_ramp(size, overlap_pixels):
#         ramp = torch.ones(size, device=device)
#         if overlap_pixels > 0:
#             ramp[:overlap_pixels] = torch.linspace(0, 1, overlap_pixels, device=device)
#             ramp[-overlap_pixels:] = torch.linspace(1, 0, overlap_pixels, device=device)
#         return ramp

#     mask_x = get_1d_ramp(w, overlap)
#     mask_y = get_1d_ramp(h, overlap)
#     mask = mask_y.unsqueeze(1) * mask_x.unsqueeze(0)
#     return mask + 1e-6

# @torch.no_grad()
# def process_single_scene(model, hs_path, depth_path, output_dir, patch_size, overlap_pixels, device):
#     scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
#     print(f"\nProcessing scene: {scene_name} ...")
    
#     try:
#         hs_gt = read_exr(hs_path) 
#         depth_gt = read_exr(depth_path)
#     except Exception as e:
#         print(f"Error reading files: {e}")
#         return

#     if depth_gt.ndim == 3: depth_gt = depth_gt.squeeze(-1)
#     depth_gt = depth_gt / 1000.0 # mm 转 m
#     hs_norm, _, _ = normalize_data(hs_gt)
#     depth_norm, _, _ = normalize_data(depth_gt)
    
#     hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
#     depth_tensor = torch.from_numpy(depth_norm).float()

#     # =========================================================================
#     #  【关键修正】参数计算
#     # =========================================================================
#     crop_width = model.hparams.crop_width
#     input_patch_size = patch_size
    
#     # 修正1：有效尺寸计算
#     # 模型 forward 中有两次 crop_boundary(x, crop_width)
#     # 所以单边总共减少了 2 * crop_width，双边就是 4 * crop_width
#     # 例如：512 - 4*32 = 384
#     valid_size = input_patch_size - 4 * crop_width
    
#     if overlap_pixels >= valid_size // 2:
#         print("Warning: Overlap 太大了，自动调整为 valid_size 的一半")
#         overlap_pixels = valid_size // 2

#     # 步长 = 有效尺寸 - 重叠量
#     stride = valid_size - overlap_pixels
    
#     print(f"  [Info] Valid Patch Size: {valid_size}x{valid_size} (Fixed: 512 - 4*{crop_width})")
#     print(f"  [Info] Overlap Pixels:    {overlap_pixels} px")
#     print(f"  [Info] Stride:            {stride} px")

#     C, H, W = hs_tensor.shape
    
#     est_hs_sum = torch.zeros((C, H, W), device=device)
#     est_hs_weight = torch.zeros((C, H, W), device=device)
#     est_depth_sum = torch.zeros((H, W), device=device)
#     est_depth_weight = torch.zeros((H, W), device=device)
    
#     # 生成 Mask (现在大小是 384x384)
#     patch_weight_mask = get_trapezoid_mask(valid_size, valid_size, overlap_pixels, device)

#     # 修正2：Padding 基础量 (2倍 crop_width)
#     # 为了让 Output 的中心对齐原图的左上角 (0,0)
#     # 512 的输入，中心产生 384 的输出，偏移量是 (512-384)/2 = 64 = 2*crop_width
#     pad_base = 2 * crop_width
    
#     hs_padded = torch.nn.functional.pad(hs_tensor.unsqueeze(0), (pad_base, pad_base, pad_base, pad_base), mode='reflect')
#     depth_padded = torch.nn.functional.pad(depth_tensor.unsqueeze(0).unsqueeze(0), (pad_base, pad_base, pad_base, pad_base), mode='reflect')
    
#     # 额外 Padding 保证滑窗能滑到最后
#     pad_buffer = patch_size
#     hs_padded = torch.nn.functional.pad(hs_padded, (0, pad_buffer, 0, pad_buffer), mode='reflect')
#     depth_padded = torch.nn.functional.pad(depth_padded, (0, pad_buffer, 0, pad_buffer), mode='reflect')
    
#     # =========================================================================
#     #  滑动窗口推理
#     # =========================================================================
#     for y in tqdm(range(0, H, stride), desc="Stitching"):
#         for x in range(0, W, stride):
            
#             # 提取
#             hs_patch = hs_padded[:, :, y:y+input_patch_size, x:x+input_patch_size].to(device)
#             depth_patch = depth_padded[:, :, y:y+input_patch_size, x:x+input_patch_size].squeeze(1).to(device)
            
#             # 推理
#             outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True))
            
#             out_hs = outputs.est_images        # 预期大小: 384x384
#             out_depth = outputs.est_depthmaps  # 预期大小: 384x384
            
#             # 这里的 y, x 对应 Output 在原图 Canvas 上的左上角坐标
#             curr_h, curr_w = valid_size, valid_size
#             target_h = min(curr_h, H - y)
#             target_w = min(curr_w, W - x)
            
#             if target_h > 0 and target_w > 0:
#                 mask_slice = patch_weight_mask[:target_h, :target_w]
                
#                 # 现在 tensor 大小应该匹配了 (都是 384 或被截断后的尺寸)
#                 est_hs_sum[:, y:y+target_h, x:x+target_w] += out_hs[0, :, :target_h, :target_w] * mask_slice
#                 est_hs_weight[:, y:y+target_h, x:x+target_w] += mask_slice
                
#                 est_depth_sum[y:y+target_h, x:x+target_w] += out_depth[0, :target_h, :target_w] * mask_slice
#                 est_depth_weight[y:y+target_h, x:x+target_w] += mask_slice

#     # 归一化
#     est_hs_weight[est_hs_weight == 0] = 1.0
#     est_depth_weight[est_depth_weight == 0] = 1.0
    
#     final_hs = est_hs_sum / est_hs_weight
#     final_depth = est_depth_sum / est_depth_weight
    
#     final_hs = torch.nan_to_num(final_hs, 0.0)
#     final_depth = torch.nan_to_num(final_depth, 0.0)
    
#     hs_cpu = final_hs.cpu()
#     depth_cpu = final_depth.cpu()
    
#     psnr_val = calculate_psnr(hs_norm, hs_cpu.numpy().transpose(1, 2, 0))
#     mae_val = calculate_mae(depth_norm, depth_cpu.numpy())
    
#     print(f"  -> Result: PSNR={psnr_val:.4f} dB, MAE={mae_val:.6f}")
    
#     scene_out_dir = os.path.join(output_dir, scene_name)
#     os.makedirs(scene_out_dir, exist_ok=True)
    
#     visualize_hyperspectral(final_hs, os.path.join(scene_out_dir, "est_hs.png"))
#     visualize_depth(final_depth, os.path.join(scene_out_dir, "est_depth.png"))
#     visualize_hyperspectral(hs_tensor, os.path.join(scene_out_dir, "gt_hs.png"))
#     visualize_depth(depth_tensor, os.path.join(scene_out_dir, "gt_depth.png"))
    
#     with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
#         f.write(f"{scene_name}: PSNR={psnr_val:.4f}, MAE={mae_val:.6f}\n")

# if __name__ == "__main__":
    
#     # ################# 配置区域 #################
#     INPUT_FOLDER = r"autodl-tmp/Baek数据集/deploy 1"
#     CKPT_PATH = r"autodl-tmp/data/Hyperspectral_LearnedDepth/version_51/checkpoints"
#     OUTPUT_DIR = "my_results_trapezoid_fixed" 
#     PATCH_SIZE = 512
#     OVERLAP_PIXELS = 64  # 重叠像素推荐值
#     # ###########################################
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     if os.path.isdir(CKPT_PATH):
#         ckpt_files = glob.glob(os.path.join(CKPT_PATH, "*.ckpt"))
#         if not ckpt_files: raise FileNotFoundError(f"No .ckpt found in {CKPT_PATH}")
#         real_ckpt_path = ckpt_files[0]
#         print(f"Auto-detected checkpoint: {real_ckpt_path}")
#     else:
#         real_ckpt_path = CKPT_PATH

#     print(f"Loading model...")
#     checkpoint = torch.load(real_ckpt_path, map_location='cpu')
#     if 'hyper_parameters' in checkpoint:
#         hparams_dict = checkpoint['hyper_parameters']
#         if 'hparams' in hparams_dict and isinstance(hparams_dict['hparams'], (dict, argparse.Namespace)):
#              hparams_obj = hparams_dict['hparams']
#              if isinstance(hparams_obj, dict): hparams = argparse.Namespace(**hparams_obj)
#              else: hparams = hparams_obj
#         else:
#             hparams = argparse.Namespace(**hparams_dict)
#     else:
#         hparams = argparse.Namespace()

#     model = SnapshotDepthHS.load_from_checkpoint(real_ckpt_path, hparams=hparams)
#     model.eval()
#     model.to(device)
    
#     search_pattern = os.path.join(INPUT_FOLDER, "*_hs.exr")
#     hs_files = glob.glob(search_pattern)
    
#     if not hs_files:
#         print(f"Error: No *_hs.exr files found in {INPUT_FOLDER}")
#     else:
#         print(f"Found {len(hs_files)} scenes to process.")
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         for hs_file in hs_files:
#             depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
#             if not os.path.exists(depth_file):
#                 print(f"Warning: Depth map not found for {hs_file}, skipping.")
#                 continue
#             process_single_scene(model, hs_file, depth_file, OUTPUT_DIR, PATCH_SIZE, OVERLAP_PIXELS, device)
# import os
# import glob
# import argparse
# import numpy as np
# import torch
# import OpenEXR
# import Imath
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # 引入项目模块
# from snapshotdepth_hs import SnapshotDepthHS

# # ==============================================================================
# #                               核心工具函数
# # ==============================================================================

# def read_exr(file_path):
#     if not OpenEXR.isOpenExrFile(file_path):
#         raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

#     exr_file = OpenEXR.InputFile(file_path)
#     header = exr_file.header()
#     dw = header['dataWindow']
#     width = dw.max.x - dw.min.x + 1
#     height = dw.max.y - dw.min.y + 1
#     channels_info = header['channels']
#     channel_names = sorted(channels_info.keys())

#     # 判断数据类型
#     first_channel_type = channels_info[channel_names[0]]
#     if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
#         dtype = np.float32
#     elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
#         dtype = np.float16
#     else:
#         raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

#     all_channels_bytes = exr_file.channels(channel_names)
#     np_channels = []
#     for i, name in enumerate(channel_names):
#         channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
#         channel_data = channel_data.reshape(height, width)
#         np_channels.append(channel_data)

#     image_np = np.stack(np_channels, axis=-1)
#     return image_np.astype(np.float32)

# def normalize_data(data):
#     """
#     Min-Max 归一化，并返回 min/max 用于后续反归一化
#     """
#     d_min = data.min()
#     d_max = data.max()
#     if d_max == d_min:
#         return np.zeros_like(data), d_min, d_max
#     norm_data = (data - d_min) / (d_max - d_min)
#     return norm_data, d_min, d_max

# def denormalize_data(norm_data, d_min, d_max):
#     """
#     【新功能】反归一化：将 [0,1] 的预测值还原回真实物理单位（米）
#     """
#     return norm_data * (d_max - d_min) + d_min

# def calculate_psnr(img1, img2, data_range=1.0):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0: return 100
#     return 10 * np.log10((data_range ** 2) / mse)

# def calculate_mse(img1, img2):
#     """计算均方误差 (MSE)"""
#     return np.mean((img1 - img2) ** 2)

# def calculate_mae(img1, img2):
#     """计算平均绝对误差 (MAE)"""
#     return np.mean(np.abs(img1 - img2))

# # ==============================================================================
# #                       可视化函数 (支持统一色阶)
# # ==============================================================================

# def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
#     hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
#     bands = [min(b, hs_np.shape[2]-1) for b in bands]
#     rgb = hs_np[..., bands]
    
#     # 鲁棒归一化用于显示
#     p_min = np.percentile(rgb, 1)
#     p_max = np.percentile(rgb, 99)
#     if p_max - p_min < 1e-6: p_max = p_min + 1.0
#     rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
#     plt.imsave(save_path, rgb)

# def visualize_depth(depth_data, save_path, vmin=None, vmax=None):
#     """
#     深度图可视化
#     Args:
#         depth_data: Numpy数组 (H, W)
#         vmin, vmax: 强制设定颜色范围，确保 GT 和 Prediction 颜色一致
#     """
#     # 如果没有指定范围，使用自身的鲁棒范围
#     if vmin is None: vmin = np.percentile(depth_data, 1)
#     if vmax is None: vmax = np.percentile(depth_data, 99)
    
#     plt.imsave(save_path, depth_data, cmap='inferno', vmin=vmin, vmax=vmax)

# # ==============================================================================
# #                    Cosine 权重掩膜 (消除块效应的核心)
# # ==============================================================================
# def get_cosine_mask(h, w, device):
#     """
#     生成 Hanning (Cosine) 窗口掩膜。
#     用于 50% 重叠拼接，中心权重 1，边缘权重 0，过渡极其平滑。
#     """
#     # 1D Hanning Window
#     # 公式: sin^2(pi * x / N)
#     idx_h = torch.linspace(0, np.pi, h, device=device)
#     idx_w = torch.linspace(0, np.pi, w, device=device)
    
#     mask_h = torch.sin(idx_h) ** 2
#     mask_w = torch.sin(idx_w) ** 2
    
#     # 2D Mask
#     mask = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)
    
#     return mask + 1e-8

# @torch.no_grad()
# def process_single_scene(model, hs_path, depth_path, output_dir, patch_size, device):
#     scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
#     print(f"\nProcessing scene: {scene_name} ...")
    
#     # 1. 读取原始数据 (Raw Data in Meters)
#     try:
#         hs_gt_raw = read_exr(hs_path) 
#         depth_gt_raw = read_exr(depth_path)
#     except Exception as e:
#         print(f"Error reading files: {e}")
#         return

#     if depth_gt_raw.ndim == 3: depth_gt_raw = depth_gt_raw.squeeze(-1)
    
#     # 单位转换: mm -> m
#     depth_gt_raw = depth_gt_raw / 1000.0 
    
#     # 2. 归一化 (Input to Model)
#     # 记录 min/max 以便最后反归一化
#     hs_norm, hs_min, hs_max = normalize_data(hs_gt_raw)
#     depth_norm, d_min, d_max = normalize_data(depth_gt_raw)
    
#     print(f"  [Data Info] Real Depth Range: [{d_min:.4f}m, {d_max:.4f}m]")

#     # 转 Tensor
#     hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
#     depth_tensor = torch.from_numpy(depth_norm).float()

#     # =========================================================================
#     #  参数计算 (Baek et al. 风格的 50% 重叠)
#     # =========================================================================
#     crop_width = model.hparams.crop_width
#     input_patch_size = patch_size
    
#     # 有效尺寸: 512 - 4*32 = 384
#     valid_size = input_patch_size - 4 * crop_width
    
#     # 【关键策略】50% 重叠 (Stride = Valid / 2)
#     # 这是消除块效应的最强手段
#     stride = valid_size // 2 
    
#     print(f"  [Stitching] Valid Patch: {valid_size}x{valid_size}")
#     print(f"  [Stitching] Stride:      {stride} px (50% Overlap for Seamless)")

#     C, H, W = hs_tensor.shape
    
#     # 累加器
#     est_hs_sum = torch.zeros((C, H, W), device=device)
#     est_hs_weight = torch.zeros((C, H, W), device=device)
#     est_depth_sum = torch.zeros((H, W), device=device)
#     est_depth_weight = torch.zeros((H, W), device=device)
    
#     # 生成 Cosine Mask (384x384)
#     patch_weight_mask = get_cosine_mask(valid_size, valid_size, device)

#     # Padding
#     pad_base = 2 * crop_width
#     pad_buffer = patch_size # 额外缓冲
    
#     hs_padded = torch.nn.functional.pad(hs_tensor.unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
#     depth_padded = torch.nn.functional.pad(depth_tensor.unsqueeze(0).unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
    
#     # =========================================================================
#     #  滑动窗口推理
#     # =========================================================================
#     # 这里的循环逻辑稍微调整，以适应 padding 偏移
#     # 起始点 y=0 对应原图 (0,0) 的有效输出
#     # 实际读取 patch 时需要偏移 pad_buffer
    
#     for y in tqdm(range(0, H, stride), desc="Inference"):
#         for x in range(0, W, stride):
            
#             # 计算在 Padded 图上的切片坐标
#             # pad_base 已经被模型内部吃掉了，所以我们需要从 pad_buffer 开始取
#             # 实际上：y + pad_buffer 就是我们需要的左上角
            
#             py = y + pad_buffer
#             px = x + pad_buffer
            
#             hs_patch = hs_padded[:, :, py:py+input_patch_size, px:px+input_patch_size].to(device)
#             depth_patch = depth_padded[:, :, py:py+input_patch_size, px:px+input_patch_size].squeeze(1).to(device)
            
#             # 推理
#             outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True))
            
#             out_hs = outputs.est_images
#             out_depth = outputs.est_depthmaps
            
#             # 累加到 Canvas
#             # 这里的 y, x 是原图坐标
#             curr_h, curr_w = valid_size, valid_size
#             target_h = min(curr_h, H - y)
#             target_w = min(curr_w, W - x)
            
#             if target_h > 0 and target_w > 0:
#                 mask_slice = patch_weight_mask[:target_h, :target_w]
                
#                 est_hs_sum[:, y:y+target_h, x:x+target_w] += out_hs[0, :, :target_h, :target_w] * mask_slice
#                 est_hs_weight[:, y:y+target_h, x:x+target_w] += mask_slice
                
#                 est_depth_sum[y:y+target_h, x:x+target_w] += out_depth[0, :target_h, :target_w] * mask_slice
#                 est_depth_weight[y:y+target_h, x:x+target_w] += mask_slice

#     # 3. 归一化 (Canvas Normalization)
#     final_hs_norm = est_hs_sum / (est_hs_weight + 1e-8)
#     final_depth_norm = est_depth_sum / (est_depth_weight + 1e-8)
    
#     # 移除 NaN
#     final_hs_norm = torch.nan_to_num(final_hs_norm, 0.0)
#     final_depth_norm = torch.nan_to_num(final_depth_norm, 0.0)
    
#     # 转 CPU numpy
#     final_hs_norm = final_hs_norm.cpu().numpy().transpose(1, 2, 0) # H,W,C
#     final_depth_norm = final_depth_norm.cpu().numpy()              # H,W
    
#     # =========================================================================
#     #  【关键步骤】反归一化与指标计算 (真实物理单位)
#     # =========================================================================
    
#     # 还原到真实深度 (Meters)
#     est_depth_real = denormalize_data(final_depth_norm, d_min, d_max)
    
#     # 计算指标 (使用真实深度)
#     mse_real = calculate_mse(depth_gt_raw, est_depth_real)
#     psnr_hs = calculate_psnr(hs_norm, final_hs_norm) # HS 通常看归一化后的视觉质量
    
#     print(f"  -> Metrics (Real World):")
#     print(f"     Depth MSE:  {mse_real:.6f} (m^2)")
#     print(f"     HS PSNR:    {psnr_hs:.4f} dB")
    
#     # =========================================================================
#     #  可视化 (统一色阶)
#     # =========================================================================
#     scene_out_dir = os.path.join(output_dir, scene_name)
#     os.makedirs(scene_out_dir, exist_ok=True)
    
#     # 设定统一的深度显示范围 (以 GT 为准)
#     # 这样 GT 和 Prediction 的颜色才是一一对应的
#     vmin_depth = np.percentile(depth_gt_raw, 1)
#     vmax_depth = np.percentile(depth_gt_raw, 99)
    
#     visualize_hyperspectral(torch.from_numpy(final_hs_norm).permute(2,0,1), 
#                            os.path.join(scene_out_dir, "est_hs.png"))
    
#     # 保存真实深度图 (Est)
#     visualize_depth(est_depth_real, 
#                     os.path.join(scene_out_dir, "est_depth_real.png"), 
#                     vmin=vmin_depth, vmax=vmax_depth)
    
#     # 保存 GT 对比
#     visualize_hyperspectral(hs_tensor, os.path.join(scene_out_dir, "gt_hs.png"))
#     visualize_depth(depth_gt_raw, 
#                     os.path.join(scene_out_dir, "gt_depth_real.png"), 
#                     vmin=vmin_depth, vmax=vmax_depth)
    
#     # 保存结果到 txt
#     with open(os.path.join(output_dir, 'metrics_real.txt'), 'a') as f:
#         f.write(f"{scene_name}: Depth_MSE={mse_real:.6f} (m^2), HS_PSNR={psnr_hs:.4f}\n")

# if __name__ == "__main__":
    
#     # ################# 配置区域 #################
#     INPUT_FOLDER = r"autodl-tmp/Baek数据集/deploy 1"
#     CKPT_PATH = r"autodl-tmp/data/Hyperspectral_LearnedDepth/version_54/checkpoints"
#     OUTPUT_DIR = "ckp-0.166_2025.1.19" 
#     PATCH_SIZE = 512
#     # ###########################################
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     if os.path.isdir(CKPT_PATH):
#         ckpt_files = glob.glob(os.path.join(CKPT_PATH, "*.ckpt"))
#         if not ckpt_files: raise FileNotFoundError(f"No .ckpt found in {CKPT_PATH}")
#         real_ckpt_path = ckpt_files[0]
#         print(f"Auto-detected checkpoint: {real_ckpt_path}")
#     else:
#         real_ckpt_path = CKPT_PATH

#     print(f"Loading model...")
#     checkpoint = torch.load(real_ckpt_path, map_location='cpu')
#     if 'hyper_parameters' in checkpoint:
#         hparams_dict = checkpoint['hyper_parameters']
#         if 'hparams' in hparams_dict and isinstance(hparams_dict['hparams'], (dict, argparse.Namespace)):
#              hparams_obj = hparams_dict['hparams']
#              if isinstance(hparams_obj, dict): hparams = argparse.Namespace(**hparams_obj)
#              else: hparams = hparams_obj
#         else:
#             hparams = argparse.Namespace(**hparams_dict)
#     else:
#         hparams = argparse.Namespace()

#     model = SnapshotDepthHS.load_from_checkpoint(real_ckpt_path, hparams=hparams)
#     model.eval()
#     model.to(device)
    
#     search_pattern = os.path.join(INPUT_FOLDER, "*_hs.exr")
#     hs_files = glob.glob(search_pattern)
    
#     if not hs_files:
#         print(f"Error: No *_hs.exr files found in {INPUT_FOLDER}")
#     else:
#         print(f"Found {len(hs_files)} scenes to process.")
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         for hs_file in hs_files:
#             depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
#             if not os.path.exists(depth_file):
#                 print(f"Warning: Depth map not found for {hs_file}, skipping.")
#                 continue
#             process_single_scene(model, hs_file, depth_file, OUTPUT_DIR, PATCH_SIZE, device)
import os
import glob
import argparse
import numpy as np
import torch
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from tqdm import tqdm

# 引入项目模块
from snapshotdepth_hs import SnapshotDepthHS

# ==============================================================================
#                               核心工具函数
# ==============================================================================

def read_exr(file_path):
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())

    # 判断数据类型
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
    return image_np.astype(np.float32)

def normalize_data_hs(data):
    """
    高光谱数据的归一化 (依然使用 Min-Max)
    """
    d_min = data.min()
    d_max = data.max()
    if d_max == d_min:
        return np.zeros_like(data), d_min, d_max
    norm_data = (data - d_min) / (d_max - d_min)
    return norm_data, d_min, d_max

def compute_depth_histogram(depth_map,
                            valid_mask,
                            depth_min=0.4,
                            depth_max=2.0,
                            num_bins=8):
    """
    统计深度图在各深度区间内的像素占比（仅统计 valid 区域）
    
    Returns:
        bin_edges: (num_bins + 1,)
        percentages: (num_bins,)  每个区间的百分比
        counts: (num_bins,)       像素数量（可选）
    """

    # 只取有效像素
    valid_depths = depth_map[valid_mask > 0.5]

    if valid_depths.size == 0:
        raise ValueError("No valid pixels found in depth map.")

    # 固定物理区间分箱
    bin_edges = np.linspace(depth_min, depth_max, num_bins + 1)

    counts, _ = np.histogram(
        valid_depths,
        bins=bin_edges
    )

    total = counts.sum()
    percentages = counts / total * 100.0

    return bin_edges, percentages, counts

    
def normalize_depth_fixed(data, min_val=0.0, max_val=2.0):
    """
    【关键修改】固定范围归一化
    使用训练时设定的 0.0m - 2.0m 进行归一化。
    """
    # 1. 截断异常值 (Clip): 保证输入不会超过训练范围
    data_clipped = np.clip(data, min_val, max_val)
    
    # 2. 归一化到 [0, 1]
    norm_data = (data_clipped - min_val) / (max_val - min_val)
    
    return norm_data

def denormalize_depth_fixed(norm_data, min_val=0.0, max_val=2.0):
    """
    【关键修改】固定范围反归一化
    将 [0, 1] 还原回 [0.0, 2.0] 米
    """
    return norm_data * (max_val - min_val) + min_val

def calculate_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * np.log10((data_range ** 2) / mse)

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def calculate_mae_masked(gt, pred, mask):
    """
    gt, pred, mask: shape (H, W)
    mask > 0 的位置才参与 MAE 计算
    """
    diff = np.abs(gt - pred)
    valid_diff = diff[mask > 0]

    if valid_diff.size == 0:
        return 0.0  # 或 np.nan，看你习惯
    return valid_diff.mean()

# ==============================================================================
#                       可视化函数 (统一色阶)
# ==============================================================================

def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
    hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
    bands = [min(b, hs_np.shape[2]-1) for b in bands]
    rgb = hs_np[..., bands]
    
    # 鲁棒归一化用于显示
    p_min = np.percentile(rgb, 1)
    p_max = np.percentile(rgb, 99)
    if p_max - p_min < 1e-6: p_max = p_min + 1.0
    rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
    plt.imsave(save_path, rgb)

def visualize_depth(depth_data, save_path, vmin=0.0, vmax=2.0):
    """
    深度图可视化
    默认使用 [0, 2.0] 米的固定范围，这样颜色绝对准确。
    """
    plt.imsave(save_path, depth_data, cmap='inferno', vmin=vmin, vmax=vmax)

# ==============================================================================
#                    Cosine 权重掩膜 (消除块效应)
# ==============================================================================
def get_cosine_mask(h, w, device):
    """
    生成 Hanning (Cosine) 窗口掩膜。
    """
    idx_h = torch.linspace(0, np.pi, h, device=device)
    idx_w = torch.linspace(0, np.pi, w, device=device)
    
    mask_h = torch.sin(idx_h) ** 2
    mask_w = torch.sin(idx_w) ** 2
    
    mask = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)
    return mask + 1e-8

@torch.no_grad()
def process_single_scene(model, hs_path, depth_path, output_dir, patch_size, device):
    scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
    print(f"\nProcessing scene: {scene_name} ...")
    
    # 1. 读取原始数据 (Raw Data)
    try:
        hs_gt_raw = read_exr(hs_path) 
        depth_gt_raw = read_exr(depth_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    if depth_gt_raw.ndim == 3: depth_gt_raw = depth_gt_raw.squeeze(-1)
    
    # 单位转换: mm -> m
    depth_gt_raw = depth_gt_raw / 1000.0 
    valid_mask = (depth_gt_raw > 1e-6).astype(np.float32)
    
    # 2. 归一化 (Input to Model)
    # HS 使用 Min-Max
    hs_norm, hs_min, hs_max = normalize_data_hs(hs_gt_raw)
    
    # 【关键】Depth 使用固定范围 [0, 2.0]
    FIXED_MIN = 0.4
    FIXED_MAX = 2.0
    depth_norm = normalize_depth_fixed(depth_gt_raw, FIXED_MIN, FIXED_MAX)
    
    print(f"  [Data Info] GT Depth Range (Raw): [{depth_gt_raw.min():.4f}m, {depth_gt_raw.max():.4f}m]")
    print(f"  [Data Info] Fixed Norm Range:     [{FIXED_MIN}m, {FIXED_MAX}m]")

    # 转 Tensor
    hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
    depth_tensor = torch.from_numpy(depth_norm).float()

    # =========================================================================
    #  参数计算 (50% Overlap)
    # =========================================================================
    crop_width = model.hparams.crop_width
    input_patch_size = patch_size
    valid_size = input_patch_size - 4 * crop_width
    
    # 50% 重叠 (Seamless)
    stride = valid_size // 2 
    
    print(f"  [Stitching] Valid Patch: {valid_size}x{valid_size}")
    print(f"  [Stitching] Stride:      {stride} px")

    C, H, W = hs_tensor.shape
    
    # 累加器
    est_hs_sum = torch.zeros((C, H, W), device=device)
    est_hs_weight = torch.zeros((C, H, W), device=device)
    est_depth_sum = torch.zeros((H, W), device=device)
    est_depth_weight = torch.zeros((H, W), device=device)
    
    # 生成 Cosine Mask
    patch_weight_mask = get_cosine_mask(valid_size, valid_size, device)

    # Padding
    pad_base = 2 * crop_width
    pad_buffer = patch_size 
    
    hs_padded = torch.nn.functional.pad(hs_tensor.unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_tensor.unsqueeze(0).unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
    
    # =========================================================================
    #  推理
    # =========================================================================
    for y in tqdm(range(0, H, stride), desc="Inference"):
        for x in range(0, W, stride):
            py = y + pad_buffer
            px = x + pad_buffer
            
            hs_patch = hs_padded[:, :, py:py+input_patch_size, px:px+input_patch_size].to(device)
            depth_patch = depth_padded[:, :, py:py+input_patch_size, px:px+input_patch_size].squeeze(1).to(device)
            
            outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True))
            
            out_hs = outputs.est_images
            out_depth = outputs.est_depthmaps
            
            curr_h, curr_w = valid_size, valid_size
            target_h = min(curr_h, H - y)
            target_w = min(curr_w, W - x)
            
            if target_h > 0 and target_w > 0:
                mask_slice = patch_weight_mask[:target_h, :target_w]
                est_hs_sum[:, y:y+target_h, x:x+target_w] += out_hs[0, :, :target_h, :target_w] * mask_slice
                est_hs_weight[:, y:y+target_h, x:x+target_w] += mask_slice
                est_depth_sum[y:y+target_h, x:x+target_w] += out_depth[0, :target_h, :target_w] * mask_slice
                est_depth_weight[y:y+target_h, x:x+target_w] += mask_slice

    # 3. 归一化结果 (Canvas Normalization)
    final_hs_norm = est_hs_sum / (est_hs_weight + 1e-8)
    final_depth_norm = est_depth_sum / (est_depth_weight + 1e-8)
    
    final_hs_norm = torch.nan_to_num(final_hs_norm, 0.0)
    final_depth_norm = torch.nan_to_num(final_depth_norm, 0.0)
    
    final_hs_norm = final_hs_norm.cpu().numpy().transpose(1, 2, 0)
    final_depth_norm = final_depth_norm.cpu().numpy()

    # =========================================================================
    #  【关键步骤】固定范围反归一化
    # =========================================================================
    
    # 还原到真实深度 (Meters)
    # 使用 0.0 和 2.0，而不是数据自身的 min/max
    est_depth_real = denormalize_depth_fixed(final_depth_norm, FIXED_MIN, FIXED_MAX)
    
    # =========================================================
    # 深度分布统计（GT vs Prediction）
    # =========================================================

    NUM_BINS = 8
    DEPTH_MIN = FIXED_MIN   # 0.4
    DEPTH_MAX = FIXED_MAX   # 2.0

    gt_bins, gt_pct, gt_cnt = compute_depth_histogram(
        depth_gt_raw,
        valid_mask,
        DEPTH_MIN,
        DEPTH_MAX,
        NUM_BINS
    )

    pred_bins, pred_pct, pred_cnt = compute_depth_histogram(
        est_depth_real,
        valid_mask,
        DEPTH_MIN,
        DEPTH_MAX,
        NUM_BINS
    )
    
    print("\n  [Depth Distribution Analysis]")
    print("  Range (m)      GT (%)     Pred (%)")
    print("  ----------------------------------")

    for i in range(NUM_BINS):
        d0 = gt_bins[i]
        d1 = gt_bins[i + 1]
        print(f"  {d0:.2f}–{d1:.2f}      {gt_pct[i]:6.2f}     {pred_pct[i]:6.2f}")


    
    print(f"  [Result Info] Pred Depth Range: [{est_depth_real.min():.4f}m, {est_depth_real.max():.4f}m]")

    # 计算指标
    mse_real = calculate_mse(depth_gt_raw, est_depth_real)
    mae_real = calculate_mae_masked(depth_gt_raw, est_depth_real, valid_mask)
    psnr_hs = calculate_psnr(hs_norm, final_hs_norm) 
    
    print(f"  -> Metrics (Real World):")
    print(f"     Depth MSE:  {mse_real:.6f} (m^2)")
    print(f"     Depth MAE:  {mae_real:.6f} (m)")
    print(f"     HS PSNR:    {psnr_hs:.4f} dB")
    
    # =========================================================================
    #  可视化
    # =========================================================================
    scene_out_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    # 1. 预测深度图 (Locked Scale 0-2m)
    # 这样如果有颜色，说明在 0-2m 之间；如果全黑/全黄，说明超出范围
    visualize_depth(est_depth_real, 
                    os.path.join(scene_out_dir, "est_depth_fixed_scale.png"), 
                    vmin=FIXED_MIN, vmax=FIXED_MAX)
    
    # 2. GT 深度图 (Locked Scale 0-2m)
    # 用于对比，颜色应当一致
    visualize_depth(depth_gt_raw, 
                    os.path.join(scene_out_dir, "gt_depth_fixed_scale.png"), 
                    vmin=FIXED_MIN, vmax=FIXED_MAX)
    
    # 3. 自动色阶版 (用于检查结构，防止因数值问题全黑/全黄看不见)
    visualize_depth(est_depth_real, 
                    os.path.join(scene_out_dir, "est_depth_auto_scale.png"), 
                    vmin=None, vmax=None)

    # HS 图像
    visualize_hyperspectral(torch.from_numpy(final_hs_norm).permute(2,0,1), 
                           os.path.join(scene_out_dir, "est_hs.png"))
    visualize_hyperspectral(hs_tensor, os.path.join(scene_out_dir, "gt_hs.png"))

    # 保存日志
    with open(os.path.join(output_dir, 'metrics_real.txt'), 'a') as f:
        f.write(f"{scene_name}: MSE={mse_real:.6f}, MAE={mae_real:.6f}, HS_PSNR={psnr_hs:.4f}\n")

if __name__ == "__main__":
    
    # ################# 配置区域 #################
    INPUT_FOLDER = r"autodl-tmp/Baek数据集/deploy 16"
    CKPT_PATH = r"autodl-tmp/data/Hyperspectral_LearnedDepth/version_57/checkpoints"
    OUTPUT_DIR = "test1.24" 
    PATCH_SIZE = 512
    # ###########################################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if os.path.isdir(CKPT_PATH):
        ckpt_files = glob.glob(os.path.join(CKPT_PATH, "*.ckpt"))
        if not ckpt_files: raise FileNotFoundError(f"No .ckpt found in {CKPT_PATH}")
        real_ckpt_path = ckpt_files[0]
        print(f"Auto-detected checkpoint: {real_ckpt_path}")
    else:
        real_ckpt_path = CKPT_PATH

    print(f"Loading model...")
    checkpoint = torch.load(real_ckpt_path, map_location='cpu')
    if 'hyper_parameters' in checkpoint:
        hparams_dict = checkpoint['hyper_parameters']
        if 'hparams' in hparams_dict and isinstance(hparams_dict['hparams'], (dict, argparse.Namespace)):
             hparams_obj = hparams_dict['hparams']
             if isinstance(hparams_obj, dict): hparams = argparse.Namespace(**hparams_obj)
             else: hparams = hparams_obj
        else:
            hparams = argparse.Namespace(**hparams_dict)
    else:
        hparams = argparse.Namespace()

    model = SnapshotDepthHS.load_from_checkpoint(real_ckpt_path, hparams=hparams)
    model.eval()
    model.to(device)
    
    search_pattern = os.path.join(INPUT_FOLDER, "*_hs.exr")
    hs_files = glob.glob(search_pattern)
    
    if not hs_files:
        print(f"Error: No *_hs.exr files found in {INPUT_FOLDER}")
    else:
        print(f"Found {len(hs_files)} scenes to process.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for hs_file in hs_files:
            depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
            if not os.path.exists(depth_file):
                print(f"Warning: Depth map not found for {hs_file}, skipping.")
                continue
            process_single_scene(model, hs_file, depth_file, OUTPUT_DIR, PATCH_SIZE, device)