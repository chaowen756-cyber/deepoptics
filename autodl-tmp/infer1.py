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
#                               核心功能函数
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

def normalize_data(data):
    d_min = data.min()
    d_max = data.max()
    if d_max == d_min:
        return np.zeros_like(data), d_min, d_max
    norm_data = (data - d_min) / (d_max - d_min)
    return norm_data, d_min, d_max

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * np.log10(1.0 / mse)

# --- 修改点 1: 将 MSE 改为 MAE 计算函数 ---
def calculate_mae(img1, img2):
    """计算平均绝对误差 (Mean Absolute Error)"""
    return np.mean(np.abs(img1 - img2))

# def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
#     hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
#     bands = [min(b, hs_np.shape[2]-1) for b in bands]
#     rgb = hs_np[..., bands]
#     rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
#     plt.imsave(save_path, rgb)

def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
    """
    修改版可视化函数：使用 1%-99% 分位数进行鲁棒归一化，解决全黑问题
    """
    # 1. 转为 Numpy (H, W, C)
    hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # 2. 选取波段
    bands = [min(b, hs_np.shape[2]-1) for b in bands]
    rgb = hs_np[..., bands]
    
    # 3. 鲁棒归一化 (Robust Normalization)
    # 计算 1% 和 99% 的分位数，忽略极值
    p_min = np.percentile(rgb, 1)
    p_max = np.percentile(rgb, 99)
    
    # 防止分母为 0
    if p_max - p_min < 1e-6:
        p_max = p_min + 1.0
        
    # 截断并拉伸
    rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
    # 4. 保存
    plt.imsave(save_path, rgb)
    print(f"  [Visual Debug] Range clipped to [{p_min:.4f}, {p_max:.4f}] for {os.path.basename(save_path)}")
    
def visualize_depth(depth_tensor, save_path):
    depth_np = depth_tensor.cpu().numpy()
    plt.imsave(save_path, depth_np, cmap='inferno')

@torch.no_grad()
def process_single_scene(model, hs_path, depth_path, output_dir, patch_size, device):
    scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
    print(f"\nProcessing scene: {scene_name} ...")
    
    try:
        hs_gt = read_exr(hs_path) 
        depth_gt = read_exr(depth_path)
    except Exception as e:
        print(f"Error reading files for {scene_name}: {e}")
        return

    if depth_gt.ndim == 3: depth_gt = depth_gt.squeeze(-1)
    
    # 归一化
    depth_gt = depth_gt / 1000.0 # mm 转 m
    hs_norm, hs_min, hs_max = normalize_data(hs_gt)
    depth_norm, d_min, d_max = normalize_data(depth_gt)
    
    hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
    depth_tensor = torch.from_numpy(depth_norm).float()

    # -------------------------------------------------------------------------
    # 无缝拼接的尺寸计算逻辑 (保持不变)
    # -------------------------------------------------------------------------
    crop_width = model.hparams.crop_width
    total_padding_needed = 2 * crop_width  
    input_patch_size = patch_size
    output_valid_size = input_patch_size - 2 * total_padding_needed
    
    print(f"  [Info] Patch Size: {input_patch_size}, Crop Width: {crop_width}")
    print(f"  [Info] Valid Output Stride: {output_valid_size}")
    
    if output_valid_size <= 0:
        raise ValueError(f"Patch size {patch_size} 太小了，无法覆盖 4*{crop_width} 的边缘裁剪！")

    C, H, W = hs_tensor.shape
    est_hs_canvas = torch.zeros_like(hs_tensor)
    est_depth_canvas = torch.zeros_like(depth_tensor)
    
    # Padding
    pad_h = (output_valid_size - (H % output_valid_size)) % output_valid_size
    pad_w = (output_valid_size - (W % output_valid_size)) % output_valid_size
    
    hs_padded = torch.nn.functional.pad(hs_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_tensor.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect')
    
    hs_padded = torch.nn.functional.pad(hs_padded, (total_padding_needed, total_padding_needed, total_padding_needed, total_padding_needed), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_padded, (total_padding_needed, total_padding_needed, total_padding_needed, total_padding_needed), mode='reflect')
    
    _, _, H_pad, W_pad = hs_padded.shape

    # 滑动窗口
    for y in tqdm(range(0, H_pad - 2*total_padding_needed, output_valid_size), desc="Stitching"):
        for x in range(0, W_pad - 2*total_padding_needed, output_valid_size):
            
            if y + input_patch_size > H_pad or x + input_patch_size > W_pad: 
                continue

            hs_patch = hs_padded[:, :, y:y+input_patch_size, x:x+input_patch_size].to(device)
            depth_patch = depth_padded[:, :, y:y+input_patch_size, x:x+input_patch_size].squeeze(1).to(device)
            
            outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True))
            
            est_hs_patch = outputs.est_images.cpu()
            est_depth_patch = outputs.est_depthmaps.cpu()
            
            canvas_y, canvas_x = y, x
            curr_h, curr_w = est_hs_patch.shape[-2:]
            
            target_h = min(curr_h, H - canvas_y)
            target_w = min(curr_w, W - canvas_x)
            
            if target_h > 0 and target_w > 0:
                est_hs_canvas[:, canvas_y:canvas_y+target_h, canvas_x:canvas_x+target_w] = \
                    est_hs_patch[0, :, :target_h, :target_w]
                est_depth_canvas[canvas_y:canvas_y+target_h, canvas_x:canvas_x+target_w] = \
                    est_depth_patch[0, :target_h, :target_w]

    # 4. 统计与保存
    est_hs_canvas = torch.nan_to_num(est_hs_canvas, 0.0)
    est_depth_canvas = torch.nan_to_num(est_depth_canvas, 0.0)
    
    psnr_val = calculate_psnr(hs_norm, est_hs_canvas.numpy().transpose(1, 2, 0))
    
    # --- 修改点 2: 调用 MAE 计算 ---
    mae_val = calculate_mae(depth_norm, est_depth_canvas.numpy())
    
    print(f"  -> Result: PSNR={psnr_val:.4f} dB, MAE={mae_val:.6f}")
    
    scene_out_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    visualize_hyperspectral(est_hs_canvas, os.path.join(scene_out_dir, "est_hs.png"))
    visualize_depth(est_depth_canvas, os.path.join(scene_out_dir, "est_depth.png"))
    visualize_hyperspectral(hs_tensor, os.path.join(scene_out_dir, "gt_hs.png"))
    visualize_depth(depth_tensor, os.path.join(scene_out_dir, "gt_depth.png"))
    
    # --- 修改点 3: 写入文本时使用 MAE 标签 ---
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        f.write(f"{scene_name}: PSNR={psnr_val:.4f}, MAE={mae_val:.6f}\n")

# ==============================================================================
#                               主程序
# ==============================================================================

if __name__ == "__main__":
    
    # ################# 配置区域 #################
    INPUT_FOLDER = r"autodl-tmp/Baek数据集/deploy 1"
    CKPT_PATH = r"autodl-tmp/data/Hyperspectral_LearnedDepth/version_53/checkpoints"
    OUTPUT_DIR = "autodl-tmp/depth_test_1.19" # 修改输出目录名以示区别
    PATCH_SIZE = 512
    # ###########################################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if os.path.isdir(CKPT_PATH):
        ckpt_files = glob.glob(os.path.join(CKPT_PATH, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"在目录 {CKPT_PATH} 中未找到 .ckpt 文件")
        real_ckpt_path = ckpt_files[0]
        print(f"Auto-detected checkpoint: {real_ckpt_path}")
    else:
        real_ckpt_path = CKPT_PATH

    print(f"Loading model...")
    
    # 加载 hparams
    checkpoint = torch.load(real_ckpt_path, map_location='cpu')
    if 'hyper_parameters' in checkpoint:
        hparams_dict = checkpoint['hyper_parameters']
        if 'hparams' in hparams_dict and isinstance(hparams_dict['hparams'], (dict, argparse.Namespace)):
             hparams_obj = hparams_dict['hparams']
             if isinstance(hparams_obj, dict):
                 hparams = argparse.Namespace(**hparams_obj)
             else:
                 hparams = hparams_obj
        else:
            hparams = argparse.Namespace(**hparams_dict)
    else:
        print("Warning: No 'hyper_parameters' found in checkpoint.")
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