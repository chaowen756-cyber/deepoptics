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

def normalize_data_hs_robust(data):
    """HS数据：使用分位数归一化，剔除极亮噪点"""
    d_min = np.percentile(data, 0.1)
    d_max = np.percentile(data, 99.9)
    if d_max <= d_min:
        d_min = data.min()
        d_max = data.max()
    if d_max == d_min:
        return np.zeros_like(data), d_min, d_max
    
    data_clipped = np.clip(data, d_min, d_max)
    norm_data = (data_clipped - d_min) / (d_max - d_min)
    return norm_data, d_min, d_max

def normalize_depth_fixed(data, min_val=0.0, max_val=2.0):
    """Depth数据：固定范围归一化 [0, 2.0]"""
    data_clipped = np.clip(data, min_val, max_val)
    norm_data = (data_clipped - min_val) / (max_val - min_val)
    return norm_data

def denormalize_depth_fixed(norm_data, min_val=0.0, max_val=2.0):
    """Depth数据：反归一化"""
    return norm_data * (max_val - min_val) + min_val

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def calculate_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * np.log10((data_range ** 2) / mse)

# ==============================================================================
#                       诊断可视化函数
# ==============================================================================

def plot_distribution_comparison(pred_norm, gt_norm, save_path):
    """
    绘制预测值和真实值的分布直方图对比
    这是检查'全黄'问题最直观的方法
    """
    plt.figure(figsize=(10, 6))
    
    # 展平数据
    p_flat = pred_norm.flatten()
    g_flat = gt_norm.flatten()
    
    plt.hist(g_flat, bins=50, range=(0, 1), alpha=0.5, label='Ground Truth (Norm)', color='blue', density=True)
    plt.hist(p_flat, bins=50, range=(0, 1), alpha=0.5, label='Prediction (Norm)', color='red', density=True)
    
    plt.title("Depth Distribution Comparison (Normalized Space)")
    plt.xlabel("Normalized Depth Value (0.0=0m, 1.0=2m)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()

def visualize_depth(depth_data, save_path, vmin=0.0, vmax=2.0):
    plt.imsave(save_path, depth_data, cmap='inferno', vmin=vmin, vmax=vmax)

def visualize_hyperspectral(hs_tensor, save_path, bands=[4, 14, 24]):
    hs_np = hs_tensor.cpu().numpy().transpose(1, 2, 0)
    bands = [min(b, hs_np.shape[2]-1) for b in bands]
    rgb = hs_np[..., bands]
    p_min = np.percentile(rgb, 1)
    p_max = np.percentile(rgb, 99)
    if p_max - p_min < 1e-6: p_max = p_min + 1.0
    rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    plt.imsave(save_path, rgb)

# ==============================================================================
#                    Cosine 权重掩膜
# ==============================================================================
def get_cosine_mask(h, w, device):
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
    
    # 1. 读取原始数据
    try:
        hs_gt_raw = read_exr(hs_path) 
        depth_gt_raw = read_exr(depth_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    if depth_gt_raw.ndim == 3: depth_gt_raw = depth_gt_raw.squeeze(-1)
    depth_gt_raw = depth_gt_raw / 1000.0 # mm -> m
    
    # 2. 归一化 (Input to Model)
    hs_norm, hs_min, hs_max = normalize_data_hs_robust(hs_gt_raw)
    
    # 这里的 depth_norm 就是 Ground Truth (Normalized)
    # 我们用它来和模型直接输出的 Tensor 进行对比
    FIXED_MIN = 0.0
    FIXED_MAX = 2.0
    depth_gt_norm = normalize_depth_fixed(depth_gt_raw, FIXED_MIN, FIXED_MAX)
    
    hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
    depth_tensor = torch.from_numpy(depth_gt_norm).float()

    # =========================================================================
    #  推理 (50% Overlap + Cosine)
    # =========================================================================
    crop_width = model.hparams.crop_width
    input_patch_size = patch_size
    valid_size = input_patch_size - 4 * crop_width
    stride = valid_size // 2 
    
    C, H, W = hs_tensor.shape
    est_hs_sum = torch.zeros((C, H, W), device=device)
    est_hs_weight = torch.zeros((C, H, W), device=device)
    est_depth_sum = torch.zeros((H, W), device=device)
    est_depth_weight = torch.zeros((H, W), device=device)
    patch_weight_mask = get_cosine_mask(valid_size, valid_size, device)
    
    pad_base = 2 * crop_width
    pad_buffer = patch_size 
    
    hs_padded = torch.nn.functional.pad(hs_tensor.unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
    depth_padded = torch.nn.functional.pad(depth_tensor.unsqueeze(0).unsqueeze(0), (pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer, pad_base+pad_buffer), mode='reflect')
    
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

    # 3. 得到模型的归一化输出
    final_hs_norm = est_hs_sum / (est_hs_weight + 1e-8)
    final_depth_norm = est_depth_sum / (est_depth_weight + 1e-8)
    
    final_hs_norm = torch.nan_to_num(final_hs_norm, 0.0).cpu().numpy().transpose(1, 2, 0)
    final_depth_norm = torch.nan_to_num(final_depth_norm, 0.0).cpu().numpy()

    # =========================================================================
    #  【核心诊断】反归一化之前的全面检查
    # =========================================================================
    print(f"\n{'-'*20} DIAGNOSTIC REPORT {'-'*20}")
    
    # 1. 统计数据对比 (Normalized Space)
    print("1. [Normalized Space Stats] (Ideal range: 0.0 ~ 1.0)")
    print(f"   GT Depth:   Min={depth_gt_norm.min():.4f}, Max={depth_gt_norm.max():.4f}, Mean={depth_gt_norm.mean():.4f}")
    print(f"   Pred Depth: Min={final_depth_norm.min():.4f}, Max={final_depth_norm.max():.4f}, Mean={final_depth_norm.mean():.4f}")
    
    # 2. 计算归一化域的 MAE
    mae_norm = calculate_mae(depth_gt_norm, final_depth_norm)
    print(f"\n2. [Normalized Metrics]")
    print(f"   MAE (Norm): {mae_norm:.4f}")
    
    if mae_norm > 0.2:
        print("   >>> 警告: 归一化数据的 MAE 很大！说明模型输出的数值分布本身就是错的。")
    elif final_depth_norm.mean() > 0.9:
        print("   >>> 警告: 预测值均值接近 1.0 (全黄)，说明模型输出了饱和值 (Saturation)。")
        print("       可能原因: HS输入太暗导致模型致盲，或者模型未收敛。")
    else:
        print("   >>> 归一化数据差异尚可，请检查物理单位换算。")

    # 3. 反归一化
    est_depth_real = denormalize_depth_fixed(final_depth_norm, FIXED_MIN, FIXED_MAX)
    
    # 4. 真实空间指标
    mse_real = calculate_mse(depth_gt_raw, est_depth_real)
    mae_real = calculate_mae(depth_gt_raw, est_depth_real)
    psnr_hs = calculate_psnr(hs_norm, final_hs_norm) 
    
    print(f"\n3. [Real World Metrics]")
    print(f"   Depth MSE:  {mse_real:.6f} (m^2)")
    print(f"   Depth MAE:  {mae_real:.6f} (m)")
    print(f"   HS PSNR:    {psnr_hs:.4f} dB")
    print(f"{'-'*55}\n")
    
    # =========================================================================
    #  保存与可视化
    # =========================================================================
    scene_out_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    # 画分布直方图 (最直观的证据)
    plot_distribution_comparison(final_depth_norm, depth_gt_norm, 
                               os.path.join(scene_out_dir, "debug_distribution.png"))
    
    # 保存深度图
    visualize_depth(est_depth_real, os.path.join(scene_out_dir, "est_depth_fixed.png"), vmin=FIXED_MIN, vmax=FIXED_MAX)
    visualize_depth(est_depth_real, os.path.join(scene_out_dir, "est_depth_auto.png"), vmin=None, vmax=None)
    visualize_depth(depth_gt_raw, os.path.join(scene_out_dir, "gt_depth_fixed.png"), vmin=FIXED_MIN, vmax=FIXED_MAX)
    
    # HS图像
    visualize_hyperspectral(torch.from_numpy(final_hs_norm).permute(2,0,1), os.path.join(scene_out_dir, "est_hs.png"))
    visualize_hyperspectral(hs_tensor, os.path.join(scene_out_dir, "gt_hs.png"))

    with open(os.path.join(output_dir, 'metrics_diagnostic.txt'), 'a') as f:
        f.write(f"{scene_name}: MAE_norm={mae_norm:.4f}, Depth_MAE={mae_real:.6f}, HS_PSNR={psnr_hs:.4f}\n")

if __name__ == "__main__":
    
    INPUT_FOLDER = r"autodl-tmp/Baek数据集/deploy 1"
    CKPT_PATH = r"autodl-tmp/data/Hyperspectral_LearnedDepth/version_54/checkpoints"
    OUTPUT_DIR = "final_diagnostic_result" 
    PATCH_SIZE = 512
    
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