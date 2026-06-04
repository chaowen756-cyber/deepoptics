#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py - 高光谱+深度联合重建推理脚本

改进点：
1. 深度归一化与训练完全一致（使用 IPS 空间）
2. 正确的反归一化（使用 ips_to_metric）
3. 支持命令行参数，灵活配置
4. 支持处理单张图像或整个文件夹
5. 完善的指标计算和可视化
6. 代码结构清晰，易于维护

Usage:
    # 处理单个场景文件夹
    python inference.py --input_dir ./Baek数据集/deploy\ 16 --ckpt_path ./checkpoints/model.ckpt
    
    # 处理多个场景（父文件夹包含多个 deploy X）
    python inference.py --input_dir ./Baek数据集 --ckpt_path ./checkpoints/model.ckpt --multi_scene
    
    # 指定输出目录和 patch 尺寸
    python inference.py --input_dir ./data --ckpt_path ./model.ckpt --output_dir ./results --patch_size 512
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Dict, Optional

# EXR 读取
import OpenEXR
import Imath

# 项目模块
from snapshotdepth_hs import SnapshotDepthHS
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf
from util.helper import metric_to_ips, ips_to_metric, crop_boundary


# ==============================================================================
#                               数据读取与预处理
# ==============================================================================

def read_exr(file_path: str) -> np.ndarray:
    """读取 EXR 文件并返回 float32 数组"""
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"不是有效的 EXR 文件: {file_path}")

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
        raise TypeError(f"不支持的数据类型: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)
    np_channels = []
    for i, name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    return image_np.astype(np.float32)


def normalize_hs(hs_image: np.ndarray, sanity_threshold: float = 10000.0) -> Tuple[np.ndarray, float, float]:
    """
    高光谱图像归一化（与训练一致）
    处理异常大值后进行 Min-Max 归一化
    """
    min_hs = 0.0
    max_hs = np.max(hs_image)
    
    if max_hs > sanity_threshold:
        valid_pixels = hs_image < sanity_threshold
        if np.any(valid_pixels):
            max_hs = np.max(hs_image[valid_pixels])
        else:
            max_hs = 1.0
        hs_image = np.clip(hs_image, min_hs, max_hs)
    
    if max_hs > min_hs:
        hs_norm = (hs_image - min_hs) / (max_hs - min_hs)
    else:
        hs_norm = np.zeros_like(hs_image)
    
    return hs_norm, min_hs, max_hs


def normalize_depth_ips(depth_map: np.ndarray, min_depth: float, max_depth: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    深度图归一化（使用 IPS 空间，与训练完全一致）
    
    Args:
        depth_map: 物理深度图 (H, W)，单位：米
        min_depth: 最小有效深度
        max_depth: 最大有效深度
        
    Returns:
        depth_norm: IPS 归一化后的深度图 [0, 1]
        valid_mask: 有效像素 mask (1=前景, 0=背景)
    """
    # 生成 valid mask（背景像素深度 < min_depth）
    valid_mask = (depth_map >= min_depth - 1e-3).astype(np.float32)
    
    # 转为 tensor 进行 IPS 变换
    depth_tensor = torch.from_numpy(depth_map).float()
    valid_mask_bool = depth_tensor >= min_depth - 1e-3
    
    # 背景像素设为 min_depth（IPS=0），与训练一致
    depth_safe = torch.where(valid_mask_bool, depth_tensor, torch.tensor(min_depth))
    
    # 使用 metric_to_ips 进行归一化
    ips_depth = metric_to_ips(depth_safe, min_depth, max_depth)
    
    # Clamp 并转回 numpy
    depth_norm = torch.clamp(ips_depth, 0.0, 1.0).numpy()
    
    return depth_norm, valid_mask


def denormalize_depth_ips(depth_norm: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """
    深度图反归一化（从 IPS 空间转回物理深度）
    
    Args:
        depth_norm: IPS 归一化的深度图 [0, 1]
        min_depth: 最小深度
        max_depth: 最大深度
        
    Returns:
        depth_real: 物理深度图（米）
    """
    depth_tensor = torch.from_numpy(depth_norm).float()
    depth_real = ips_to_metric(depth_tensor, min_depth, max_depth)
    return depth_real.numpy()


# ==============================================================================
#                               指标计算
# ==============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def calculate_mae_masked(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """计算 Masked MAE（只在有效区域计算）"""
    valid_diff = np.abs(gt - pred)[mask > 0.5]
    if valid_diff.size == 0:
        return 0.0
    return float(valid_diff.mean())


def calculate_rmse_masked(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """计算 Masked RMSE"""
    valid_diff_sq = ((gt - pred) ** 2)[mask > 0.5]
    if valid_diff_sq.size == 0:
        return 0.0
    return float(np.sqrt(valid_diff_sq.mean()))


def compute_depth_histogram(depth_map: np.ndarray, valid_mask: np.ndarray,
                           depth_min: float, depth_max: float, num_bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """计算深度直方图分布"""
    valid_depths = depth_map[valid_mask > 0.5]
    if valid_depths.size == 0:
        return np.linspace(depth_min, depth_max, num_bins + 1), np.zeros(num_bins)
    
    bin_edges = np.linspace(depth_min, depth_max, num_bins + 1)
    counts, _ = np.histogram(valid_depths, bins=bin_edges)
    percentages = counts / counts.sum() * 100.0 if counts.sum() > 0 else counts
    
    return bin_edges, percentages


# ==============================================================================
#                               可视化
# ==============================================================================

def visualize_hyperspectral(hs_data: np.ndarray, save_path: str, bands: list = [4, 14, 24]):
    """高光谱伪彩色可视化"""
    if hs_data.ndim == 3 and hs_data.shape[0] < hs_data.shape[2]:
        # (C, H, W) -> (H, W, C)
        hs_data = hs_data.transpose(1, 2, 0)
    
    bands = [min(b, hs_data.shape[2] - 1) for b in bands]
    rgb = hs_data[..., bands]
    
    p_min = np.percentile(rgb, 1)
    p_max = np.percentile(rgb, 99)
    if p_max - p_min < 1e-6:
        p_max = p_min + 1.0
    rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    
    plt.imsave(save_path, rgb)


def visualize_depth(depth_data: np.ndarray, save_path: str, 
                   vmin: Optional[float] = None, vmax: Optional[float] = None):
    """深度图可视化"""
    if vmin is None:
        vmin = np.percentile(depth_data, 1)
    if vmax is None:
        vmax = np.percentile(depth_data, 99)
    
    plt.imsave(save_path, depth_data, cmap='inferno', vmin=vmin, vmax=vmax)


def visualize_depth_comparison(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                               save_path: str, vmin: float, vmax: float):
    """GT 与预测深度图对比可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # GT
    im0 = axes[0].imshow(gt_depth, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Prediction
    im1 = axes[1].imshow(pred_depth, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Error Map
    error = np.abs(gt_depth - pred_depth)
    im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')
    
    # Colorbar
    fig.colorbar(im0, ax=axes[:2], shrink=0.6, label='Depth (m)')
    fig.colorbar(im2, ax=axes[2], shrink=0.6, label='Error (m)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
#                               滑动窗口推理
# ==============================================================================

def get_cosine_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """生成 Hanning (Cosine) 窗口掩膜，用于平滑拼接"""
    idx_h = torch.linspace(0, np.pi, h, device=device)
    idx_w = torch.linspace(0, np.pi, w, device=device)
    
    mask_h = torch.sin(idx_h) ** 2
    mask_w = torch.sin(idx_w) ** 2
    
    mask = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)
    return mask + 1e-8


@torch.no_grad()
def generate_captimgs_full(model: SnapshotDepthHS,
                           hs_tensor_bchw: torch.Tensor,
                           depth_ips_bhw: torch.Tensor,
                           device: torch.device,
                           *,
                           disable_noise: bool) -> torch.Tensor:
    """整图一次性生成 captimgs（模拟传感器观测）。"""
    hs_tensor_bchw = hs_tensor_bchw.to(device)
    depth_ips_bhw = depth_ips_bhw.to(device)

    # 使用相机 forward 生成观测（与训练 forward 内部一致）
    captimgs, _, _ = model.camera.forward(
        hs_tensor_bchw,
        depth_ips_bhw,
        occlusion=getattr(model.hparams, 'occlusion', True),
        is_training=False,
    )

    # 推理评估通常关闭噪声；若需要模拟拍摄噪声，则复用训练中的噪声注入逻辑
    if not disable_noise:
        sigma_min = float(getattr(model.hparams, 'noise_sigma_min', 0.0))
        sigma_max = float(getattr(model.hparams, 'noise_sigma_max', 0.0))
        if sigma_max > 0 or sigma_min > 0:
            noise_sigma = (sigma_max - sigma_min) * torch.rand(
                (captimgs.shape[0], 1, 1, 1), device=device
            ) + sigma_min
            captimgs = captimgs + noise_sigma * torch.randn_like(captimgs)

    return captimgs


@torch.no_grad()
def generate_captimgs_tiled(model: SnapshotDepthHS,
                            hs_tensor_bchw: torch.Tensor,
                            depth_ips_bhw: torch.Tensor,
                            device: torch.device,
                            *,
                            disable_noise: bool,
                            tile_out_size: int,
                            offload_cpu: bool) -> torch.Tensor:
    """分块生成 captimgs，避免整图 PSF FFT 导致 OOM。

    思路：
    - 训练/推理最终都会裁掉边界 crop_width/2*crop_width 来规避 circular convolution 边缘伪影。
    - 因此对每个 tile：输入额外带 2*crop_width 的边界，输出再丢掉 crop_width 边界，拼接回整图。

    返回：captimgs_full (B, C, H, W)。默认可 offload 到 CPU，后续重建时按 patch 搬运到 GPU。
    """
    crop_width = int(model.hparams.crop_width)
    if tile_out_size <= 0:
        raise ValueError("tile_out_size 必须为正整数")
    if tile_out_size <= 2 * crop_width:
        raise ValueError(f"tile_out_size={tile_out_size} 太小，必须 > 2*crop_width={2*crop_width}")

    if hs_tensor_bchw.ndim != 4:
        raise ValueError(f"hs_tensor_bchw 维度应为 (B,C,H,W)，但得到 {hs_tensor_bchw.shape}")
    if depth_ips_bhw.ndim != 3:
        raise ValueError(f"depth_ips_bhw 维度应为 (B,H,W)，但得到 {depth_ips_bhw.shape}")

    B, C, H, W = hs_tensor_bchw.shape
    if B != 1:
        raise ValueError("当前推理脚本只支持 B=1（单场景整图）")

    H_in = H - 2 * crop_width
    W_in = W - 2 * crop_width
    if H_in <= 0 or W_in <= 0:
        raise ValueError(f"图像太小：crop_width={crop_width} 导致有效区域为 {H_in}x{W_in}")

    out_device = torch.device('cpu') if offload_cpu else device
    captimgs_full = torch.zeros((B, C, H, W), dtype=torch.float32, device=out_device)

    occlusion = getattr(model.hparams, 'occlusion', True)

    # 只需要保证 captimgs_full 的中心区域 [crop_width:-crop_width] 正确即可
    for y in tqdm(range(0, H_in, tile_out_size), desc="Capture(tiled)", leave=False):
        for x in range(0, W_in, tile_out_size):
            tile_out_h = min(tile_out_size, H_in - y)
            tile_out_w = min(tile_out_size, W_in - x)

            # 输入 tile 带上 2*crop_width 的边界，用于消除 FFT circular wrap-around
            y0 = y
            x0 = x
            y1 = y + tile_out_h + 2 * crop_width
            x1 = x + tile_out_w + 2 * crop_width

            hs_patch = hs_tensor_bchw[:, :, y0:y1, x0:x1].to(device, non_blocking=True)
            depth_patch = depth_ips_bhw[:, y0:y1, x0:x1].to(device, non_blocking=True)

            capt_patch, _, _ = model.camera.forward(
                hs_patch,
                depth_patch,
                occlusion=occlusion,
                is_training=False,
            )

            if not disable_noise:
                sigma_min = float(getattr(model.hparams, 'noise_sigma_min', 0.0))
                sigma_max = float(getattr(model.hparams, 'noise_sigma_max', 0.0))
                if sigma_max > 0 or sigma_min > 0:
                    noise_sigma = (sigma_max - sigma_min) * torch.rand(
                        (capt_patch.shape[0], 1, 1, 1), device=device
                    ) + sigma_min
                    capt_patch = capt_patch + noise_sigma * torch.randn_like(capt_patch)

            # 丢掉 tile 的边界 crop_width，只保留中心有效区域
            if crop_width > 0:
                capt_center = capt_patch[:, :, crop_width:-crop_width, crop_width:-crop_width]
            else:
                capt_center = capt_patch

            # 写回整图中心区域
            oy0 = crop_width + y
            ox0 = crop_width + x
            oy1 = oy0 + tile_out_h
            ox1 = ox0 + tile_out_w
            capt_center = capt_center[:, :, :tile_out_h, :tile_out_w]
            captimgs_full[:, :, oy0:oy1, ox0:ox1] = capt_center.to(out_device)

            # 释放临时变量，降低峰值显存
            del hs_patch, depth_patch, capt_patch, capt_center
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return captimgs_full


@torch.no_grad()
def reconstruct_from_captimgs_sliding_window(model: SnapshotDepthHS,
                                             captimgs_bchw: torch.Tensor,
                                             patch_size: int,
                                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    只用解码器做滑窗重建：
    - 不再每个 patch 重复跑相机光学 forward（加速）
    - 如开启 preinverse，则每个 patch 计算一次 tikhonov 伪逆体素

    Returns:
        est_hs_out: (C, H_out, W_out)
        est_depth_out: (H_out, W_out) in IPS
    """
    crop_width = int(model.hparams.crop_width)
    if patch_size <= 4 * crop_width:
        raise ValueError(f"patch_size={patch_size} 太小，必须 > 4*crop_width={4*crop_width}")

    # 1) 对齐训练 forward：先裁掉 captimgs 的边界 crop_width
    captimgs_in = crop_boundary(captimgs_bchw, crop_width)  # (B, C, H-2cw, W-2cw)

    B, C, H_in, W_in = captimgs_in.shape
    H_out = H_in - 2 * crop_width
    W_out = W_in - 2 * crop_width
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"输入太小：crop_width={crop_width} 导致输出尺寸为 {H_out}x{W_out}")

    # 2) 滑窗参数：输入 patch 尺寸与输出有效尺寸
    patch_in = patch_size - 2 * crop_width
    patch_out = patch_in - 2 * crop_width  # == patch_size - 4*crop_width
    stride = patch_out // 2
    if stride <= 0:
        raise ValueError("stride<=0，请增大 patch_size")

    # 3) 累加器（输出空间）
    est_hs_sum = torch.zeros((C, H_out, W_out), device=device)
    est_hs_weight = torch.zeros((C, H_out, W_out), device=device)
    est_depth_sum = torch.zeros((H_out, W_out), device=device)
    est_depth_weight = torch.zeros((H_out, W_out), device=device)

    patch_weight_mask = get_cosine_mask(patch_out, patch_out, device)

    # 4) 为了能在边界提 patch，对 captimgs_in 做 padding
    pad = patch_in
    # 注意：captimgs_in 可能在 CPU（为了省显存）。这里允许在 CPU 上做 padding，后续每个 patch 再搬到 GPU。
    captimgs_padded = F.pad(captimgs_in, (pad, pad, pad, pad), mode='reflect')

    # 5) 预先计算 psf_pure（之后按 patch_in 裁剪即可）
    psf_pure = model.camera.psf_at_camera(is_training=torch.tensor(False, device=device)).unsqueeze(0)

    # 6) 滑窗只跑 decoder（+可选 preinverse）
    for y in tqdm(range(0, H_out, stride), desc="Reconstruct", leave=False):
        for x in range(0, W_out, stride):
            py = y + pad
            px = x + pad
            capt_patch = captimgs_padded[:, :, py:py + patch_in, px:px + patch_in].to(device, non_blocking=True)

            if model.hparams.preinverse:
                psf_cropped = crop_psf(psf_pure, capt_patch.shape[-2:])
                pinv_volumes = apply_tikhonov_inverse(
                    capt_patch, psf_cropped, float(model.hparams.reg_tikhonov), apply_edgetaper=True
                )
            else:
                pinv_volumes = torch.zeros(
                    (capt_patch.shape[0], capt_patch.shape[1], model.hparams.n_depths, capt_patch.shape[-2], capt_patch.shape[-1]),
                    device=device,
                    dtype=capt_patch.dtype,
                )

            outputs = model.decoder(capt_patch, pinv_volumes)
            est_images_raw = outputs.est_images  # (B, C, patch_in, patch_in)
            est_depth_raw = outputs.est_depthmaps  # (B, patch_in, patch_in)

            # 对齐训练 forward：再裁掉 crop_width 得到最终输出 patch
            est_images = est_images_raw[:, :, crop_width:-crop_width, crop_width:-crop_width]
            est_depth = est_depth_raw[:, crop_width:-crop_width, crop_width:-crop_width]

            target_h = min(patch_out, H_out - y)
            target_w = min(patch_out, W_out - x)
            if target_h <= 0 or target_w <= 0:
                continue

            mask_slice = patch_weight_mask[:target_h, :target_w]
            est_hs_sum[:, y:y + target_h, x:x + target_w] += est_images[0, :, :target_h, :target_w] * mask_slice
            est_hs_weight[:, y:y + target_h, x:x + target_w] += mask_slice
            est_depth_sum[y:y + target_h, x:x + target_w] += est_depth[0, :target_h, :target_w] * mask_slice
            est_depth_weight[y:y + target_h, x:x + target_w] += mask_slice

    est_hs_out = est_hs_sum / (est_hs_weight + 1e-8)
    est_depth_out = est_depth_sum / (est_depth_weight + 1e-8)
    est_hs_out = torch.nan_to_num(est_hs_out, 0.0)
    est_depth_out = torch.nan_to_num(est_depth_out, 0.0)
    return est_hs_out.cpu(), est_depth_out.cpu()


# ==============================================================================
#                               场景处理
# ==============================================================================

@torch.no_grad()
def process_scene(model, hs_path: str, depth_path: str, output_dir: str,
                  patch_size: int, device: torch.device,
                  min_depth: float, max_depth: float,
                  *,
                  disable_noise: bool,
                  captimgs_mode: str,
                  captimgs_tile_out: int,
                  captimgs_offload_cpu: bool,
                  captimgs_oom_fallback: bool) -> Dict[str, float]:
    """
    处理单个场景
    
    Returns:
        metrics: 指标字典
    """
    scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"{'='*60}")
    
    # 1. 读取数据
    try:
        hs_gt_raw = read_exr(hs_path)
        depth_gt_raw = read_exr(depth_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return {}
    
    if depth_gt_raw.ndim == 3:
        depth_gt_raw = depth_gt_raw.squeeze(-1)
    
    # 单位转换: mm -> m
    depth_gt_raw = depth_gt_raw / 1000.0
    
    print(f"  Image size: {hs_gt_raw.shape[:2]}")
    print(f"  GT Depth range: [{depth_gt_raw.min():.4f}m, {depth_gt_raw.max():.4f}m]")
    
    # 2. 归一化（与训练一致）
    hs_norm, hs_min, hs_max = normalize_hs(hs_gt_raw)
    depth_norm, valid_mask = normalize_depth_ips(depth_gt_raw, min_depth, max_depth)
    
    # 3. 转 Tensor（注意：整图生成 captimgs 需要 batch 维度）
    hs_tensor_bchw = torch.from_numpy(hs_norm).permute(2, 0, 1).unsqueeze(0).float()
    depth_tensor_bhw = torch.from_numpy(depth_norm).unsqueeze(0).float()

    model.to(device)
    model.eval()

    # 4. 生成观测 captimgs
    # - full：整图一次性相机 forward（可能因 PSF FFT 过大 OOM）
    # - tiled：分块相机 forward，显存友好（推荐大分辨率）
    try:
        if captimgs_mode == 'full':
            captimgs_full = generate_captimgs_full(
                model,
                hs_tensor_bchw,
                depth_tensor_bhw,
                device,
                disable_noise=disable_noise,
            )
        elif captimgs_mode == 'tiled':
            captimgs_full = generate_captimgs_tiled(
                model,
                hs_tensor_bchw,
                depth_tensor_bhw,
                device,
                disable_noise=disable_noise,
                tile_out_size=captimgs_tile_out,
                offload_cpu=captimgs_offload_cpu,
            )
        else:
            raise ValueError(f"未知 captimgs_mode: {captimgs_mode}（应为 full 或 tiled）")
    except torch.cuda.OutOfMemoryError as e:
        if not captimgs_oom_fallback:
            raise
        print("[OOM] 整图生成 captimgs 发生 CUDA OOM，自动切换到 tiled 模式继续。")
        print(f"      OOM: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        captimgs_full = generate_captimgs_tiled(
            model,
            hs_tensor_bchw,
            depth_tensor_bhw,
            device,
            disable_noise=disable_noise,
            tile_out_size=captimgs_tile_out,
            offload_cpu=captimgs_offload_cpu,
        )

    # 5. 再对 decoder 做滑窗重建（输出已经对齐到 crop 2*crop_width 的区域）
    est_hs, est_depth_ips = reconstruct_from_captimgs_sliding_window(
        model,
        captimgs_full,
        patch_size,
        device,
    )

    # 6. 转回 numpy
    est_hs_np = est_hs.numpy().transpose(1, 2, 0)  # (H_out, W_out, C)
    est_depth_ips_np = est_depth_ips.numpy()  # (H_out, W_out)
    
    # 7. 深度反归一化（IPS -> 物理深度）
    est_depth_real = denormalize_depth_ips(est_depth_ips_np, min_depth, max_depth)
    
    print(f"  Pred Depth range: [{est_depth_real.min():.4f}m, {est_depth_real.max():.4f}m]")
    
    # 8. 指标必须在对齐后的区域计算（与训练 forward 最终输出一致：crop 2*crop_width）
    crop2 = int(model.hparams.crop_width) * 2
    gt_depth_crop = depth_gt_raw[crop2:-crop2, crop2:-crop2]
    gt_hs_crop = hs_norm[crop2:-crop2, crop2:-crop2, :]
    valid_mask_crop = valid_mask[crop2:-crop2, crop2:-crop2]

    mae_depth = calculate_mae_masked(gt_depth_crop, est_depth_real, valid_mask_crop)
    rmse_depth = calculate_rmse_masked(gt_depth_crop, est_depth_real, valid_mask_crop)
    psnr_hs = calculate_psnr(gt_hs_crop, est_hs_np)
    
    print(f"\n  Metrics:")
    print(f"    Depth MAE:  {mae_depth:.4f} m")
    print(f"    Depth RMSE: {rmse_depth:.4f} m")
    print(f"    HS PSNR:    {psnr_hs:.2f} dB")
    
    # 8. 深度分布分析
    print(f"\n  Depth Distribution (valid pixels):")
    gt_bins, gt_pct = compute_depth_histogram(gt_depth_crop, valid_mask_crop, min_depth, max_depth)
    pred_bins, pred_pct = compute_depth_histogram(est_depth_real, valid_mask_crop, min_depth, max_depth)
    
    print(f"    {'Range (m)':<12} {'GT (%)':<10} {'Pred (%)':<10}")
    print(f"    {'-'*32}")
    for i in range(len(gt_pct)):
        print(f"    {gt_bins[i]:.2f}-{gt_bins[i+1]:.2f}       {gt_pct[i]:6.2f}     {pred_pct[i]:6.2f}")
    
    # 9. 保存结果
    scene_out_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    # 深度图（固定色阶）
    visualize_depth(est_depth_real,
                   os.path.join(scene_out_dir, "pred_depth.png"),
                   vmin=min_depth, vmax=max_depth)
    visualize_depth(gt_depth_crop,
                   os.path.join(scene_out_dir, "gt_depth.png"),
                   vmin=min_depth, vmax=max_depth)
    
    # 深度对比图
    visualize_depth_comparison(gt_depth_crop, est_depth_real,
                              os.path.join(scene_out_dir, "depth_comparison.png"),
                              vmin=min_depth, vmax=max_depth)
    
    # 高光谱图
    visualize_hyperspectral(est_hs_np, os.path.join(scene_out_dir, "pred_hs.png"))
    visualize_hyperspectral(gt_hs_crop, os.path.join(scene_out_dir, "gt_hs.png"))
    
    # 保存 IPS 空间的预测结果（用于调试）
    np.save(os.path.join(scene_out_dir, "pred_depth_ips.npy"), est_depth_ips_np)
    np.save(os.path.join(scene_out_dir, "pred_depth_real.npy"), est_depth_real)
    
    metrics = {
        'scene': scene_name,
        'mae_depth': mae_depth,
        'rmse_depth': rmse_depth,
        'psnr_hs': psnr_hs,
    }
    
    return metrics


# ==============================================================================
#                               主函数
# ==============================================================================

def load_model(ckpt_path: str, device: torch.device, *, disable_noise: bool) -> SnapshotDepthHS:
    """加载模型（推理评估用）"""
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 提取 hparams
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
        raise ValueError("Checkpoint 中没有找到 hyper_parameters")
    
    # 推理评估：默认关闭 forward 内部的随机噪声（否则每次推理结果都不一致）
    if disable_noise:
        if hasattr(hparams, 'noise_sigma_min'):
            hparams.noise_sigma_min = 0.0
        if hasattr(hparams, 'noise_sigma_max'):
            hparams.noise_sigma_max = 0.0

    # 加载模型
    model = SnapshotDepthHS.load_from_checkpoint(ckpt_path, hparams=hparams)
    model.eval()
    model.to(device)

    # 双保险：LightningModule 内部也保存了一份 hparams
    if disable_noise and hasattr(model, 'hparams'):
        if hasattr(model.hparams, 'noise_sigma_min'):
            model.hparams.noise_sigma_min = 0.0
        if hasattr(model.hparams, 'noise_sigma_max'):
            model.hparams.noise_sigma_max = 0.0
    
    print(f"  Model loaded successfully")
    print(f"  min_depth: {model.hparams.min_depth}m, max_depth: {model.hparams.max_depth}m")
    print(f"  crop_width: {model.hparams.crop_width}")
    
    return model


def set_reproducible(seed: int, deterministic: bool = False):
    """尽量保证推理可复现。

    说明：
    - 设定 seed 本身通常足够用于“推理看看效果”。
    - deterministic=True 会强制一些 CUDA 算子走确定性路径；对本项目的光学 forward（CuBLAS matmul）
      可能直接报错，除非你额外设置 CUBLAS_WORKSPACE_CONFIG。
    """
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 说明：开启确定性可能会略慢，但推理评估更稳定
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_checkpoint(ckpt_path: str) -> str:
    """自动查找 checkpoint 文件"""
    if os.path.isfile(ckpt_path):
        return ckpt_path
    
    if os.path.isdir(ckpt_path):
        ckpt_files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
        if ckpt_files:
            # 选择最新的或按名称排序的第一个
            ckpt_files.sort(key=os.path.getmtime, reverse=True)
            print(f"Found checkpoint: {ckpt_files[0]}")
            return ckpt_files[0]
    
    raise FileNotFoundError(f"No checkpoint found at: {ckpt_path}")


def find_scenes(input_dir: str, multi_scene: bool) -> list:
    """查找场景文件"""
    scenes = []
    
    if multi_scene:
        # 查找所有 deploy X 子文件夹
        deploy_folders = glob.glob(os.path.join(input_dir, "deploy*"))
        for folder in deploy_folders:
            hs_files = glob.glob(os.path.join(folder, "*_hs.exr"))
            for hs_file in hs_files:
                depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
                if os.path.exists(depth_file):
                    scenes.append((hs_file, depth_file))
    else:
        # 单个场景文件夹
        hs_files = glob.glob(os.path.join(input_dir, "*_hs.exr"))
        for hs_file in hs_files:
            depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
            if os.path.exists(depth_file):
                scenes.append((hs_file, depth_file))
    
    return scenes


def main():
    parser = argparse.ArgumentParser(description='高光谱+深度联合重建推理')
    
    # 必需参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入数据目录（包含 *_hs.exr 和 *_depth_map.exr 文件）')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='模型 checkpoint 路径（文件或目录）')
    
    # 可选参数
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='输出目录 (default: ./inference_results)')
    parser.add_argument('--patch_size', type=int, default=512,
                       help='推理 patch 尺寸 (default: 512)')
    parser.add_argument('--multi_scene', action='store_true',
                       help='是否处理多个场景（input_dir 包含多个 deploy X 文件夹）')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=123,
                        help='随机种子（用于保证推理可复现）')
    parser.add_argument('--disable_noise', action='store_true',
                        help='关闭 SnapshotDepthHS.forward() 内部的随机噪声（强烈推荐：否则每次推理结果会抖动）')
    parser.add_argument('--deterministic', action='store_true',
                        help='强制确定性算法（CUDA 下可能需要设置 CUBLAS_WORKSPACE_CONFIG；不建议默认开启）')

    # captimgs 生成策略（解决大分辨率 OOM）
    parser.add_argument('--captimgs_mode', type=str, default='tiled', choices=['full', 'tiled'],
                        help='captimgs 生成方式：full=整图一次性(可能OOM)，tiled=分块生成(推荐)')
    parser.add_argument('--captimgs_tile_out', type=int, default=1024,
                        help='tiled 模式下每个 tile 的有效输出尺寸（越大越快但越吃显存）')
    parser.add_argument('--captimgs_offload_cpu', action='store_true',
                        help='tiled 模式下把生成的整图 captimgs 放到 CPU，重建时按 patch 搬到 GPU（更省显存）')
    parser.add_argument('--no_captimgs_oom_fallback', action='store_true',
                        help='关闭 OOM 自动回退（默认：full OOM 会自动切 tiled）')
    
    args = parser.parse_args()

    set_reproducible(args.seed, deterministic=args.deterministic)

    if args.deterministic and torch.cuda.is_available():
        print("[Info] 已开启 --deterministic。如果报 CuBLAS 非确定性错误，请在运行前设置：")
        print("       export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        print("       (或 :16:8，二选一)")

    # 你的项目在 forward() 里会无条件加噪声；为了“推理看看效果”，默认建议关闭。
    # 如果你确实想模拟拍摄噪声，请在命令行不加 --disable_noise，或改成在代码里用真实噪声参数。
    if not args.disable_noise:
        print("[Warn] --disable_noise 未开启：forward() 会注入随机噪声，结果可能每次不同。")
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    ckpt_path = find_checkpoint(args.ckpt_path)
    model = load_model(ckpt_path, device, disable_noise=args.disable_noise)
    
    # 获取深度范围（从模型 hparams）
    min_depth = model.hparams.min_depth
    max_depth = model.hparams.max_depth
    
    # 查找场景
    scenes = find_scenes(args.input_dir, args.multi_scene)
    
    if not scenes:
        print(f"Error: No valid scenes found in {args.input_dir}")
        return
    
    print(f"\nFound {len(scenes)} scene(s) to process")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个场景
    all_metrics = []
    for hs_file, depth_file in scenes:
        metrics = process_scene(
            model, hs_file, depth_file, args.output_dir,
            args.patch_size, device, min_depth, max_depth,
            disable_noise=args.disable_noise,
            captimgs_mode=args.captimgs_mode,
            captimgs_tile_out=args.captimgs_tile_out,
            captimgs_offload_cpu=args.captimgs_offload_cpu,
            captimgs_oom_fallback=(not args.no_captimgs_oom_fallback),
        )
        if metrics:
            all_metrics.append(metrics)
    
    # 汇总结果
    if all_metrics:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        avg_mae = np.mean([m['mae_depth'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse_depth'] for m in all_metrics])
        avg_psnr = np.mean([m['psnr_hs'] for m in all_metrics])
        
        print(f"  Average Depth MAE:  {avg_mae:.4f} m")
        print(f"  Average Depth RMSE: {avg_rmse:.4f} m")
        print(f"  Average HS PSNR:    {avg_psnr:.2f} dB")
        
        # 保存汇总结果
        with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
            f.write("Scene-by-Scene Results:\n")
            f.write("-" * 60 + "\n")
            for m in all_metrics:
                f.write(f"{m['scene']}: MAE={m['mae_depth']:.4f}m, RMSE={m['rmse_depth']:.4f}m, PSNR={m['psnr_hs']:.2f}dB\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"Average: MAE={avg_mae:.4f}m, RMSE={avg_rmse:.4f}m, PSNR={avg_psnr:.2f}dB\n")
        
        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()