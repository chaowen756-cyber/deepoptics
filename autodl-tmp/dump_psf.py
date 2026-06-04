#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_psf.py

用项目现有的 PSF 生成逻辑（optics/hyperspectral_camera.py -> MixedCamera.psf_at_camera）
生成并“打印所有 PSF”。

说明：
- 生成 PSF 本质上不需要场景数据；只依赖相机参数（波长、focal_length、mask 等）和输出 size。
- 但你希望在需要时使用场景数据：这里会读取 deploy 1 的 EXR，仅用于获取图像尺寸/做 sanity print。

输出：
- 终端打印：每个 (wavelength, depth-layer) 的 PSF 统计（sum/max/center）。
- 保存：完整 PSF stack 到 .npy，便于后续可视化/分析。

运行示例：
  python dump_psf.py \
    --data_dir "./deploy 1" \
    --psf_size 256 \
    --device cpu

推荐（最贴合“按项目配置生成 PSF”）：
    python dump_psf.py \
        --ckpt_path "/path/to/your.ckpt" \
        --data_dir "./deploy 1" \
        --psf_size 256 \
        --device cuda

如果你有 GPU：
  python dump_psf.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import Tuple

import numpy as np
import torch

try:
    import argparse as _argparse_for_namespace
    from snapshotdepth_hs import SnapshotDepthHS
except Exception:  # pragma: no cover
    SnapshotDepthHS = None
    _argparse_for_namespace = None

try:
    import OpenEXR  # type: ignore
    import Imath  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenEXR = None
    Imath = None

from optics import hyperspectral_camera as camera


def _resolve_ckpt_path(path: str) -> str:
    """Resolve a checkpoint path.

    Supports:
    - Direct checkpoint file path (*.ckpt / *.pth / *.pt)
    - A directory (e.g. .../checkpoints) containing checkpoints: picks newest.
    """
    if path is None:
        raise ValueError("ckpt_path is None")

    path = os.path.expanduser(path)
    if os.path.isdir(path):
        candidates = []
        for name in os.listdir(path):
            lower = name.lower()
            if lower.endswith(".ckpt") or lower.endswith(".pth") or lower.endswith(".pt"):
                candidates.append(os.path.join(path, name))

        if not candidates:
            raise FileNotFoundError(f"目录中未找到 checkpoint 文件（*.ckpt/*.pth/*.pt）: {path}")

        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _safe_print(*args, **kwargs):
    """print() that exits quietly if stdout pipe is closed (e.g., piped to head)."""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)


def read_exr(file_path: str) -> np.ndarray:
    """读取 EXR 文件并返回 float32 数组 (H, W, C)。"""
    if OpenEXR is None or Imath is None:
        raise RuntimeError(
            "当前环境未安装 OpenEXR/Imath，无法读取 .exr。"
            "PSF 生成不依赖场景像素，你可以忽略此项，或安装依赖后再启用读取。"
        )
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"不是有效的 EXR 文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header["channels"]
    channel_names = sorted(channels_info.keys())
    if not channel_names:
        raise ValueError(f"EXR 文件无通道: {file_path}")

    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"不支持的 EXR 数据类型: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)
    np_channels = []
    for i, _name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    return image_np.astype(np.float32)


def _hparams_from_checkpoint(ckpt_path: str):
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "hyper_parameters" not in checkpoint:
        raise ValueError("Checkpoint 中没有找到 hyper_parameters")

    hparams_dict = checkpoint["hyper_parameters"]
    if "hparams" in hparams_dict and isinstance(hparams_dict["hparams"], (dict, argparse.Namespace)):
        hparams_obj = hparams_dict["hparams"]
        if isinstance(hparams_obj, dict):
            return argparse.Namespace(**hparams_obj)
        return hparams_obj

    return argparse.Namespace(**hparams_dict)


def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    if SnapshotDepthHS is None:
        raise RuntimeError(
            "无法导入 SnapshotDepthHS（可能缺 pytorch_lightning 或导入路径问题）。"
            "如果你不需要按 checkpoint 复用配置，请不传 --ckpt_path，改用手动相机参数生成 PSF。"
        )

    ckpt_path = _resolve_ckpt_path(ckpt_path)
    _safe_print(f"Resolved ckpt: {ckpt_path}")

    hparams = _hparams_from_checkpoint(ckpt_path)
    model = SnapshotDepthHS.load_from_checkpoint(ckpt_path, hparams=hparams)
    model.eval()
    model.to(device)
    return model


def build_camera(
    *,
    image_size: Tuple[int, int],
    hs_channels: int,
    start_wl: float,
    end_wl: float,
    min_depth: float,
    max_depth: float,
    n_depths: int,
    focal_depth: float,
    focal_length: float,
    f_number: float,
    camera_pixel_pitch: float,
    mask_sz: int,
    mask_upsample_factor: int,
    diffraction_efficiency: float,
    full_size: int,
    use_virtual_lens_phase: bool,
    requires_grad: bool,
) -> camera.MixedCamera:
    mask_diameter = focal_length / f_number
    wavelengths = np.linspace(start_wl, end_wl, hs_channels)

    cam = camera.MixedCamera(
        wavelengths=wavelengths,
        min_depth=min_depth,
        max_depth=max_depth,
        focal_depth=focal_depth,
        n_depths=n_depths,
        image_size=list(image_size),
        camera_pixel_pitch=camera_pixel_pitch,
        focal_length=focal_length,
        mask_diameter=mask_diameter,
        mask_size=mask_sz,
        mask_upsample_factor=mask_upsample_factor,
        diffraction_efficiency=diffraction_efficiency,
        full_size=full_size,
        use_virtual_lens_phase=use_virtual_lens_phase,
        requires_grad=requires_grad,
    )
    return cam


@torch.no_grad()
def main() -> int:
    # 当输出被管道到 `head`/`tail` 等工具时，对端可能提前关闭管道。
    # Python 默认会忽略 SIGPIPE 并在写 stdout 时抛 BrokenPipeError。
    # 这里设置为默认行为 + 兜底捕获，确保脚本安静退出。
    if hasattr(signal, "SIGPIPE"):
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Generate and print all PSFs using project camera logic")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="可选：Lightning checkpoint 路径。提供后将复用项目训练时的相机配置生成 PSF。",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "deploy 1"),
        help="包含 scene01_hs.exr / scene01_depth_map.exr 的目录",
    )
    parser.add_argument("--hs_file", type=str, default="scene01_hs.exr")
    parser.add_argument("--depth_file", type=str, default="scene01_depth_map.exr")

    parser.add_argument("--psf_size", type=int, default=256, help="生成的 PSF 输出尺寸（正方形）")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="运行设备")
    parser.add_argument(
        "--max_print",
        type=int,
        default=None,
        help="最多打印多少个 (wl,depth) 的 PSF 统计；默认打印全部。建议大模型时设一个值避免刷屏。",
    )

    # 相机/深度参数（默认值尽量与项目里的常用设置一致）
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--n_depths", type=int, default=8)
    parser.add_argument("--focal_depth", type=float, default=1.0)

    parser.add_argument("--hs_channels", type=int, default=29)
    parser.add_argument("--start_wl", type=float, default=420e-9)
    parser.add_argument("--end_wl", type=float, default=700e-9)

    parser.add_argument("--focal_length", type=float, default=50e-3)
    parser.add_argument("--f_number", type=float, default=6.3)
    parser.add_argument("--camera_pixel_pitch", type=float, default=6.5e-6)

    parser.add_argument("--mask_sz", type=int, default=256)
    parser.add_argument("--mask_upsample_factor", type=int, default=1)

    parser.add_argument("--diffraction_efficiency", type=float, default=0.7)
    parser.add_argument("--full_size", type=int, default=1920)

    parser.add_argument(
        "--use_virtual_lens_phase",
        action="store_true",
        help="在 pupil 处叠加理想薄透镜相位（默认关闭）",
    )
    parser.add_argument(
        "--no-use_virtual_lens_phase",
        dest="use_virtual_lens_phase",
        action="store_false",
    )
    parser.set_defaults(use_virtual_lens_phase=False)

    parser.add_argument(
        "--requires_grad",
        action="store_true",
        help="是否让 DOE 参数可导（仅影响参数 requires_grad；不影响 PSF 数值）",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    hs_path = os.path.join(data_dir, args.hs_file)
    depth_path = os.path.join(data_dir, args.depth_file)

    # 读取场景数据（仅用于确认尺寸/深度范围；PSF 生成本身不依赖这些像素值）
    _safe_print("=" * 72)
    _safe_print("Scene sanity (optional)")
    _safe_print(f"  hs_path:    {hs_path}")
    _safe_print(f"  depth_path: {depth_path}")
    if not (os.path.exists(hs_path) and os.path.exists(depth_path)):
        _safe_print("  [Skip] deploy 1 场景文件不存在或路径不对，跳过 EXR 读取（不影响 PSF 生成）。")
    elif OpenEXR is None:
        _safe_print("  [Skip] OpenEXR/Imath 未安装，跳过 EXR 读取（不影响 PSF 生成）。")
        _safe_print("  [Hint] 若要读取 EXR，请安装依赖：pip install OpenEXR Imath")
    else:
        hs = read_exr(hs_path)
        depth = read_exr(depth_path)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depth_m = depth.astype(np.float32) / 1000.0
        _safe_print(f"  hs.shape:   {hs.shape} (H,W,C)")
        _safe_print(f"  depth.shape:{depth.shape} (H,W)")
        _safe_print(f"  depth(m) range: [{float(depth_m.min()):.4f}, {float(depth_m.max()):.4f}]")
    _safe_print("=" * 72)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        _safe_print("[Warn] CUDA 不可用，自动切回 CPU")

    # PSF 生成的输出尺寸：用 args.psf_size（避免直接用整图导致内存爆炸）
    psf_size = int(args.psf_size)

    if args.ckpt_path:
        _safe_print(f"Loading checkpoint (for camera config): {args.ckpt_path}")
        model = _load_model_from_ckpt(args.ckpt_path, device)
        cam = model.camera
        cam.eval()
        _safe_print("Checkpoint camera setup")
        _safe_print(f"  hs_channels: {getattr(model.hparams, 'hs_channels', 'N/A')}")
        _safe_print(
            f"  depth range: {getattr(model.hparams, 'min_depth', 'N/A')}m ~ {getattr(model.hparams, 'max_depth', 'N/A')}m, "
            f"n_depths={getattr(model.hparams, 'n_depths', 'N/A')}, focal_depth={getattr(model.hparams, 'focal_depth', 'N/A')}"
        )
        _safe_print(f"  use_virtual_lens_phase: {getattr(model.hparams, 'use_virtual_lens_phase', 'N/A')}")
        _safe_print("=" * 72)
    else:
        cam = build_camera(
            image_size=(psf_size, psf_size),
            hs_channels=args.hs_channels,
            start_wl=args.start_wl,
            end_wl=args.end_wl,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            n_depths=args.n_depths,
            focal_depth=args.focal_depth,
            focal_length=args.focal_length,
            f_number=args.f_number,
            camera_pixel_pitch=args.camera_pixel_pitch,
            mask_sz=args.mask_sz,
            mask_upsample_factor=args.mask_upsample_factor,
            diffraction_efficiency=args.diffraction_efficiency,
            full_size=args.full_size,
            use_virtual_lens_phase=args.use_virtual_lens_phase,
            requires_grad=args.requires_grad,
        ).to(device)

        cam.eval()

    # 生成 PSF stack: [n_wl, n_depths, H, W]
    psf = cam.psf_at_camera(size=(psf_size, psf_size), is_training=False)

    # 打印一些关键信息
    wavelengths_nm = cam.wavelengths.detach().cpu().numpy() * 1e9
    scene_distances_m = cam.scene_distances.detach().cpu().numpy()

    _safe_print("Camera setup")
    _safe_print(f"  use_virtual_lens_phase: {cam.use_virtual_lens_phase}")
    _safe_print(f"  n_wl: {len(wavelengths_nm)}, n_depths: {len(scene_distances_m)}")
    _safe_print(f"  depth layers (m): {np.array2string(scene_distances_m, precision=4, separator=', ')}")
    _safe_print(f"  wavelengths (nm): {np.array2string(wavelengths_nm, precision=1, separator=', ')}")
    _safe_print("=" * 72)

    psf_cpu = psf.detach().cpu().float().numpy()

    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), "psf_dump_DOEpretrain")
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"psf_w{len(wavelengths_nm)}_d{len(scene_distances_m)}_{psf_size}x{psf_size}.npy")
    np.save(npy_path, psf_cpu)

    # 额外保存元数据（便于可视化时叠加真实波长/深度）
    meta_path = npy_path.replace(".npy", "_meta.npz")
    np.savez(
        meta_path,
        wavelengths_nm=wavelengths_nm.astype(np.float32),
        depth_m=scene_distances_m.astype(np.float32),
        psf_size=np.array([psf_size], dtype=np.int32),
    )

    # 打印“所有 PSF”的统计（逐张打印，避免打印整个 2D 数组）
    # 你如果真的要逐元素打印，可以后续对 npy 做处理，但不建议直接刷终端。
    center = (psf_size // 2, psf_size // 2)
    printed = 0
    for wi, wl_nm in enumerate(wavelengths_nm):
        for di, depth_mi in enumerate(scene_distances_m):
            p = psf_cpu[wi, di]
            p_sum = float(p.sum())
            p_max = float(p.max())
            p_min = float(p.min())
            p_ctr = float(p[center])
            _safe_print(
                f"PSF wl[{wi:02d}]={wl_nm:7.1f}nm  depth[{di:02d}]={depth_mi:6.3f}m  "
                f"shape={p.shape}  sum={p_sum:.6f}  max={p_max:.6e}  min={p_min:.6e}  center={p_ctr:.6e}"
            )
            printed += 1
            if args.max_print is not None and printed >= int(args.max_print):
                _safe_print(f"[Info] Reached --max_print={args.max_print}, stop printing.")
                wi = len(wavelengths_nm)  # break outer
                break
        if args.max_print is not None and printed >= int(args.max_print):
            break

    _safe_print("=" * 72)
    _safe_print("Done")
    _safe_print(f"  Saved PSF stack: {npy_path}")
    _safe_print(f"  Saved PSF meta:  {meta_path}")
    _safe_print("Note: PSF generation does NOT require scene pixels; scene EXR was read for sanity only.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # 兼容某些环境在解释器退出时才触发 BrokenPipe 的情况
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)