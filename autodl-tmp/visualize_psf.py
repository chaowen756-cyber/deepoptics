#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visualize_psf.py

可视化 dump_psf.py 导出的 PSF stack（.npy）。

PSF stack 形状约为: [n_wl, n_depths, H, W]

示例：
  python visualize_psf.py --npy "psf_dump/psf_w29_d16_64x64.npy" --wl_idx 0 --depth_idx 0
  python visualize_psf.py --npy "psf_dump/psf_w29_d16_64x64.npy" --grid 4 4 --log
  python visualize_psf.py --npy "psf_dump/psf_w29_d16_64x64.npy" --wl_idx 10 --depth_idx 5 --log --save_dir psf_vis

依赖：numpy, matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np


def _maybe_center_psf(p: np.ndarray, *, shift: bool) -> np.ndarray:
    """Center PSF for visualization.

    Many optical/FFT pipelines store the impulse response with the origin at (0,0).
    When plotted directly, it looks like it's split into 4 quadrants with the
    center at image corners. `fftshift` moves it to the image center.
    """
    if not shift:
        return p
    return np.fft.fftshift(p, axes=(-2, -1))


def _pick_latest_npy(default_glob: str) -> Optional[str]:
    candidates = glob.glob(default_glob)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _auto_meta_path(npy_path: str) -> str:
    base, _ = os.path.splitext(npy_path)
    return base + "_meta.npz"


def _load_meta(meta_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not meta_path or not os.path.exists(meta_path):
        return None, None
    meta = np.load(meta_path)
    wavelengths_nm = meta.get("wavelengths_nm")
    depth_m = meta.get("depth_m")
    if wavelengths_nm is not None:
        wavelengths_nm = np.asarray(wavelengths_nm)
    if depth_m is not None:
        depth_m = np.asarray(depth_m)
    return wavelengths_nm, depth_m


def _normalize_for_display(img: np.ndarray, *, log: bool, eps: float = 1e-12) -> np.ndarray:
    x = img.astype(np.float64)
    if log:
        x = np.log10(np.maximum(x, eps))
    # Robust display range: percentiles
    lo, hi = np.percentile(x, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            hi = lo + 1.0
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 matplotlib，无法可视化。请先安装：\n"
            "  pip install matplotlib\n"
            "或在 conda 环境：\n"
            "  conda install matplotlib"
        ) from e


def _fmt_value_or_idx(value: Optional[float], idx: int, suffix: str) -> str:
    if value is None:
        return f"idx={idx}"
    return f"{value:.1f}{suffix}"


def _show_single(
    psf: np.ndarray,
    wl_idx: int,
    depth_idx: int,
    *,
    log: bool,
    save_path: Optional[str],
    wavelengths_nm: Optional[np.ndarray],
    depth_m: Optional[np.ndarray],
    shift: bool,
):
    import matplotlib.pyplot as plt

    p = psf[wl_idx, depth_idx]
    p = _maybe_center_psf(p, shift=shift)
    disp = _normalize_for_display(p, log=log)

    h, w = p.shape
    cy, cx = h // 2, w // 2

    plt.figure(figsize=(5, 5))
    plt.imshow(disp, cmap="gray", interpolation="nearest")
    wl_val = None
    if wavelengths_nm is not None and 0 <= wl_idx < len(wavelengths_nm):
        wl_val = float(wavelengths_nm[wl_idx])
    depth_val = None
    if depth_m is not None and 0 <= depth_idx < len(depth_m):
        depth_val = float(depth_m[depth_idx])

    wl_str = _fmt_value_or_idx(wl_val, wl_idx, "nm")
    depth_str = _fmt_value_or_idx(depth_val * 1e3 if depth_val is not None else None, depth_idx, "mm")

    plt.title(
        f"PSF wl={wl_str}, depth={depth_str} | sum={p.sum():.6f} | center={p[cy, cx]:.3e}"
        + (" | log10" if log else "")
        + (" | fftshift" if shift else " | raw")
    )
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def _show_grid(
    psf: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    log: bool,
    save_path: Optional[str],
    wavelengths_nm: Optional[np.ndarray],
    depth_m: Optional[np.ndarray],
    shift: bool,
):
    import matplotlib.pyplot as plt

    n_wl, n_depths, _, _ = psf.shape
    gh, gw = grid_hw

    # 均匀采样若干波长/深度层
    wl_idxs = np.linspace(0, n_wl - 1, gh).round().astype(int)
    depth_idxs = np.linspace(0, n_depths - 1, gw).round().astype(int)

    fig, axes = plt.subplots(gh, gw, figsize=(2.2 * gw, 2.2 * gh))
    if gh == 1 and gw == 1:
        axes = np.array([[axes]])
    elif gh == 1:
        axes = axes.reshape(1, -1)
    elif gw == 1:
        axes = axes.reshape(-1, 1)

    for i, wi in enumerate(wl_idxs):
        for j, di in enumerate(depth_idxs):
            p = psf[wi, di]
            p = _maybe_center_psf(p, shift=shift)
            disp = _normalize_for_display(p, log=log)
            ax = axes[i, j]
            ax.imshow(disp, cmap="gray", interpolation="nearest")

            wl_val = None
            if wavelengths_nm is not None and 0 <= wi < len(wavelengths_nm):
                wl_val = float(wavelengths_nm[wi])
            depth_val = None
            if depth_m is not None and 0 <= di < len(depth_m):
                depth_val = float(depth_m[di])

            wl_label = f"{wl_val:.0f}nm" if wl_val is not None else f"w{wi}"
            depth_label = f"{depth_val:.2f}m" if depth_val is not None else f"d{di}"
            ax.set_title(f"{wl_label} {depth_label}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"PSF grid ({gh}x{gw})" + (" | log10" if log else "") + (" | fftshift" if shift else " | raw"),
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def _show_wl_strip(
    psf: np.ndarray,
    wl_idx: int,
    depth_indices: np.ndarray,
    *,
    log: bool,
    save_path: Optional[str],
    wavelengths_nm: Optional[np.ndarray],
    depth_m: Optional[np.ndarray],
    shift: bool,
):
    """Show many depths for a single wavelength in one row."""
    import matplotlib.pyplot as plt

    n_wl, n_depths, _, _ = psf.shape
    if not (0 <= wl_idx < n_wl):
        raise ValueError(f"wl_idx 越界: {wl_idx} not in [0,{n_wl-1}]")

    depth_indices = np.asarray(depth_indices).astype(int)
    if depth_indices.ndim != 1 or depth_indices.size == 0:
        raise ValueError("depth_indices 不能为空")
    if (depth_indices < 0).any() or (depth_indices >= n_depths).any():
        raise ValueError(f"depth_indices 越界: 有值不在 [0,{n_depths-1}] 内")

    wl_val = None
    if wavelengths_nm is not None and 0 <= wl_idx < len(wavelengths_nm):
        wl_val = float(wavelengths_nm[wl_idx])
    wl_label = f"{wl_val:.1f}nm" if wl_val is not None else f"idx={wl_idx}"

    n = int(depth_indices.size)
    fig_w = max(2.0 * n, 8.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.2))
    if n == 1:
        axes = np.array([axes])

    for j, di in enumerate(depth_indices.tolist()):
        p = psf[wl_idx, di]
        p = _maybe_center_psf(p, shift=shift)
        disp = _normalize_for_display(p, log=log)

        depth_val = None
        if depth_m is not None and 0 <= di < len(depth_m):
            depth_val = float(depth_m[di])
        depth_label = f"{depth_val:.3f}m" if depth_val is not None else f"idx={di}"

        ax = axes[j]
        ax.imshow(disp, cmap="gray", interpolation="nearest")
        ax.set_title(depth_label, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"PSF by depth @ wl={wl_label}" + (" | log10" if log else "") + (" | fftshift" if shift else " | raw"),
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize PSF stack saved by dump_psf.py")
    parser.add_argument(
        "--npy",
        type=str,
        default=None,
        help="PSF .npy 路径；不填则自动取 psf_dump 下最新的一个",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="可选：对应的 meta .npz 路径（包含 wavelengths_nm / depth_m）。不填则自动尝试同名 *_meta.npz。",
    )
    parser.add_argument("--wl_idx", type=int, default=0, help="波长索引")
    parser.add_argument("--depth_idx", type=int, default=0, help="深度层索引")
    parser.add_argument(
        "--log",
        action="store_true",
        help="用 log10 强度显示（PSF 动态范围大时更直观）",
    )
    parser.add_argument(
        "--no_fftshift",
        action="store_true",
        help="默认会对 PSF 做 fftshift 以把中心移到图像中心；加此开关可查看原始(0,0)为中心的 raw 布局。",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("GH", "GW"),
        help="显示 GHxGW 的网格概览（均匀采样波长/深度）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="如果提供，则保存 PNG 到该目录；否则弹窗显示",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出所有 PSF（每个 wl/depth 一张 PNG）。必须配合 --save_dir。",
    )
    parser.add_argument(
        "--wl_strip",
        type=int,
        default=None,
        help="可选：把某一个波长下的多个深度层 PSF 画在同一张图里（横向一排），便于对比。",
    )
    parser.add_argument(
        "--strip_depth_idxs",
        type=int,
        nargs="*",
        default=None,
        help="配合 --wl_strip 使用：指定要展示的 depth 索引列表；不填则默认展示全部 depth。",
    )
    args = parser.parse_args()

    npy_path = args.npy
    if not npy_path:
        npy_path = _pick_latest_npy(os.path.join(os.path.dirname(__file__), "psf_dump", "*.npy"))
        if not npy_path:
            raise FileNotFoundError("未找到 psf_dump/*.npy；请先运行 dump_psf.py 生成 PSF")

    psf = np.load(npy_path)
    if psf.ndim != 4:
        raise ValueError(f"期望 psf 为 4D [n_wl,n_depths,H,W]，但得到 shape={psf.shape}")

    n_wl, n_depths, h, w = psf.shape
    print(f"Loaded: {npy_path}")
    print(f"Shape: [n_wl={n_wl}, n_depths={n_depths}, H={h}, W={w}]")

    meta_path = args.meta or _auto_meta_path(npy_path)
    wavelengths_nm, depth_m = _load_meta(meta_path)
    if wavelengths_nm is not None and depth_m is not None:
        print(f"Meta:   {meta_path}")
        print(f"  wavelengths_nm: {len(wavelengths_nm)}")
        print(f"  depth_m:        {len(depth_m)}")
    else:
        print("Meta:   (not found) - 将仅显示索引，不显示真实 nm/m")

    if not (0 <= args.wl_idx < n_wl):
        raise ValueError(f"--wl_idx 越界: {args.wl_idx} not in [0,{n_wl-1}]")
    if not (0 <= args.depth_idx < n_depths):
        raise ValueError(f"--depth_idx 越界: {args.depth_idx} not in [0,{n_depths-1}]")

    _ensure_matplotlib()

    shift = not args.no_fftshift

    if args.wl_strip is not None:
        if args.strip_depth_idxs is None or len(args.strip_depth_idxs) == 0:
            depth_indices = np.arange(n_depths, dtype=int)
        else:
            depth_indices = np.array(args.strip_depth_idxs, dtype=int)

        save_path = None
        if args.save_dir:
            wl_val = None
            if wavelengths_nm is not None and 0 <= int(args.wl_strip) < len(wavelengths_nm):
                wl_val = float(wavelengths_nm[int(args.wl_strip)])
            wl_tag = f"{wl_val:06.1f}nm" if wl_val is not None else f"w{int(args.wl_strip):02d}"
            save_path = os.path.join(
                args.save_dir,
                f"psf_strip_{wl_tag}_depths{len(depth_indices)}{'_log' if args.log else ''}.png",
            )

        _show_wl_strip(
            psf,
            int(args.wl_strip),
            depth_indices,
            log=args.log,
            save_path=save_path,
            wavelengths_nm=wavelengths_nm,
            depth_m=depth_m,
            shift=shift,
        )
        return 0

    if args.all:
        if not args.save_dir:
            raise ValueError("使用 --all 时必须提供 --save_dir")
        os.makedirs(args.save_dir, exist_ok=True)
        total = n_wl * n_depths
        k = 0
        for wi in range(n_wl):
            wl_val = None
            if wavelengths_nm is not None and 0 <= wi < len(wavelengths_nm):
                wl_val = float(wavelengths_nm[wi])
            for di in range(n_depths):
                depth_val = None
                if depth_m is not None and 0 <= di < len(depth_m):
                    depth_val = float(depth_m[di])

                wl_tag = f"{wl_val:06.1f}nm" if wl_val is not None else f"w{wi:02d}"
                depth_tag = f"{depth_val:05.3f}m" if depth_val is not None else f"d{di:02d}"
                out_name = f"psf_{wl_tag}_{depth_tag}{'_log' if args.log else ''}.png"
                out_path = os.path.join(args.save_dir, out_name)
                _show_single(
                    psf,
                    wi,
                    di,
                    log=args.log,
                    save_path=out_path,
                    wavelengths_nm=wavelengths_nm,
                    depth_m=depth_m,
                    shift=shift,
                )
                k += 1
                if k % 25 == 0 or k == total:
                    print(f"Exported {k}/{total}")
        print(f"Done. All PSFs saved to: {args.save_dir}")
        return 0

    if args.grid is not None:
        gh, gw = int(args.grid[0]), int(args.grid[1])
        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"psf_grid_{gh}x{gw}{'_log' if args.log else ''}.png")
        _show_grid(
            psf,
            (gh, gw),
            log=args.log,
            save_path=save_path,
            wavelengths_nm=wavelengths_nm,
            depth_m=depth_m,
            shift=shift,
        )
    else:
        save_path = None
        if args.save_dir:
            save_path = os.path.join(
                args.save_dir,
                f"psf_w{args.wl_idx:02d}_d{args.depth_idx:02d}{'_log' if args.log else ''}.png",
            )
        _show_single(
            psf,
            args.wl_idx,
            args.depth_idx,
            log=args.log,
            save_path=save_path,
            wavelengths_nm=wavelengths_nm,
            depth_m=depth_m,
            shift=shift,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
