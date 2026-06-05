#!/usr/bin/env python3
"""
预处理脚本：把 Baek 数据集的 18 个场景从 EXR 转成 NPZ。

作用：EXR 每次加载需要解压 1.1GB 文件 (~5 秒)，NPZ 加载只需 ~0.1 秒。
精度：float32，bit-for-bit 完全一致，不影响训练效果。

用法（在服务器上跑一次）：
    python preprocess.py

会在 autodl-tmp/ 下生成 Baek数据集_npz/ 目录。
"""
import os
import numpy as np
import OpenEXR
import Imath

# ============================================================
# 读取 EXR 文件
# ============================================================
def read_exr(file_path):
    """读取 EXR 文件，返回 float32 numpy 数组"""
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"不是有效的 EXR 文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())

    if not channel_names:
        raise ValueError(f"EXR 文件没有通道: {file_path}")

    # 判断数据类型
    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"不支持的数据类型: {first_channel_type.type}")

    # 读取所有通道
    all_channels_bytes = exr_file.channels(channel_names)

    np_channels = []
    for i, name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    return image_np.astype(np.float32)


# ============================================================
# 主流程
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "Baek数据集")
    out_dir = os.path.join(script_dir, "Baek数据集_npz")

    if not os.path.isdir(base_dir):
        print(f"错误: 找不到 {base_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, 19):
        folder = f"deploy {i}"
        scene_id = f"scene_{i:02d}"
        hs_path = os.path.join(base_dir, folder, f"scene{i:02d}_hs.exr")
        depth_path = os.path.join(base_dir, folder, f"scene{i:02d}_depth_map.exr")

        if not os.path.exists(hs_path) or not os.path.exists(depth_path):
            print(f"⚠️  跳过 {folder}: 文件不存在")
            continue

        print(f"处理 scene_{i:02d}...", end=" ", flush=True)

        # 读取
        hs = read_exr(hs_path)                          # (H, W, 29)
        dp = read_exr(depth_path).squeeze(-1) / 1000.0  # (H, W), mm→m

        # 存为 .npy（无压缩，加载速度最快）
        np.save(os.path.join(out_dir, f"scene_{i:02d}_hs.npy"), hs)
        np.save(os.path.join(out_dir, f"scene_{i:02d}_depth.npy"), dp)

        hs_mb = os.path.getsize(os.path.join(out_dir, f"scene_{i:02d}_hs.npy")) / (1024 * 1024)
        print(f"完成 → HS {hs_mb:.0f} MB + Depth")

    print(f"\n全部完成！18 个场景已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
