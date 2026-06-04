# optics/hyperspectral_camera.py (最终的最终修正版)
import os
import tifffile
import abc
import math
from typing import List, Union
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import complex, cubicspline
from util.fft import fftshift
from util.helper import copy_quadruple, depthmap_to_layereddepth, heightmap_to_phase, ips_to_metric, over_op, \
    refractive_index


class BaseCamera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, focal_depth, min_depth, max_depth, n_depths, image_size, mask_size,
                 focal_length, mask_diameter, camera_pixel_pitch, wavelengths, **kwargs):
        super().__init__()
        # ============================================================
        # 【IPS 迁移】使用逆深度均匀采样，与 Dataset 对齐
        # ============================================================
        # Dataset 侧使用 IPS 归一化：d_norm = (1/d - 1/d_max) / (1/d_min - 1/d_max)
        # 为了对齐，Camera 生成 PSF 时的 scene_distances 也要在 IPS 空间均匀分布
        # 
        # 策略：在 IPS 空间 [0, 1] 等间距采样，然后反算到物理深度
        # ips_coords = [0, 1/(n-1), 2/(n-1), ..., 1]
        # 物理深度: d = ips_to_metric(ips_coords, min_depth, max_depth)
        #
        # 这样确保：Dataset 输出 d_norm=0.5 -> Camera Layer 4 的 PSF 物理对应
        # ============================================================
        
        # 在 IPS 归一化空间 [0, 1] 等间距采样
        ips_normalized = torch.linspace(0, 1, steps=n_depths)
        
        # 反算回物理深度，使用 ips_to_metric 函数
        # d = (d_max * d_min) / (d_max - (d_max - d_min) * ips_norm)
        scene_distances = ips_to_metric(ips_normalized, min_depth, max_depth)
        self._register_wavlength(wavelengths)
        self.n_depths, self.min_depth, self.max_depth = len(scene_distances), min_depth, max_depth
        self.focal_depth, self.mask_diameter, self.camera_pixel_pitch, self.focal_length = focal_depth, mask_diameter, camera_pixel_pitch, focal_length
         # 是否在 pupil 处叠加一个“理想薄透镜”的相位（用于传统成像/对焦建模）。
        # 对 Baek-like “DOE 充当唯一主透镜”的设定，应关闭此项，否则会出现双重透镜叠加。
        self.use_virtual_lens_phase = bool(kwargs.get('use_virtual_lens_phase', True))
        self.f_number = self.focal_length / self.mask_diameter
        self.image_size = self._normalize_image_size(image_size)
        self.mask_pitch, self.mask_size = self.mask_diameter / mask_size, mask_size
        self.register_buffer('scene_distances', scene_distances)
        self.build_camera()

    def _register_wavlength(self, wavelengths):
        if isinstance(wavelengths, (list, np.ndarray)): wavelengths = torch.tensor(wavelengths)
        self.n_wl = len(wavelengths)
        self.register_buffer('wavelengths', wavelengths)

    def _capture_impl(self, volume, layered_depth, psf, occlusion, eps=1e-8):
        scale = volume.max();
        if scale > 0: volume = volume / scale
        Fpsf = torch.fft.rfft2(psf, dim=(-2, -1))
        if occlusion:
            print(f"使用了occlusion！！！！")
            # 进行傅立叶计算 转换到频域
            Fvolume = torch.fft.rfft2(volume, dim=(-2, -1))
            Flayered_depth = torch.fft.rfft2(layered_depth, dim=(-2, -1))
            # 卷积FPSF和Flayer_depth 模拟前景物体失焦的时候 其遮挡边缘也会模糊的情况
            # layer_depth 每一层清晰的0/1区域也被模糊成半透明区域 模拟前后物体交界处平滑模糊
            blurred_alpha = torch.fft.irfft2(Fpsf * Flayered_depth, s=volume.shape[-2:], dim=(-2, -1))
            # 对场景颜色进行模糊处理 同上 普适意义物体失焦上的模糊
            blurred_volume = torch.fft.irfft2(Fpsf * Fvolume, s=volume.shape[-2:], dim=(-2, -1))
            # 计算累计不透明度 按深度顺序 第d层的alpha 是本层的不透明度累加至N层的总和 并转换到频域
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
            Fcumsum_alpha = torch.fft.rfft2(cumsum_alpha, dim=(-2, -1))
            # 模糊累计不透明度 .irfft2()将其转换回图像空间
            blurred_cumsum_alpha = torch.fft.irfft2(Fpsf * Fcumsum_alpha, s=volume.shape[-2:], dim=(-2, -1))
            # 进行归一化 利用上一部得到的模糊后的累计不透明度（基准） 层层对应计算 还原出物体原始的颜色属性和不透明度
            # 此时的值不再是累加的值 而是一个修正校准的真实的 不透明度
            blurred_volume /= (blurred_cumsum_alpha + eps);
            blurred_alpha /= (blurred_cumsum_alpha + eps)
            # 从前往后算 算每一层的透明度与blurred_volume的乘积 再进行结果叠加
            captimg = torch.sum(over_op(blurred_alpha) * blurred_volume, dim=-3)
        else:
            Fvolume = torch.fft.rfft2(volume, dim=(-2, -1))
            captimg = torch.fft.irfft2(torch.sum(Fpsf * Fvolume, dim=2), s=volume.shape[-2:], dim=(-2, -1))
        if scale > 0: captimg *= scale; volume *= scale
        return captimg, volume

    def _capture_from_rgbd_with_psf_impl(self, img, depthmap, psf, occlusion):

        if depthmap.ndim == 3:
            depthmap = depthmap.unsqueeze(1) # -> [B, 1, H, W]
        
        layered_depth = depthmap_to_layereddepth(depthmap, self.n_depths, binary=True)


        
        # 2. 获取维度信息
        B, C, H, W = img.shape
        # 此时 layered_depth 应该是 (Batch, Depth, H, W)
        
        # 3. 显式重塑以进行广播
        # 目标 volume 形状: (Batch, Channel, Depth, H, W)
        
        # img: (B, C, H, W) -> (B, C, 1, H, W)
        img_expanded = img.view(B, C, 1, H, W)
        
        # 4. 计算 Volume
        # (2, 29, 1, 512, 512) * (2, 1, 16, 512, 512) -> (2, 29, 16, 512, 512)
        volume = layered_depth * img_expanded
        return self._capture_impl(volume, layered_depth > 0, psf, occlusion)

    def forward(self, img, depthmap, occlusion, is_training=False):
        psf = self.psf_at_camera(size=img.shape[-2:], is_training=is_training).unsqueeze(0)
        psf = self.normalize_psf(psf)
        captimg, volume = self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    @abc.abstractmethod
    def build_camera(self):
        pass

    def sensor_distance(self):
        return 1. / (1. / self.focal_length - 1. / self.focal_depth)

    def normalize_psf(self, psfimg):
        return psfimg / (psfimg.sum(dim=(-2, -1), keepdims=True) + 1e-8)

    def _normalize_image_size(self, image_size):
        if isinstance(image_size, int): image_size = [image_size, image_size]
        return image_size


class BaseRotationallySymmetricCamera(BaseCamera):
    def __init__(self, *args, full_size=1920, **kwargs):
        self.full_size = self._normalize_image_size(full_size)
        super().__init__(*args, **kwargs)

    def build_camera(self):
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(self.scene_distances)
        H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        ind = self.find_index(rho_grid, rho_sampling)
        H_full, rho_grid_full, _ = self.precompute_H(self.full_size)
        self.register_buffer('prop_amplitude', prop_amplitude);
        self.register_buffer('prop_phase', prop_phase)
        self.register_buffer('H', H);
        self.register_buffer('rho_grid', rho_grid);
        self.register_buffer('rho_sampling', rho_sampling);
        self.register_buffer('ind', ind)
        self.register_buffer('H_full', H_full);
        self.register_buffer('rho_grid_full', rho_grid_full)

    def pointsource_inputfield1d(self, scene_distances):
        r = (self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2)).double().to(
            scene_distances.device)
        wavelengths, scene_distances = self.wavelengths.double().reshape(-1, 1, 1), scene_distances.double().reshape(1,
                                                                                                                     -1,
                                                                                                                     1)
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths
        radius = torch.sqrt(scene_distances ** 2 + r ** 2)
        amplitude = scene_distances / wavelengths / radius ** 2
        phase = wave_number * (radius - scene_distances)
#         if not math.isinf(self.focal_depth):
        if self.use_virtual_lens_phase and (not math.isinf(self.focal_depth)):
            focal_depth = torch.tensor(self.focal_depth).double().reshape(1, 1, 1).to(r.device)
            phase -= wave_number * (torch.sqrt(focal_depth ** 2 + r ** 2) - focal_depth)
        return amplitude / amplitude.max(), phase

    def heightmap(self):
        heightmap1d = torch.cat([self.heightmap1d().cpu(), torch.zeros((self.mask_size // 2))], dim=0).reshape(1, 1, -1)
        r_grid = torch.arange(0, self.mask_size, dtype=torch.double).reshape(1, -1)
        y_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        ind = self.find_index(r_grid, r_coord)
        heightmap_quarter = cubicspline.interp(r_grid, heightmap1d, r_coord, ind).float()
        return copy_quadruple(heightmap_quarter).squeeze()

    def find_index(self, a, v):
        a = a.squeeze(1).cpu().numpy();
        v = v.cpu().numpy()
        return torch.from_numpy(
            np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0))

    def precompute_H(self, image_size):
        coords = [self.camera_pixel_pitch * torch.arange(1, s // 2 + 1) for s in image_size]
        rho_sampling = torch.sqrt(coords[0].double().reshape(-1, 1) ** 2 + coords[1].double().reshape(1, -1) ** 2)
        max_dim = max(image_size)
        rho_grid = math.sqrt(2) * self.camera_pixel_pitch * (
                    torch.arange(-1, max_dim // 2 + 1, dtype=torch.double) + 0.5)
        sensor_dist = self.sensor_distance()
        rho_grid = rho_grid.reshape(1, 1, -1) / (self.wavelengths.reshape(-1, 1, 1) * sensor_dist)
        rho_sampling = rho_sampling.unsqueeze(0) / (self.wavelengths.reshape(-1, 1, 1) * sensor_dist)
        r = (self.mask_pitch * torch.linspace(1, self.mask_size / 2, self.mask_size // 2)).double().reshape(1, -1, 1)
        J = torch.where(rho_grid == 0, 0.5 * r ** 2,
                        (r / (2 * math.pi * rho_grid)) * scipy.special.jv(1, 2 * math.pi * rho_grid * r))
        return torch.cat([J[:, 0:1, :], J[:, 1:, :] - J[:, :-1, :]], dim=1), rho_grid.squeeze(1), rho_sampling


class RotationallySymmetricCamera(BaseRotationallySymmetricCamera):
    def __init__(self, *args, mask_upsample_factor=1, requires_grad=False, **kwargs):
        super().__init__(*args, **kwargs)
#         init_heightmap = torch.zeros(self.mask_size // 2 // mask_upsample_factor)
         # ============================================================
        # 菲涅尔透镜初始化 (Fresnel DOE Lens Pattern)
        # ============================================================
        # 目标：打破对称性，解决深度反转问题
        # 方法：使用传统菲涅尔透镜模式初始化 DOE，赋予系统初始光焦度
        
        # 1. 计算物理坐标
        n_samples = self.mask_size // 2 // mask_upsample_factor
        # 每个可优化参数点对应的物理径向坐标
        r_indices = torch.arange(n_samples, dtype=torch.float32)
        r_physical = r_indices * self.mask_pitch * mask_upsample_factor
        
        # 2. 设计基准：选取中心波长
        mid_idx = len(self.wavelengths) // 2
        design_wavelength = self.wavelengths[mid_idx].item()
        n_design = refractive_index(torch.tensor(design_wavelength)).item()
        
        # 3. 高度与相位公式
        # 产生 2π 相位延迟的物理高度周期
        h_max = design_wavelength / (n_design - 1.0)
        
        # 4. 透镜参数：使用主透镜焦距
        f_lens = self.focal_length
        
        # 5. 薄透镜的理想抛物面矢高公式
        # h_ideal(r) = r^2 / (2 * f * (n - 1))
        h_ideal = (r_physical ** 2) / (2.0 * f_lens * (n_design - 1.0))
        
        # 6. 菲涅尔折叠 (Fresnel Wrapping)
        # 通过取模运算将抛物面折叠为菲涅尔结构
        # 公式：h_init = h_max - (h_ideal mod h_max)
        # 目的：生成聚光透镜（中心厚/相位延迟大，边缘薄）
        h_wrapped = torch.fmod(h_ideal, h_max)
        init_heightmap = h_max - h_wrapped
        
        # 7. 打破对称性：加入极微量的高斯噪声
        # 防止优化器因数值完全对称而无法启动
        noise = torch.randn_like(init_heightmap) * 1e-6 * h_max
        init_heightmap = init_heightmap + noise
        
        # 8. 确保非负（物理约束）
        init_heightmap = torch.clamp(init_heightmap, min=0.0)

        self.heightmap1d_ = nn.Parameter(init_heightmap, requires_grad=requires_grad)
# ============================================================
#         # 方案：加载预训练的 TIF DOE 文件 (Pre-trained Loading)
#         # ============================================================
#         tif_path = "autodl-tmp/doe_design.tif"  # <---【请修改这里为你的真实路径】
        
#         if os.path.exists(tif_path):
#             print(f">>> Loading DOE from TIF: {tif_path} <<<")
            
#             # 1. 读取 TIF 数据
#             doe_data = tifffile.imread(tif_path).astype(np.float32)
            
#             # 2. 提取径向轮廓 (Radial Profile)
#             # 因为是对称的，直接取中心行的一半即可
#             h, w = doe_data.shape
#             cy, cx = h // 2, w // 2
            
#             # 从中心向右提取 (Radius)
#             # 注意：这里假设 TIF 的中心就是图像中心。你的检查结果显示是对称的，所以没问题。
#             radial_profile_np = doe_data[cy, cx:] 
            
#             # 3. 尺寸对齐 (Resampling)
#             # TIF 的半径像素数可能和你的 mask_size 不一样，需要插值缩放
#             # 目标尺寸:
#             n_target = self.mask_size // 2 // mask_upsample_factor
            
#             # 转为 Tensor 并增加维度用于 interpolate: [1, 1, Src_Len]
#             profile_tensor = torch.from_numpy(radial_profile_np).reshape(1, 1, -1)
            
#             # 线性插值到目标长度
#             init_heightmap = F.interpolate(
#                 profile_tensor, 
#                 size=n_target, 
#                 mode='linear', 
#                 align_corners=False
#             ).reshape(-1) # 变回 1D 向量
            
#             # 4. 强制非负 (物理高度不能为负)
#             init_heightmap = torch.clamp(init_heightmap, min=0.0)
            
#             # 5. 【关键】设置为不可训练 (如果你想“免DOE训练”)
#             # 如果 requires_grad=False，参数会被固定，网络只学后端重建。
#             self.heightmap1d_ = nn.Parameter(init_heightmap, requires_grad=requires_grad)
            
#         else:
#             raise FileNotFoundError(f"找不到 DOE 文件: {tif_path}")
            
        self.mask_upsample_factor = mask_upsample_factor

    def heightmap1d(self):
        return F.interpolate(self.heightmap1d_.reshape(1, 1, -1), scale_factor=self.mask_upsample_factor,
                             mode='nearest').reshape(-1)


class MixedCamera(RotationallySymmetricCamera):
    def __init__(self, *args, diffraction_efficiency=0.7, **kwargs):
        self.diffraction_efficiency = diffraction_efficiency
        super().__init__(*args, **kwargs)

    def build_camera(self):
        H, rho_grid, rho_sampling = self.precompute_H(self.image_size)
        ind = self.find_index(rho_grid, rho_sampling)
        H_full, rho_grid_full, _ = self.precompute_H(self.full_size)
        self.register_buffer('H', H);
        self.register_buffer('rho_grid', rho_grid)
        self.register_buffer('rho_sampling', rho_sampling);
        self.register_buffer('ind', ind)
        self.register_buffer('H_full', H_full);
        self.register_buffer('rho_grid_full', rho_grid_full)

    def psf1d(self, H, scene_distances, modulate_phase=True):
        prop_amplitude, prop_phase = self.pointsource_inputfield1d(scene_distances)
        H, wavelengths = H.unsqueeze(1), self.wavelengths.double().reshape(-1, 1, 1).to(H.device)
        phase = prop_phase
        if modulate_phase:
            phase_delays = heightmap_to_phase(self.heightmap1d(), wavelengths, refractive_index(wavelengths))
            phase += phase_delays.reshape(wavelengths.shape[0], 1, -1)
        real = torch.matmul(prop_amplitude.unsqueeze(2) * torch.cos(phase.unsqueeze(2)), H).squeeze(-2)
        imag = torch.matmul(prop_amplitude.unsqueeze(2) * torch.sin(phase.unsqueeze(2)), H).squeeze(-2)
        return (2 * math.pi / wavelengths / self.sensor_distance()) ** 2 * (real ** 2 + imag ** 2)

    def _psf_at_camera_impl(self, H, rho_grid, rho_sampling, ind, size, scene_distances, modulate_phase):
        psf1d = self.psf1d(H, scene_distances, modulate_phase)
        rho_sampling, ind = rho_sampling.to(psf1d.device), ind.to(psf1d.device)
        psf_rd = F.relu(cubicspline.interp(rho_grid, psf1d, rho_sampling, ind).float())
        return copy_quadruple(psf_rd.reshape(self.n_wl, -1, size[0] // 2, size[1] // 2))

    def psf_at_camera(self, size=None, modulate_phase=True, is_training=False):
        device = self.H.device
#         if is_training:
#             sd = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device) +
#                                (torch.rand(self.n_depths, device=device) - 0.5) / self.n_depths,
#                                self.min_depth, self.max_depth)
#             # ####################################################################
#             # ## 核心修改点：将 .item() 或 [0] 添加到随机数生成中 ##
#             # ####################################################################
#             rand_val = torch.rand(1, device=device).item() * (100.0 - self.max_depth)
#             sd[-1] += rand_val
#         else:
# #             sd = self.scene_distances.to(device)
#             sd = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
#                                         self.min_depth, self.max_depth)
        # 【IPS 迁移】生成与 Dataset 对齐的深度采样点
        # 在 IPS 归一化空间 [0, 1] 等间距采样，然后反算回物理深度
        ips_normalized = torch.linspace(0, 1, steps=self.n_depths, device=device)
        
        if is_training:
            # 2. [保留] 随机抖动 (Jitter)
            # 这有助于网络学会连续的深度，而不是死记硬背几个离散的深度点
            # 这里的抖动是在 bin 宽度内微调，是好的
            jitter = 1.0 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5)
            ips_jittered = torch.clamp(ips_normalized + jitter, 0, 1)
            
            # 3. 反算到物理深度（使用 IPS 逆函数）
            scene_distances = ips_to_metric(ips_jittered, self.min_depth, self.max_depth)
            
            # ==========================================================
            # ❌ [删除/注释] 这一行绝对不能要！
            # 你的 2.0m 是物体，不是无穷远背景。不要把它推到 100米去。
            # ==========================================================
            # scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
            
        else:
            # 测试时保持稳定的线性采样
#             scene_distances = self.min_depth + (self.max_depth - self.min_depth) * normalized_depths
            scene_distances = ips_to_metric(ips_normalized, self.min_depth, self.max_depth)
        diff_psf = self._psf_at_camera_impl(self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances,
                                            modulate_phase)
        undiff_psf = self._psf_at_camera_impl(self.H, self.rho_grid, self.rho_sampling, self.ind, self.image_size, scene_distances,
                                              False)
        self.diff_norm = diff_psf.sum(dim=(-1, -2), keepdims=True)
        self.undiff_norm = undiff_psf.sum(dim=(-1, -2), keepdims=True)
        psf = self.diffraction_efficiency * self.normalize_psf(diff_psf) + \
              (1 - self.diffraction_efficiency) * self.normalize_psf(undiff_psf)
        if size:
            pad = [(s - i) // 2 for s, i in zip(size, self.image_size)]
            psf = F.pad(psf, (pad[1], pad[1], pad[0], pad[0]))
        return fftshift(psf, dims=(-1, -2))

    def psf_out_of_fov_energy(self, psf_size: int):
        device = self.H.device
#         sd = self.scene_distances.to(self.H.device)

        sd = scene_distances = ips_to_metric(torch.linspace(0, 1, steps=self.n_depths, device=device),
                                         self.min_depth, self.max_depth)
        psf1d = self.psf1d_full(sd)
        if hasattr(self, 'diff_norm'): psf1d /= self.diff_norm.squeeze(-1)
        edge = psf_size / 2 * self.camera_pixel_pitch / (
                    self.wavelengths.reshape(-1, 1, 1).to(sd.device) * self.sensor_distance())
        return (psf1d * (self.rho_grid_full.unsqueeze(1).to(sd.device) > edge).float()).sum(), psf1d.max()

    def psf1d_full(self, scene_distances):
        return self.psf1d(self.H_full, scene_distances, modulate_phase=True)

    def forward_train(self, img, depthmap, occlusion):
        return self.forward(img, depthmap, occlusion, is_training=True)