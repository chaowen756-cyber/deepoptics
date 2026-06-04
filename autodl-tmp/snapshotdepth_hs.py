# snapshotdepth_hs.py (最终完整版)

import copy
from argparse import ArgumentParser
from collections import namedtuple
import numpy as np

import pytorch_lightning as pl
import torch
import torch.optim
import torchvision.transforms
import torchvision.utils
# from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError
# PL 1.5+: metrics 迁移到 torchmetrics
try:
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
except ImportError:
    try:
        from torchmetrics import MeanAbsoluteError, MeanSquaredError
    except ImportError:
        # 最后兜底：老版本 PL
        from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError

# 导入我们修改过的文件
from models.simple_model_hs import SimpleModelHS as SimpleModel
from optics import hyperspectral_camera as camera
from util.hs_loss import CombinedLoss

# 导入项目原有的辅助工具
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize

SnapshotOutputs = namedtuple('SnapshotOutputs',
                             field_names=['captimgs', 'captimgs_linear',
                                          'est_images', 'est_depthmaps',
                                          'target_images', 'target_depthmaps',
                                          'psf'])


class SnapshotDepthHS(pl.LightningModule):

    def __init__(self, hparams, log_dir=None):
        super().__init__()
        # PL 1.9+: hparams 是只读属性，不能直接赋值
        self._hparams_local = copy.deepcopy(hparams)
        self.save_hyperparameters(self._hparams_local)
        self.__build_model()
        # PL 1.9+: Metric 必须作为 Module 属性注册，用 ModuleDict
        self.metrics = torch.nn.ModuleDict({
            'mae_depthmap': MeanAbsoluteError(),
            'mse_depthmap': MeanSquaredError(),
            'mae_image': MeanAbsoluteError(),
            'mse_image': MeanSquaredError(),
        })
        self.log_dir = log_dir

    # =================================================================================
    # ## 以下是之前缺失的、从原始文件迁移过来的 PyTorch Lightning 核心方法 ##
    # =================================================================================

#     def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
#                        using_native_amp=False, using_lbfgs=False):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs):
        # 学习率预热 (warm-up)
        # 签名兼容 PL 1.5+：移除了 on_tpu/using_native_amp/using_lbfgs，统一用 **kwargs 吸收
        # 学习率预热 (warm-up)
        if self.trainer.global_step < 54:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 54.)
            optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.optics_lr
            optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.cnn_lr
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        params = [
            {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer

#     def training_step(self, samples, batch_idx):
#         target_images = samples['hs_image']
#         target_depthmaps = samples['depth_map']
# #         print(f"[原始数据检查]")
# #         print(f"  target_images:    min={target_images.min():.4f}, max={target_images.max():.4f}")
# #         print(f"  target_depthmaps: min={target_depthmaps.min():.4f}, max={target_depthmaps.max():.4f}")
# #         print(f"  target_depthmaps 是否全零: {(target_depthmaps == 0).all()}")
#         # 修复多余维度
#         if target_images.ndim == 5:
#             target_images = target_images.squeeze(1)
    
#         if target_depthmaps.ndim == 4:
#             target_depthmaps = target_depthmaps.squeeze(1)
    
#         if batch_idx == 0:
#             print(f"\n{'='*70}")
#             print(f"DEBUG training_step - 数据集直接输出:")
#             print(f"  target_images.shape:    {target_images.shape}")
#             print(f"  target_depthmaps.shape: {target_depthmaps.shape}")
#             print(f"  target_images.ndim:     {target_images.ndim}")
#             print(f"  target_depthmaps.ndim:  {target_depthmaps.ndim}")
#             print(f"{'='*70}\n")

#         # 在这个版本中，我们假设所有像素都是有效的，创建一个全1的置信度图
#         depth_conf = torch.ones_like(target_depthmaps)
#         depth_conf = crop_boundary(depth_conf, self.crop_width * 2)

#         outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))
# #         print(f"[forward 后检查]")
# #         print(f"  outputs.target_depthmaps: min={outputs.target_depthmaps.min():.4f}, max={outputs.target_depthmaps.max():.4f}")
# #         print(f"  outputs.target_depthmaps 是否全零: {(outputs.target_depthmaps == 0).all()}")
        
#         data_loss, loss_logs = self.__compute_loss(outputs, outputs.target_depthmaps, outputs.target_images, depth_conf)
#         loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

#         misc_logs = {
#             'train_misc/est_depth_max': outputs.est_depthmaps.max(),
#             'train_misc/est_depth_min': outputs.est_depthmaps.min(),
#             'train_misc/est_image_max': outputs.est_images.max(),
#             'train_misc/est_image_min': outputs.est_images.min(),
#         }
#         if self.hparams.optimize_optics:
#             misc_logs.update({
#                 'optics/heightmap_max': self.camera.heightmap1d().max(),
#                 'optics/heightmap_min': self.camera.heightmap1d().min(),
#             })

#         logs = {**loss_logs, **misc_logs}

#         if not self.global_step % self.hparams.summary_track_train_every:
#             self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'train')

#         self.log_dict(logs)
#         return data_loss

# 2026.1.22 修改
    def training_step(self, samples, batch_idx):
        # 1. 获取数据，包括新增的 mask
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']
        
        # [NEW] 获取 Dataset 返回的 mask
        # 形状通常是 [B, H, W] 或 [B, 1, H, W]，我们需要统一处理
        valid_mask = samples['mask'] 

        # 修复多余维度 (Squeeze logic)
        if target_images.ndim == 5:
            target_images = target_images.squeeze(1)
        if target_depthmaps.ndim == 4:
            target_depthmaps = target_depthmaps.squeeze(1)
        if valid_mask.ndim == 4:
            valid_mask = valid_mask.squeeze(1) # 确保 mask 是 [B, H, W]

        # 2. 合并 Mask 和 边界裁剪 (depth_conf)
        # 你原来的 depth_conf 是为了裁剪边缘效应
        boundary_mask = torch.ones_like(target_depthmaps)
        boundary_mask = crop_boundary(boundary_mask, self.crop_width * 2)
        valid_mask = crop_boundary(valid_mask, self.crop_width * 2)
        # [NEW] 最终的有效区域 = Dataset提供的物理Mask * 边界裁剪Mask
        # 这是一个二值 mask (0.0 或 1.0)
        final_mask = valid_mask * boundary_mask
        
        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))
        
        # 3. 计算 Loss (传入 final_mask)
        data_loss, loss_logs = self.__compute_loss(outputs, outputs.target_depthmaps, outputs.target_images, final_mask)
        
        # Logging ... (保持不变)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}
        misc_logs = {
            'train_misc/est_depth_max': outputs.est_depthmaps.max(),
            'train_misc/est_depth_min': outputs.est_depthmaps.min(),
            'train_misc/est_image_max': outputs.est_images.max(),
            'train_misc/est_image_min': outputs.est_images.min(),
        }
        if self.hparams.optimize_optics:
             misc_logs.update({
                'optics/heightmap_max': self.camera.heightmap1d().max(),
                'optics/heightmap_min': self.camera.heightmap1d().min(),
            })
        
        logs = {**loss_logs, **misc_logs}
        
#         if not self.global_step % self.hparams.summary_track_train_every:
#             # 可以在 log images 时把 mask 也画出来看看，或者只画 masked 后的深度图
        self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'train',final_mask)

        self.log_dict(logs)
        return data_loss


    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)

#     def validation_step(self, samples, batch_idx):
#         target_images = samples['hs_image']
#         target_depthmaps = samples['depth_map']
#         depth_conf = torch.ones_like(target_depthmaps)
#         depth_conf = crop_boundary(depth_conf, 2 * self.crop_width)

#         outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

#         est_depthmaps = outputs.est_depthmaps * depth_conf
#         target_depthmaps_val = outputs.target_depthmaps * depth_conf

#         self.metrics['mae_depthmap'](est_depthmaps, target_depthmaps_val)
#         self.metrics['mse_depthmap'](est_depthmaps, target_depthmaps_val)
#         self.metrics['mae_image'](outputs.est_images, outputs.target_images)
#         self.metrics['mse_image'](outputs.est_images, outputs.target_images)

#         self.log('validation/mse_depthmap', self.metrics['mse_depthmap'], on_step=False, on_epoch=True)
#         self.log('validation/mae_depthmap', self.metrics['mae_depthmap'], on_step=False, on_epoch=True)
#         self.log('validation/mse_image', self.metrics['mse_image'], on_step=False, on_epoch=True)
#         self.log('validation/mae_image', self.metrics['mae_image'], on_step=False, on_epoch=True)

#         if batch_idx == 0:
#             self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'validation')
    def validation_step(self, samples, batch_idx):
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']
        
        # [NEW] 获取并处理 Mask
        valid_mask = samples['mask']
        if valid_mask.ndim == 4: valid_mask = valid_mask.squeeze(1)
        
        # 边界裁剪
        boundary_mask = torch.ones_like(target_depthmaps)
        boundary_mask = crop_boundary(boundary_mask, 2 * self.crop_width)
        valid_mask = crop_boundary(valid_mask, 2 * self.crop_width)
        final_mask = valid_mask * boundary_mask

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

        # [NEW] 这里的评估必须非常小心
        # 如果直接用 metrics 库 (如 TorchMetrics)，它们通常默认会对所有像素求平均
        # 所以我们需要把非 mask 区域设为 nan 或者手动计算，但最简单的方法是只看 mask 区域
        
        # 策略：手动计算 Valid 区域的 MAE/MSE 
        # (因为直接传给 self.metrics 会被背景的0拉低 error)
        
        est = outputs.est_depthmaps
        tgt = outputs.target_depthmaps
        
        # 计算差异
        diff = torch.abs(est - tgt) * final_mask
        diff_sq = (est - tgt)**2 * final_mask
        
        # 计算平均值 (Sum / Count)
        num_valid = final_mask.sum() + 1e-6
        mae = diff.sum() / num_valid
        mse = diff_sq.sum() / num_valid
        
        # Log 手动计算的指标
        self.log('validation/mse_depthmap', mse, on_step=False, on_epoch=True)
        self.log('validation/mae_depthmap', mae, on_step=False, on_epoch=True)
        
        # Image Metrics (通常全图都算，或者也乘 mask，看你需求，通常全图算)
        self.metrics['mae_image'](outputs.est_images, outputs.target_images)
        self.metrics['mse_image'](outputs.est_images, outputs.target_images)
        self.log('validation/mse_image', self.metrics['mse_image'], on_step=False, on_epoch=True)
        self.log('validation/mae_image', self.metrics['mae_image'], on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'validation',final_mask)

#     def validation_epoch_end(self, outputs):
#         val_loss = self.__combine_loss(self.metrics['mae_depthmap'].compute(),
#                                        self.metrics['mae_image'].compute(),
#                                        0.)
#         self.log('val_loss', val_loss)
#         mse_image = self.metrics['mse_image'].compute()
#         # PSNR = 10 * log10(MAX^2 / MSE)
#         # 假设图像范围是 [0, 1]，MAX = 1.0
#         psnr_image = 10 * torch.log10(1.0 / (mse_image + 1e-10))  # 加小值避免除零
#         self.log('validation/psnr_image', psnr_image)

    def validation_epoch_end(self, outputs):
        # 1. 获取深度图 MAE
        # 关键点：不再使用 .compute()，而是从 trainer.callback_metrics 获取
        # 这是因为我们在 validation_step 里手动计算了 Masked MAE 并 log 了 'validation/mae_depthmap'
        # Lightning 会自动帮我们将 log 的值在 epoch 维度求平均
        val_mae_depth = self.trainer.callback_metrics.get('validation/mae_depthmap')
        
        # Sanity Check 保护：如果是第一次运行sanity check可能还没值，给个默认tensor
        if val_mae_depth is None:
            val_mae_depth = torch.tensor(0.0, device=self.device)

        # 2. 获取图像 MAE (图像通常是全图计算，所以之前的 metric 依然有效)
        # 如果你在 validation_step 里调用了 self.metrics['mae_image'].update(...)，这里就可以 .compute()
        val_mae_image = self.metrics['mae_image'].compute()

        # 3. 组合 Loss (传入两个标量 tensor)
        # 注意：确保 psf_loss 传 0.0，因为验证集通常不看 PSF 约束
        val_loss = self.__combine_loss(val_mae_depth, val_mae_image, 0.0)
        
        self.log('val_loss', val_loss)

        # 4. 计算 PSNR
        # 同样使用 compute() 获取 MSE
        mse_image = self.metrics['mse_image'].compute()
        
        # PSNR = 10 * log10(MAX^2 / MSE)
        # 假设图像范围是 [0, 1]，MAX = 1.0
        # 加上 1e-10 避免除以 0
        psnr_image = 10 * torch.log10(1.0 / (mse_image + 1e-10))
        
        self.log('validation/psnr_image', psnr_image)


    def __build_model(self):
        hparams = self.hparams
        self.crop_width = hparams.crop_width
        mask_diameter = hparams.focal_length / hparams.f_number
        wavelengths = np.linspace(hparams.start_wl, hparams.end_wl, hparams.hs_channels)
        print(
            f"Initializing camera with {hparams.hs_channels} channels, from {hparams.start_wl * 1e9:.1f}nm to {hparams.end_wl * 1e9:.1f}nm")
        camera_recipe = {
            'wavelengths': wavelengths, 'min_depth': hparams.min_depth, 'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth, 'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz + 4 * self.crop_width, 'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length, 'mask_diameter': mask_diameter, 'mask_size': hparams.mask_sz,
            'mask_upsample_factor': hparams.mask_upsample_factor,
            'diffraction_efficiency': hparams.diffraction_efficiency,
            'full_size': hparams.full_size,
            'use_virtual_lens_phase': getattr(hparams, 'use_virtual_lens_phase', True),
        }
        self.camera = camera.MixedCamera(**camera_recipe, requires_grad=hparams.optimize_optics)
        self.decoder = SimpleModel(hparams)
        # self.image_lossfn = CombinedLoss(l1_weight=hparams.l1_loss_weight, sam_weight=hparams.sam_loss_weight)
        self.image_lossfn = CombinedLoss(l1_weight=hparams.l1_loss_weight)
        self.depth_lossfn = torch.nn.L1Loss()
        print(self.camera)

    def forward(self, images, depthmaps, is_testing):
        if False:  # 改为 True 来启用调试
            print(f"\nDEBUG forward - 输入:")
            print(f"  images.shape:    {images.shape} (ndim={images.ndim})")
            print(f"  depthmaps.shape: {depthmaps.shape} (ndim={depthmaps.ndim})")
            
        while images.ndim > 4:
            if images.shape[1] == 1:
                images = images.squeeze(1)  # (B, 1, C, H, W) → (B, C, H, W)
            else:
                break
    
        while depthmaps.ndim > 3:
            if depthmaps.shape[1] == 1:
                depthmaps = depthmaps.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            else:
                break
    
#         print(f"DEBUG forward - 修复后:")
#         print(f"  images.shape:    {images.shape}")
#         print(f"  depthmaps.shape: {depthmaps.shape}")
#         print(f"{'='*70}\n")
        images_linear = images
#         print(f"DEBUG: 调用 camera.forward")
#         print(f"  images_linear.shape:  {images_linear.shape}")
#         print(f"  depthmaps.shape:      {depthmaps.shape}")
        captimgs, target_volumes, psf = self.camera.forward(images_linear, depthmaps,
                                                            occlusion=self.hparams.occlusion,
                                                            is_training=self.training)
        # 计算一个纯净的psf用于preinverse
        psf_pure = self.camera.psf_at_camera(is_training=torch.tensor(False)).unsqueeze(0)
        noise_sigma = (self.hparams.noise_sigma_max - self.hparams.noise_sigma_min) * torch.rand(
            (captimgs.shape[0], 1, 1, 1), device=images.device) + self.hparams.noise_sigma_min
        captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=images.device, dtype=images.dtype)
#         print(f"\n[STEP 4] 裁剪（crop_width={self.crop_width}）")
        # 🔍 调试点：crop_boundary 前
#         print(f"DEBUG: crop_boundary 前:")
#         print(f"  captimgs.shape:       {captimgs.shape}")
#         print(f"  target_volumes.shape: {target_volumes.shape}")
#         print(f"  crop_width:           {self.crop_width}")
        captimgs = crop_boundary(captimgs, self.crop_width)
        target_volumes = crop_boundary(target_volumes, self.crop_width)
        if self.hparams.preinverse:
            # 这里也改了psf_pure
            psf_cropped = crop_psf(psf_pure, captimgs.shape[-2:])
            pinv_volumes = apply_tikhonov_inverse(captimgs, psf_cropped, self.hparams.reg_tikhonov,
                                                  apply_edgetaper=True)
        else:
            pinv_volumes = torch.zeros_like(target_volumes)
        model_outputs = self.decoder(captimgs=captimgs, pinv_volumes=pinv_volumes, images=images_linear,
                                     depthmaps=depthmaps)
        # 🔍 调试点：final crop_boundary
#         print(f"DEBUG: final crop_boundary:")
#         print(f"  images.shape:            {images.shape}")
#         print(f"  depthmaps.shape:         {depthmaps.shape}")
#         print(f"  model_outputs.est_images.shape: {model_outputs.est_images.shape}")
#         print(f"  model_outputs.est_depthmaps.shape: {model_outputs.est_depthmaps.shape}")
        target_images = crop_boundary(images, 2 * self.crop_width)
        target_depthmaps = crop_boundary(depthmaps, 2 * self.crop_width)
        est_images = crop_boundary(model_outputs.est_images, self.crop_width)
        est_depthmaps = crop_boundary(model_outputs.est_depthmaps, self.crop_width)
        
        return SnapshotOutputs(
            target_images=target_images, target_depthmaps=target_depthmaps,
            captimgs=captimgs, captimgs_linear=captimgs, est_images=est_images,
            est_depthmaps=est_depthmaps, psf=psf,
        )

    def __combine_loss(self, depth_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
            self.hparams.image_loss_weight * image_loss + \
            self.hparams.psf_loss_weight * psf_loss
    
#     def __compute_loss(self, outputs, target_depthmaps, target_images, depth_conf):
#         est_images = outputs.est_images
#         est_depthmaps = outputs.est_depthmaps

#         # --- 1. 计算各损失分量 ---
#         depth_loss = self.depth_lossfn(est_depthmaps * depth_conf, target_depthmaps * depth_conf)
        
#         image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images) 
        
#         # --- 修复：在这里为 'psf_out_of_fov_max' 提供一个默认值 ---
#         psf_loss = torch.tensor(0.0, device=depth_loss.device) 
#         psf_out_of_fov_max = torch.tensor(0.0, device=depth_loss.device) # <--- 添加这一行
#         # --- 修复结束 ---
        
#         if self.hparams.psf_loss_weight > 0:
#             psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
#             psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels
# #              psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels

#         # --- 2. 计算加权后的损失 ---
#         weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
#         weighted_image_loss = self.hparams.image_loss_weight * image_loss
#         weighted_psf_loss = self.hparams.psf_loss_weight * psf_loss

#         total_loss = weighted_depth_loss + weighted_image_loss + weighted_psf_loss

#         # --- 3. 添加详细的调试信息 (关键！) ---
#         if self.training and self.global_step % 100 == 0:  # 每 5 步打印一次
#             print(f"\n==================== 损失分量分析 (Step {self.global_step}) ====================")
#             print(f"--- 1. 原始损失分量 (Unweighted) ---")
#             print(f"  Depth Loss (L1):       {depth_loss.item():.6f}")
#             print(f"  Image L1 Loss (Raw):   {image_l1.item():.6f}")
#             print(f"  Image SAM Loss (Raw):  {image_sam.item():.6f}")
#             print(f"  Image Loss (Combined): {image_loss.item():.6f}") # (L1*w_l1 + SAM*w_sam)
#             print(f"  PSF Loss (Normalized): {psf_loss.item():.6f}")
            
#             print(f"\n--- 2. 权重设置 (Weights) ---")
#             print(f"  Depth Weight: {self.hparams.depth_loss_weight}")
#             print(f"  Image Weight: {self.hparams.image_loss_weight}") # (e.g., 0.1)
#             print(f"  PSF Weight:   {self.hparams.psf_loss_weight}")   # (e.g., 0)

#             print(f"\n--- 3. 加权后损失分量 (Weighted) ---")
#             print(f"  Weighted Depth: {weighted_depth_loss.item():.6f}")
#             print(f"  Weighted Image: {weighted_image_loss.item():.6f}")
#             print(f"  Weighted PSF:   {weighted_psf_loss.item():.6f}")
            
#             print(f"\n--- 4. 最终总损失 ---")
#             print(f"  TOTAL LOSS:     {total_loss.item():.6f}")
#             image_contribution = weighted_image_loss / total_loss
#             depth_contribution = weighted_depth_loss / total_loss
#             psf_contribution = weighted_psf_loss / total_loss
#             print(f"Loss比例: image={image_contribution:.2%}, depth={depth_contribution:.2%}, psf={psf_contribution:.2%}")

            
#             print(f"\n--- 5. 数据范围检查 (关键！) ---")
#             print(f"  Target Depth:  min={target_depthmaps.min().item():.3f}, max={target_depthmaps.max().item():.3f}, mean={target_depthmaps.mean().item():.3f}")
#             print(f"  Est Depth:     min={est_depthmaps.min().item():.3f}, max={est_depthmaps.max().item():.3f}, mean={est_depthmaps.mean().item():.3f}")
#             print(f"  Target Images: min={target_images.min().item():.3f}, max={target_images.max().item():.3f}, mean={target_images.mean().item():.3f}")
#             print(f"  Est Images:    min={est_images.min().item():.3f}, max={est_images.max().item():.3f}, mean={est_images.mean().item():.3f}")
#             print("========================================================================\n")

#         return total_loss, {
#             'total_loss': total_loss, 'depth_loss': depth_loss, 'image_loss_total': image_loss,
#             'image_loss_l1': image_l1, 'psf_loss': psf_loss,
#             'psf_out_of_fov_max': psf_out_of_fov_max, # 这一行现在安全了
#         }
    def __compute_loss(self, outputs, target_depthmaps, target_images, final_mask):
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps

        # --- 1. 计算 Masked Depth Loss (关键修改) ---
        # 不要直接调用 self.depth_lossfn，因为它内部可能是 mean reduction
        # 我们手动写，或者确认你的 lossfn 配置为 reduction='none'
        
        # 假设 self.depth_lossfn 是 L1Loss 或 MSELoss
        # 推荐：手动计算以确保万无一失
        
        # 绝对误差图
        diff = torch.abs(est_depthmaps - target_depthmaps)
        
        # 只保留 mask 区域的误差
        masked_diff = diff * final_mask
        
        # 归一化：除以有效像素数量，而不是总像素数量
        # +1e-6 防止除以零
        depth_loss = masked_diff.sum() / (final_mask.sum() + 1e-6)
        
        if self.training and self.global_step % 100 == 0:
            with torch.no_grad():
                # 只看 mask 区域的统计
                valid_gt = target_depthmaps[final_mask > 0.5]
                valid_est = est_depthmaps[final_mask > 0.5]
                
                if valid_gt.numel() > 0:
                    print(f"\n[Step {self.global_step}] IPS 深度预测诊断:")
                    print(f"  GT 深度 (IPS 归一化): min={valid_gt.min():.4f}, max={valid_gt.max():.4f}, "
                          f"mean={valid_gt.mean():.4f}, std={valid_gt.std():.4f}")
                    print(f"  预测深度 (IPS 归一化): min={valid_est.min():.4f}, max={valid_est.max():.4f}, "
                          f"mean={valid_est.mean():.4f}, std={valid_est.std():.4f}")
                    
                    # 物理深度反算（仅用于可视化）
                    # 反演公式：d = (d_min * d_max) / (d_max - (d_max - d_min) * d_norm)
                    def ips_to_physical(ips_norm, min_d=0.4, max_d=2.0):
                        # 避免除以零
                        safe_norm = torch.clamp(ips_norm, 1e-6, 1.0 - 1e-6)
                        return (max_d * min_d) / (max_d - (max_d - min_d) * safe_norm)
                    
                    gt_phys_mean = ips_to_physical(valid_gt.mean()).item()
                    est_phys_mean = ips_to_physical(valid_est.mean()).item()
                    
                    print(f"  GT 物理深度: 平均 ≈ {gt_phys_mean:.3f}m")
                    print(f"  预测物理深度: 平均 ≈ {est_phys_mean:.3f}m")
                    
                    # 检查预测的动态范围
                    gt_range = valid_gt.max() - valid_gt.min()
                    est_range = valid_est.max() - valid_est.min()
                    print(f"  动态范围: GT={gt_range:.4f} (IPS), EST={est_range:.4f} (IPS), "
                          f"比值={est_range/(gt_range+1e-6):.2%}")
                    
                    if est_range < 0.1 and gt_range > 0.3:
                        print(f"  ⚠️ 警告：预测深度动态范围过小！网络可能陷入常数输出。")
                        print(f"      => 检查是否：")
                        print(f"         1. 开启了 optimize_optics=True（DOE 优化）")
                        print(f"         2. Loss 权重设置是否合理（depth_loss_weight）")
                        print(f"         3. 数据中物体间深度差异是否足够大")
        
        
                    print(f"  预测深度 (mask内): min={valid_est.min():.4f}, max={valid_est.max():.4f}, "
                          f"mean={valid_est.mean():.4f}, std={valid_est.std():.4f}")
                    
                    # 检查预测的动态范围
                    gt_range = valid_gt.max() - valid_gt.min()
                    est_range = valid_est.max() - valid_est.min()
                    print(f"  动态范围: GT={gt_range:.4f}, EST={est_range:.4f}, "
                          f"比值={est_range/(gt_range+1e-6):.2%}")
                    
                    if est_range < 0.1 and gt_range > 0.3:
                        print(f"  ⚠️ 警告：预测深度动态范围过小！网络可能陷入常数输出。")
        # --- Image Loss (通常图像重建是对全图的，包括背景) ---
        # 如果你也想让图像重建忽略背景，也可以乘 mask，但通常不需要
        image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images) 
        
        # --- PSF Loss ---
        psf_loss = torch.tensor(0.0, device=depth_loss.device) 
        psf_out_of_fov_max = torch.tensor(0.0, device=depth_loss.device)
        
        if self.hparams.psf_loss_weight > 0:
            psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
            psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels

        # --- 2. 加权 ---
        weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
        weighted_image_loss = self.hparams.image_loss_weight * image_loss
        weighted_psf_loss = self.hparams.psf_loss_weight * psf_loss

        total_loss = weighted_depth_loss + weighted_image_loss + weighted_psf_loss
        
        return total_loss, {
            'total_loss': total_loss, 
            'depth_loss': depth_loss, # 记录的是纯净的 mask 内 loss
            'image_loss_total': image_loss,
            'image_loss_l1': image_l1, 
            'psf_loss': psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max,
        }
    
    @torch.no_grad()
    def __log_images(self, outputs, target_images, target_depthmaps, tag: str, final_mask):
        # 在你的 __log_images 函数中添加
        diff_map = torch.abs(outputs.est_depthmaps - outputs.target_depthmaps) * final_mask
        # 归一化以便显示
        diff_vis = diff_map / (diff_map.max() + 1e-6) 
        # =========== 【修复代码】 ===========
        # 3. 增加通道维度 (B, H, W) -> (B, 1, H, W)
        # TensorBoard 需要 4D 张量
        if diff_vis.ndim == 3:
            diff_vis = diff_vis.unsqueeze(1)
        # ==================================

        # 4. 现在 diff_vis 的形状是 (2, 1, 384, 384)，符合 NCHW
        self.logger.experiment.add_images(f'{tag}/diff_error', diff_vis, self.global_step)
        captimgs, est_images, est_depthmaps = outputs.captimgs, outputs.est_images, outputs.est_depthmaps
        summary_image_sz = self.hparams.summary_image_sz
        summary_max_images = min(self.hparams.summary_max_images, captimgs.shape[0])
        n_channels = target_images.shape[1]
        vis_channels = [n_channels // 4, n_channels // 2, 3 * n_channels // 4]
        captimgs_vis, target_images_vis, est_images_vis = captimgs[:, vis_channels, ...], target_images[:, vis_channels,
                                                                                          ...], est_images[:,
                                                                                                vis_channels, ...]
        target_depthmaps_4d = target_depthmaps.unsqueeze(1)  # (B, 1, H, W)
        est_depthmaps_4d = est_depthmaps.unsqueeze(1)        # (B, 1, H, W)
        captimgs_resized, target_images_resized, target_depthmaps_resized, est_images_resized, est_depthmaps_resized = [
        imresize(x, (summary_image_sz, summary_image_sz)) for x in
        [captimgs_vis, target_images_vis, target_depthmaps_4d, est_images_vis, est_depthmaps_4d]
        ]
        
        target_depthmaps = target_depthmaps_resized.squeeze(1)  # (B, 1, H, W) → (B, H, W)
        est_depthmaps = est_depthmaps_resized.squeeze(1)        # (B, 1, H, W) → (B, H, W)
        captimgs, target_images, est_images = \
            captimgs_resized, target_images_resized, est_images_resized
#        # ✅ 为深度图添加通道维度用于拼接
#         target_depthmaps, est_depthmaps = \
#             gray_to_rgb(1.0 - target_depthmaps), \
#             gray_to_rgb(1.0 - est_depthmaps)
        # --- 修改开始 ---
        
        # 1. 此时 target_depthmaps 是 (B, H, W)，先反转颜色 (1.0 - x)
        # 注意：不需要 gray_to_rgb，我们手动变 RGB
        td_inv = 1.0 - target_depthmaps
        ed_inv = 1.0 - est_depthmaps

        # 2. 强制转为 3 通道 (B, 3, H, W)
        # 无论之前是 3D 还是 4D，都统一处理
        if td_inv.dim() == 3:
             td_rgb = td_inv.unsqueeze(1).repeat(1, 3, 1, 1)
        elif td_inv.dim() == 4 and td_inv.shape[1] == 1:
             td_rgb = td_inv.repeat(1, 3, 1, 1)
        else:
             td_rgb = td_inv # 假设已经是3通道
             
        if ed_inv.dim() == 3:
             ed_rgb = ed_inv.unsqueeze(1).repeat(1, 3, 1, 1)
        elif ed_inv.dim() == 4 and ed_inv.shape[1] == 1:
             ed_rgb = ed_inv.repeat(1, 3, 1, 1)
        else:
             ed_rgb = ed_inv
#         summary = torch.cat([captimgs, target_images, est_images, target_depthmaps, est_depthmaps], dim=-2)[
#                   :summary_max_images]
        # 3. 拼接 (使用刚才定义的变量)
        summary = torch.cat([captimgs, target_images, est_images, td_rgb, ed_rgb], dim=-2)[
                  :summary_max_images]
        grid_summary = torchvision.utils.make_grid(summary, nrow=summary_max_images)
        self.logger.experiment.add_image(f'{tag}/summary', grid_summary, self.global_step)
        if self.hparams.optimize_optics or self.global_step == 0:
            psf = self.camera.psf_at_camera(size=(128, 128), is_training=torch.tensor(False))
            psf = self.camera.normalize_psf(psf)
            psf = fftshift(crop_psf(psf, 64), dims=(-1, -2))
            psf_vis = psf / psf.view(psf.shape[0], psf.shape[1], -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            heightmap = imresize(self.camera.heightmap()[None, None, ...],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
            grid_psf = torchvision.utils.make_grid(
                psf_vis[vis_channels, ::self.hparams.summary_depth_every].transpose(0, 1), nrow=len(vis_channels),
                pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_normalized_per_depth', grid_psf, self.global_step)
            self.logger.experiment.add_image('optics/heightmap', heightmap, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--summary_max_images', type=int, default=4)
        parser.add_argument('--summary_image_sz', type=int, default=256)
        parser.add_argument('--summary_mask_sz', type=int, default=256)
        parser.add_argument('--summary_depth_every', type=int, default=1)
        parser.add_argument('--summary_track_train_every', type=int, default=4000)
        parser.add_argument('--cnn_lr', type=float, default=1e-3)
        parser.add_argument('--optics_lr', type=float, default=1e-9)
        parser.add_argument('--batch_sz', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--randcrop', default=False, action='store_true')
        parser.add_argument('--augment', default=False, action='store_true')
        parser.add_argument('--depth_loss_weight', type=float, default=1.0)
        parser.add_argument('--image_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_size', type=int, default=64)
        parser.add_argument('--l1_loss_weight', type=float, default=1.0)
        parser.add_argument('--sam_loss_weight', type=float, default=0.5)
        parser.add_argument('--image_sz', type=int, default=512)
        parser.add_argument('--n_depths', type=int, default=8)
        parser.add_argument('--min_depth', type=float, default=0.4)
        parser.add_argument('--max_depth', type=float, default=2.0)
        parser.add_argument('--crop_width', type=int, default=32)
        parser.add_argument('--reg_tikhonov', type=float, default=1.0)
        parser.add_argument('--model_base_ch', type=int, default=32)
        parser.add_argument('--preinverse', dest='preinverse', action='store_true')
        parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
        parser.set_defaults(preinverse=True)
        parser.add_argument('--camera_type', type=str, default='mixed')
        parser.add_argument('--mask_sz', type=int, default=8000)
        parser.add_argument('--focal_length', type=float, default=50e-3)
        parser.add_argument('--focal_depth', type=float, default=0.67)
        parser.add_argument('--use_virtual_lens_phase', dest='use_virtual_lens_phase', action='store_true',
                    help='在 pupil 处叠加基类“理想薄透镜”相位（传统成像/对焦建模）。')
        parser.add_argument('--no-use_virtual_lens_phase', dest='use_virtual_lens_phase', action='store_false',
                    help='关闭基类“理想薄透镜”相位（Baek-like：DOE 充当唯一主透镜时推荐）。')
        parser.set_defaults(use_virtual_lens_phase=True)
        parser.add_argument('--f_number', type=float, default=6.3)
        parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--full_size', type=int, default=1920)
        parser.add_argument('--mask_upsample_factor', type=int, default=10)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)
        parser.add_argument('--occlusion', dest='occlusion', action='store_true')
        parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
        parser.set_defaults(occlusion=True)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=False)
        parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
        parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
        parser.set_defaults(psf_jitter=True)
        parser.add_argument('--hs_channels', type=int, default=29, help='高光谱数据的通道数')
        parser.add_argument('--start_wl', type=float, default=420e-9, help='起始波长（米, 例如 420nm）')
        parser.add_argument('--end_wl', type=float, default=700e-9, help='结束波长（米, 例如 700nm）')
        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=False)
        return parser
