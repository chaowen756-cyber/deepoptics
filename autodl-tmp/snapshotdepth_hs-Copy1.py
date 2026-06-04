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
        self.hparams = copy.deepcopy(hparams)
        self.save_hyperparameters(self.hparams)
        self.__build_model()
        self.metrics = {
            'mae_depthmap': MeanAbsoluteError(),
            'mse_depthmap': MeanSquaredError(),
            'mae_image': MeanAbsoluteError(),
            'mse_image': MeanSquaredError(),
        }
        self.log_dir = log_dir

    # =================================================================================
    # ## 以下是之前缺失的、从原始文件迁移过来的 PyTorch Lightning 核心方法 ##
    # =================================================================================

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # 学习率预热 (warm-up)
        if self.trainer.global_step < 180:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 180.)
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

    def training_step(self, samples, batch_idx):
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']

        # 在这个版本中，我们假设所有像素都是有效的，创建一个全1的置信度图
        depth_conf = torch.ones_like(target_depthmaps)
        depth_conf = crop_boundary(depth_conf, self.crop_width * 2)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

        data_loss, loss_logs = self.__compute_loss(outputs, outputs.target_depthmaps, outputs.target_images, depth_conf)
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

        if not self.global_step % self.hparams.summary_track_train_every:
            self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'train')

        self.log_dict(logs)
        return data_loss

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_step(self, samples, batch_idx):
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']
        depth_conf = torch.ones_like(target_depthmaps)
        depth_conf = crop_boundary(depth_conf, 2 * self.crop_width)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

        est_depthmaps = outputs.est_depthmaps * depth_conf
        target_depthmaps_val = outputs.target_depthmaps * depth_conf

        self.metrics['mae_depthmap'](est_depthmaps, target_depthmaps_val)
        self.metrics['mse_depthmap'](est_depthmaps, target_depthmaps_val)
        self.metrics['mae_image'](outputs.est_images, outputs.target_images)
        self.metrics['mse_image'](outputs.est_images, outputs.target_images)

        self.log('validation/mse_depthmap', self.metrics['mse_depthmap'], on_step=False, on_epoch=True)
        self.log('validation/mae_depthmap', self.metrics['mae_depthmap'], on_step=False, on_epoch=True)
        self.log('validation/mse_image', self.metrics['mse_image'], on_step=False, on_epoch=True)
        self.log('validation/mae_image', self.metrics['mae_image'], on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'validation')

    def validation_epoch_end(self, outputs):
        val_loss = self.__combine_loss(self.metrics['mae_depthmap'].compute(),
                                       self.metrics['mae_image'].compute(),
                                       0.)
        self.log('val_loss', val_loss)

    # =================================================================================
    # ## 以下是我们之前修改过的函数，保持不变 ##
    # =================================================================================

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
        }
        self.camera = camera.MixedCamera(**camera_recipe, requires_grad=hparams.optimize_optics)
        self.decoder = SimpleModel(hparams)
        # self.image_lossfn = CombinedLoss(l1_weight=hparams.l1_loss_weight, sam_weight=hparams.sam_loss_weight)
        self.image_lossfn = CombinedLoss(l1_weight=hparams.l1_loss_weight)
        self.depth_lossfn = torch.nn.L1Loss()
        print(self.camera)

    def forward(self, images, depthmaps, is_testing):
        images_linear = images
        captimgs, target_volumes, psf = self.camera.forward(images_linear, depthmaps,
                                                            occlusion=self.hparams.occlusion,
                                                            is_training=self.training)
        noise_sigma = (self.hparams.noise_sigma_max - self.hparams.noise_sigma_min) * torch.rand(
            (captimgs.shape[0], 1, 1, 1), device=images.device) + self.hparams.noise_sigma_min
        captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=images.device, dtype=images.dtype)
        captimgs = crop_boundary(captimgs, self.crop_width)
        target_volumes = crop_boundary(target_volumes, self.crop_width)
        if self.hparams.preinverse:
            psf_cropped = crop_psf(psf, captimgs.shape[-2:])
            pinv_volumes = apply_tikhonov_inverse(captimgs, psf_cropped, self.hparams.reg_tikhonov,
                                                  apply_edgetaper=True)
        else:
            pinv_volumes = torch.zeros_like(target_volumes)
        model_outputs = self.decoder(captimgs=captimgs, pinv_volumes=pinv_volumes, images=images_linear,
                                     depthmaps=depthmaps)
        target_images = crop_boundary(images, 2 * self.crop_width)
        target_depthmaps = crop_boundary(depthmaps, 2 * self.crop_width)
        captimgs = crop_boundary(captimgs, self.crop_width)
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
    
    def __compute_loss(self, outputs, target_depthmaps, target_images, depth_conf):
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps

        # --- 1. 计算各损失分量 ---
        depth_loss = self.depth_lossfn(est_depthmaps * depth_conf, target_depthmaps * depth_conf)
        
        image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images) 
        
        # --- 修复：在这里为 'psf_out_of_fov_max' 提供一个默认值 ---
        psf_loss = torch.tensor(0.0, device=depth_loss.device) 
        psf_out_of_fov_max = torch.tensor(0.0, device=depth_loss.device) # <--- 添加这一行
        # --- 修复结束 ---
        
        if self.hparams.psf_loss_weight > 0:
             psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
             psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels

        # --- 2. 计算加权后的损失 ---
        weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
        weighted_image_loss = self.hparams.image_loss_weight * image_loss
        weighted_psf_loss = self.hparams.psf_loss_weight * psf_loss

        total_loss = weighted_depth_loss + weighted_image_loss + weighted_psf_loss

        # --- 3. 添加详细的调试信息 (关键！) ---
        if self.training and self.global_step % 5 == 0:  # 每 5 步打印一次
            print(f"\n==================== 损失分量分析 (Step {self.global_step}) ====================")
            print(f"--- 1. 原始损失分量 (Unweighted) ---")
            print(f"  Depth Loss (L1):       {depth_loss.item():.6f}")
            print(f"  Image L1 Loss (Raw):   {image_l1.item():.6f}")
            print(f"  Image SAM Loss (Raw):  {image_sam.item():.6f}")
            print(f"  Image Loss (Combined): {image_loss.item():.6f}") # (L1*w_l1 + SAM*w_sam)
            print(f"  PSF Loss (Normalized): {psf_loss.item():.6f}")
            
            print(f"\n--- 2. 权重设置 (Weights) ---")
            print(f"  Depth Weight: {self.hparams.depth_loss_weight}")
            print(f"  Image Weight: {self.hparams.image_loss_weight}") # (e.g., 0.1)
            print(f"  PSF Weight:   {self.hparams.psf_loss_weight}")   # (e.g., 0)

            print(f"\n--- 3. 加权后损失分量 (Weighted) ---")
            print(f"  Weighted Depth: {weighted_depth_loss.item():.6f}")
            print(f"  Weighted Image: {weighted_image_loss.item():.6f}")
            print(f"  Weighted PSF:   {weighted_psf_loss.item():.6f}")
            
            print(f"\n--- 4. 最终总损失 ---")
            print(f"  TOTAL LOSS:     {total_loss.item():.6f}")
            
            print(f"\n--- 5. 数据范围检查 (关键！) ---")
            print(f"  Target Depth:  min={target_depthmaps.min().item():.3f}, max={target_depthmaps.max().item():.3f}, mean={target_depthmaps.mean().item():.3f}")
            print(f"  Est Depth:     min={est_depthmaps.min().item():.3f}, max={est_depthmaps.max().item():.3f}, mean={est_depthmaps.mean().item():.3f}")
            print(f"  Target Images: min={target_images.min().item():.3f}, max={target_images.max().item():.3f}, mean={target_images.mean().item():.3f}")
            print(f"  Est Images:    min={est_images.min().item():.3f}, max={est_images.max().item():.3f}, mean={est_images.mean().item():.3f}")
            print("========================================================================\n")

        return total_loss, {
            'total_loss': total_loss, 'depth_loss': depth_loss, 'image_loss_total': image_loss,
            'image_loss_l1': image_l1, 'psf_loss': psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max, # 这一行现在安全了
        }

#     def __compute_loss(self, outputs, target_depthmaps, target_images, depth_conf):
#         est_images = outputs.est_images
#         est_depthmaps = outputs.est_depthmaps
#         depth_loss = self.depth_lossfn(est_depthmaps * depth_conf, target_depthmaps * depth_conf)
#         image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images)
#         psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
#         # 按波长数归一化PSF损失
#         psf_loss = psf_out_of_fov_sum / 29
#         total_loss = self.__combine_loss(depth_loss, image_loss, psf_loss)
#         return total_loss, {
#             'total_loss': total_loss, 'depth_loss': depth_loss, 'image_loss_total': image_loss,
#             'image_loss_l1': image_l1, 'image_loss_sam': image_sam, 'psf_loss': psf_loss,
#             'psf_out_of_fov_max': psf_out_of_fov_max,
#         }


    @torch.no_grad()
    def __log_images(self, outputs, target_images, target_depthmaps, tag: str):
        captimgs, est_images, est_depthmaps = outputs.captimgs, outputs.est_images, outputs.est_depthmaps
        summary_image_sz = self.hparams.summary_image_sz
        summary_max_images = min(self.hparams.summary_max_images, captimgs.shape[0])
        n_channels = target_images.shape[1]
        vis_channels = [n_channels // 4, n_channels // 2, 3 * n_channels // 4]
        captimgs_vis, target_images_vis, est_images_vis = captimgs[:, vis_channels, ...], target_images[:, vis_channels,
                                                                                          ...], est_images[:,
                                                                                                vis_channels, ...]
        captimgs, target_images, target_depthmaps, est_images, est_depthmaps = [
            imresize(x, (summary_image_sz, summary_image_sz)) for x in
            [captimgs_vis, target_images_vis, target_depthmaps, est_images_vis, est_depthmaps]
        ]
        target_depthmaps, est_depthmaps = gray_to_rgb(1.0 - target_depthmaps), gray_to_rgb(1.0 - est_depthmaps)
        summary = torch.cat([captimgs, target_images, est_images, target_depthmaps, est_depthmaps], dim=-2)[
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
        parser.add_argument('--cnn_lr', type=float, default=1e-4)
        parser.add_argument('--optics_lr', type=float, default=1e-5)
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
        parser.add_argument('--image_sz', type=int, default=256)
        parser.add_argument('--n_depths', type=int, default=16)
        parser.add_argument('--min_depth', type=float, default=1.0)
        parser.add_argument('--max_depth', type=float, default=5.0)
        parser.add_argument('--crop_width', type=int, default=32)
        parser.add_argument('--reg_tikhonov', type=float, default=1.0)
        parser.add_argument('--model_base_ch', type=int, default=32)
        parser.add_argument('--preinverse', dest='preinverse', action='store_true')
        parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
        parser.set_defaults(preinverse=True)
        parser.add_argument('--camera_type', type=str, default='mixed')
        parser.add_argument('--mask_sz', type=int, default=8000)
        parser.add_argument('--focal_length', type=float, default=50e-3)
        parser.add_argument('--focal_depth', type=float, default=1.7)
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
        parser.add_argument('--end_wl', type=float, default=680e-9, help='结束波长（米, 例如 680nm）')
        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=False)
        return parser
