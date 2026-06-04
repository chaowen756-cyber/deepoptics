# snapshotdepth_trainer_hs.py

import os
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# 确保这些我们创建的文件与此脚本在同一项目路径下
from datasets.hyperspectral_dataset import HyperspectralDepthDataset
from snapshotdepth_hs import SnapshotDepthHS as SnapshotDepth
from util.log_manager import LogManager

seed_everything(123)


def _patch_pl_ddp_sync_params_if_missing():
    """兼容某些 torch 版本移除了 DDP 私有方法 `_sync_params` 的情况。"""
    try:
        from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
        if not hasattr(LightningDistributedDataParallel, '_sync_params'):
            LightningDistributedDataParallel._sync_params = lambda self: None
            print('[Compat] Patched LightningDistributedDataParallel._sync_params')
    except Exception:
        pass


def _strip_deprecated_trainer_args(args):
    """移除/转换 PL 1.5+ 废弃的 Trainer 参数。
    
    1. checkpoint_callback / resume_from_checkpoint: 已移除
    2. accelerator='ddp' -> strategy='ddp' + accelerator='auto'
    3. gpus -> devices (PL 1.7+)
    """
    # accelerator='ddp' 转换
    if hasattr(args, 'accelerator') and getattr(args, 'accelerator', None) == 'ddp':
        args.strategy = 'ddp'
        args.accelerator = 'auto'
        print('[Compat] 已将 accelerator=ddp 转换为 strategy=ddp, accelerator=auto')
    
    # gpus -> devices (如果 PL 版本需要)
    if hasattr(args, 'gpus') and getattr(args, 'gpus', None) is not None:
        if not hasattr(args, 'devices') or getattr(args, 'devices', None) is None:
            args.devices = args.gpus
            print(f'[Compat] 已将 gpus={args.gpus} 复制到 devices')
    
    # 删除废弃参数
    for key in ("checkpoint_callback", "resume_from_checkpoint"):
        if hasattr(args, key):
            try:
                delattr(args, key)
            except Exception:
                setattr(args, key, None)


def prepare_data(hparams):
    """
    为高光谱数据准备训练和验证的 DataLoader。
    """
    all_scene_folders = [f'deploy {i}' for i in range(1, 19)]
    train_folders = all_scene_folders[:15]
    val_folders = all_scene_folders[15:]

    print(f"训练集文件夹数量: {len(train_folders)}")
    print(f"验证集文件夹数量: {len(val_folders)}")

    train_dataset = HyperspectralDepthDataset(
        base_dir=hparams.data_root,
        scene_folders=train_folders,
        image_size=(hparams.image_sz, hparams.image_sz),
        hs_channels=hparams.hs_channels,
        is_training=True,
        randcrop=True,
        augment=hparams.augment,
        min_depth=hparams.min_depth,   # <--- 新增
        max_depth=hparams.max_depth    # <--- 新增
    )

    val_dataset = HyperspectralDepthDataset(
        base_dir=hparams.data_root,
        scene_folders=val_folders,
        image_size=(hparams.image_sz, hparams.image_sz),
        hs_channels=hparams.hs_channels,
        is_training=False,
        randcrop=False,
        augment=False,
        min_depth=hparams.min_depth,   # <--- 新增
        max_depth=hparams.max_depth    # <--- 新增
    )

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                  num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def main(args):
    _patch_pl_ddp_sync_params_if_missing()
    _strip_deprecated_trainer_args(args)
    
    logger = TensorBoardLogger(args.default_root_dir, name=args.experiment_name)
    logmanager_callback = LogManager()
    
    # ModelCheckpoint 兼容新旧 PL 版本
    try:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(logger.log_dir, 'checkpoints'),
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
    except TypeError:
        # 老版本 PL 用 filepath
        checkpoint_callback = ModelCheckpoint(
            verbose=True,
            monitor='val_loss',
            filepath=os.path.join(logger.log_dir, 'checkpoints', '{epoch}-{val_loss:.4f}'),
            save_top_k=1,
            mode='min',
        )
    
    model = SnapshotDepth(hparams=args, log_dir=logger.log_dir)
    train_dataloader, val_dataloader = prepare_data(hparams=args)
    
    # 新版本 PL: checkpoint 放入 callbacks 列表，不用 checkpoint_callback= 参数
    callbacks = [logmanager_callback, checkpoint_callback]
    
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
        sync_batchnorm=True,
        benchmark=True,
    )
    # PL 1.5+: fit() 参数名改为 train_dataloaders (复数) 和 val_dataloaders
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)




if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='Hyperspectral_LearnedDepth')

    # --- 核心修改点：动态计算默认的数据集路径 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "Baek数据集")
    parser.add_argument('--data_root', type=str,
                        default=default_data_path,
                        help="包含 'deploy X' 文件夹的数据集根目录")

    # 从 Trainer, Model 添加所有需要的参数
    parser = Trainer.add_argparse_args(parser)
    parser = SnapshotDepth.add_model_specific_args(parser)

    # 设置一些默认值
    parser.set_defaults(
        # gpus=1, # 在M1上运行时，最好从命令行指定
        default_root_dir='autodl-tmp/data',
        max_epochs=100,
    )

    args = parser.parse_args()

    print("-" * 50)
    print(f"数据根目录将使用: {args.data_root}")
    print("-" * 50)

    main(args)