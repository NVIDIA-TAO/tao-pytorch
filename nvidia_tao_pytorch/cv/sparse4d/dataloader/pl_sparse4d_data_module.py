# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4D DataModule for TAO PyTorch."""

import os
import random
from typing import Optional, Dict
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import Omniverse3DDetTrackDataset
from nvidia_tao_pytorch.cv.sparse4d.dataloader.sampler import GroupInBatchSampler
from nvidia_tao_pytorch.cv.sparse4d.dataloader.transforms import (
    LoadMultiViewImageFromFiles,
    LoadDepthMap,
    InstanceNameFilter,
    AICitySparse4DAdaptor,
    Compose
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.augment import (
    ResizeCropFlipImage,
    ResizeCropFlipMultiScaleDepthMap,
    BBoxRotation,
    PhotoMetricDistortionMultiViewImage,
    NormalizeMultiviewImage
)


def collate_fn(batch):
    """Custom collate function for Sparse4D dataset."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return {}

    data = {}
    for key in batch[0].keys():
        data[key] = [item[key] for item in batch]

    # Stack or concatenate data if needed
    if "img" in data:
        if isinstance(data["img"][0], torch.Tensor):
            data["img"] = torch.stack(data["img"], dim=0)

    if "projection_mat" in data:
        if isinstance(data["projection_mat"][0], (torch.Tensor, np.ndarray)):
            data["projection_mat"] = torch.stack([torch.as_tensor(x) for x in data["projection_mat"]], dim=0)

    if "image_wh" in data:
        if isinstance(data["image_wh"][0], (torch.Tensor, np.ndarray)):
            data["image_wh"] = torch.stack([torch.as_tensor(x) for x in data["image_wh"]], dim=0)

    if "focal" in data:
        if isinstance(data["focal"][0], (torch.Tensor, np.ndarray)):
            data["focal"] = torch.cat([torch.as_tensor(x.flatten()) for x in data["focal"]], dim=0)

    if "instance_inds" in data:
        data["instance_id"] = [torch.as_tensor(x) for x in data["instance_inds"]]

    if "asset_inds" in data:
        data["asset_id"] = [torch.as_tensor(x) for x in data["asset_inds"]]

    if "gt_visibility" in data:
        data["gt_visibility"] = [torch.as_tensor(x) for x in data["gt_visibility"]]

    if "gt_depth" in data:
        num_scales = len(data["gt_depth"][0])
        data["gt_depth"] = [
            torch.stack([torch.as_tensor(batch_item[scale_idx]) for batch_item in data["gt_depth"]])
            for scale_idx in range(num_scales)
        ]

    if "timestamp" in data:
        data["timestamp"] = torch.stack([torch.as_tensor(x) for x in data["timestamp"]], dim=0)

    return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Initialize worker seed."""
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Sparse4DDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Sparse4D."""

    def __init__(self, config: Dict):
        """Initialize DataModule.

        Args:
            dataset_config: Dataset configuration
        """
        super().__init__()
        self.config = config
        self.dataset_config = config.dataset
        self.train_config = config.train
        self.calib_dataset = None

        # Extract dataset parameters
        self.data_root = self.dataset_config["data_root"]
        self.classes = self.dataset_config["classes"]
        self.batch_size = self.dataset_config["batch_size"]
        self.num_workers = self.dataset_config["num_workers"]
        self.augmentation = self.dataset_config["augmentation"]
        self.normalize = self.dataset_config["normalize"]
        self.sequences = self.dataset_config["sequences"]
        self.train_dataset_cfg = self.dataset_config["train_dataset"]
        self.val_dataset_cfg = self.dataset_config["val_dataset"]
        self.use_h5_file_for_rgb = self.dataset_config["use_h5_file_for_rgb"]
        self.use_h5_file_for_depth = self.dataset_config["use_h5_file_for_depth"]

        # Lazy loading / pkl sampling parameters
        self.lazy_load = self.dataset_config.get("lazy_load", False)
        self.lazy_load_cache_size = self.dataset_config.get("lazy_load_cache_size", 50)
        self.pkl_sample_size = self.dataset_config.get("pkl_sample_size", 0)
        pkl_cam_counts_path = self.dataset_config.get("pkl_cam_counts_path", "")
        self.pkl_cam_counts_path = pkl_cam_counts_path if pkl_cam_counts_path else None

        # FPS drop augmentation
        self.fps_drop_prob = self.dataset_config.get("fps_drop_prob", 0)
        target_fps = self.dataset_config.get("target_fps_choices", None)
        self.target_fps_choices = list(target_fps) if target_fps else None

        # Camera subsampling
        self.max_cameras = self.dataset_config.get("max_cameras", -1)

        # Evaluation settings
        self.eval_dist_fcn = self.dataset_config.get("eval_dist_fcn", "center_distance")
        self.eval_hota = self.dataset_config.get("eval_hota", False)

        # Create transforms
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
        self.test_transforms = self._build_test_transforms()
        self.vis_transforms = self._build_vis_transforms()

        self.train_anno_file = self.dataset_config["train_dataset"]["ann_file"]
        self.val_anno_file = self.dataset_config["val_dataset"]["ann_file"]
        self.test_anno_file = self.dataset_config["test_dataset"]["ann_file"]

    def _build_train_transforms(self):
        """Build transforms for training data."""
        return Compose([
            LoadMultiViewImageFromFiles(to_float32=True, h5_file=self.use_h5_file_for_rgb),
            LoadDepthMap(max_depth=50, h5_file=self.use_h5_file_for_depth),
            ResizeCropFlipImage(),
            ResizeCropFlipMultiScaleDepthMap(downsample=[4, 8, 16]),
            BBoxRotation(),
            PhotoMetricDistortionMultiViewImage(),
            NormalizeMultiviewImage(
                mean=self.normalize["mean"],
                std=self.normalize["std"],
                to_rgb=self.normalize["to_rgb"]
            ),
            InstanceNameFilter(classes=self.classes),
            AICitySparse4DAdaptor()
        ])

    def _build_val_transforms(self):
        """Build transforms for validation data."""
        return Compose([
            LoadMultiViewImageFromFiles(to_float32=True, h5_file=self.use_h5_file_for_rgb),
            ResizeCropFlipImage(),
            NormalizeMultiviewImage(
                mean=self.normalize["mean"],
                std=self.normalize["std"],
                to_rgb=self.normalize["to_rgb"]
            ),
            AICitySparse4DAdaptor()
        ])

    def _build_vis_transforms(self):
        """Build transforms for visualization data."""
        return Compose([
            LoadMultiViewImageFromFiles(to_float32=True, h5_file=self.use_h5_file_for_rgb),
        ])

    def _build_test_transforms(self):
        """Build transforms for test data."""
        return self._build_val_transforms()

    def setup(self, stage: Optional[str] = None):
        """Setup datasets based on stage.

        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        if stage == 'fit' or stage is None:
            # Setup training dataset
            self.train_dataset = Omniverse3DDetTrackDataset(
                data_root=self.data_root,
                anno_file=self.train_anno_file,
                classes=self.classes,
                test_mode=False,
                use_valid_flag=True,
                augmentation=self.augmentation,
                sequences_split_num=self.sequences["split_num"],
                with_seq_flag=True,
                keep_consistent_seq_aug=self.sequences["keep_consistent_aug"],
                same_scene_in_batch=self.sequences["same_scene_in_batch"],
                transforms=self.train_transforms,
                train_dataset_cfg=self.train_dataset_cfg,
                lazy_load=self.lazy_load,
                lazy_load_cache_size=self.lazy_load_cache_size,
                pkl_sample_size=self.pkl_sample_size,
                pkl_cam_counts_path=self.pkl_cam_counts_path,
                fps_drop_prob=self.fps_drop_prob,
                target_fps_choices=self.target_fps_choices,
                max_cameras=self.max_cameras,
            )

            # Setup validation dataset
            self.val_dataset = Omniverse3DDetTrackDataset(
                data_root=self.data_root,
                anno_file=self.val_anno_file,
                classes=self.classes,
                test_mode=True,
                use_valid_flag=True,
                augmentation=self.augmentation,
                tracking=True,
                tracking_threshold=0.2,
                transforms=self.val_transforms,
                train_dataset_cfg=self.val_dataset_cfg,
                eval_dist_fcn=self.eval_dist_fcn,
                eval_hota=self.eval_hota,
            )

            logging.info(
                f"Loaded {len(self.train_dataset)} training samples "
                f"and {len(self.val_dataset)} validation samples"
            )

            # exit if len is 0
            if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
                raise ValueError("No samples found in training or validation dataset")

        if stage in ('test', 'predict'):
            # Setup test dataset

            self.test_dataset = Omniverse3DDetTrackDataset(
                data_root=self.data_root,
                anno_file=self.test_anno_file,
                classes=self.classes,
                test_mode=True,
                augmentation=self.augmentation,
                tracking=True,
                tracking_threshold=0.2,
                transforms=self.test_transforms,
                eval_dist_fcn=self.eval_dist_fcn,
                eval_hota=self.eval_hota,
            )
            self.val_dataset = self.test_dataset

            logging.info(f"Loaded {len(self.test_dataset)} test samples")

        if stage == 'calibration':
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            if isinstance(calib_cfg, dict):
                calib_images_dir = calib_cfg.get("images_dir", "")
            else:
                calib_images_dir = getattr(calib_cfg, "images_dir", "")

            if calib_images_dir:
                # Use test dataset for calibration with calibration images dir
                self.calib_dataset = Omniverse3DDetTrackDataset(
                    data_root=calib_images_dir,
                    anno_file=self.test_anno_file,
                    classes=self.classes,
                    test_mode=True,
                    augmentation=self.augmentation,
                    tracking=True,
                    tracking_threshold=0.2,
                    transforms=self.test_transforms
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )

    def train_dataloader(self):
        """Create training dataloader."""
        # Get distributed training settings
        world_size = self.train_config["num_gpus"]
        rank = int(os.environ.get('LOCAL_RANK', 0))

        # Create sampler
        sampler = GroupInBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            world_size=world_size,
            rank=rank,
            seed=self.train_config["seed"],
            skip_prob=0.5,
            sequence_flip_prob=0.1
        )

        # Create dataloader with batch_size=1 since sampler already creates batches
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            batch_sampler=sampler
        )

        return train_loader

    def val_dataloader(self):
        """Create validation dataloader."""
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return val_loader

    def test_dataloader(self):
        """Create test dataloader."""
        sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
        test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        return test_loader

    def predict_dataloader(self):
        """Create dataloader for prediction."""
        # Return the test dataloader for prediction
        sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
        test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return test_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration.

        Returns:
            calib_loader: PyTorch DataLoader used for calibration.
        """
        if self.calib_dataset is None:
            raise ValueError(
                "Calibration dataset is not initialized. "
                "Call setup(stage='calibration') first."
            )
        sampler = torch.utils.data.SequentialSampler(self.calib_dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
        calib_loader = DataLoader(
            self.calib_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return calib_loader
