# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Lightning data module for OneFormer unified segmentation.

This module provides data loading and preparation functionality for training
and evaluation of OneFormer models using PyTorch Lightning framework.
"""

import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.oneformer.dataloader.datasets import COCOUnifiedDataset
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.oneformer.dataloader.datasets import OneFormerPredictDataset

logger = logging.getLogger(__name__)


def _normalize_to_list(val, n=None):
    """Normalize a config value (str, list, or None) to a plain Python list."""
    if not val:
        return [""] * (n or 1)
    if isinstance(val, str):
        return [val]
    return list(val)


def _derive_dataset_names(ann_paths):
    """Derive short human-readable names from annotation file paths."""
    names = []
    for p in ann_paths:
        parts = Path(p).parts
        names.append(parts[-2] if len(parts) >= 2 else str(p))
    if len(names) != len(set(names)):
        names = [f"dataset_{i}" for i in range(len(names))]
    return names


class SemSegmDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for semantic segmentation."""

    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        # Prepare calibration dataset when stage is 'calibration'
        if stage == "calibration":
            calib_cfg = getattr(self.data_cfg.dataset, "quant_calibration_dataset", None)
            if calib_cfg is None:
                if isinstance(self.data_cfg.dataset, dict):
                    calib_cfg = self.data_cfg.dataset.get("quant_calibration_dataset", {})
                else:
                    calib_cfg = {}

            if hasattr(calib_cfg, "images_dir"):
                calib_images_dir = getattr(calib_cfg, "images_dir", "")
            else:
                calib_images_dir = calib_cfg.get("images_dir", "")

            if calib_images_dir:
                self.calib_dataset = OneFormerPredictDataset(
                    cfg=self.data_cfg,
                    images_dir=calib_images_dir,
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided for calibration stage."
                )

    def train_dataloader(self):
        """Create training dataloader."""
        dataset_train = COCOUnifiedDataset(
            ann_path=self.data_cfg.dataset.train.annotations,
            img_dir=self.data_cfg.dataset.train.images,
            panoptic_dir=self.data_cfg.dataset.train.panoptic,
            cfg=self.data_cfg,
            is_training=True,
        )

        train_sampler = None
        if is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=True
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)

        collate_fn = (
            dataset_train.collate_fn
            if hasattr(dataset_train, "collate_fn")
            else dataset_train.dataset.collate_fn
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.data_cfg.dataset.train.batch_size,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=self.data_cfg.dataset.train.num_workers,
            drop_last=False,
            pin_memory=self.data_cfg.dataset.pin_memory,
            sampler=train_sampler,
        )
        return train_loader

    def _build_eval_loaders(self, split_cfg):
        """Build one DataLoader per annotation file for a val/test split.

        Returns a single DataLoader when there is only one annotation,
        or a list of DataLoaders when there are multiple.  Also returns
        a parallel list of human-readable dataset names.
        """
        ann_paths = _normalize_to_list(split_cfg.annotations)
        n = len(ann_paths)
        img_dirs = _normalize_to_list(split_cfg.images, n)
        panoptic_dirs = _normalize_to_list(split_cfg.panoptic, n)
        if len(img_dirs) == 1 and n > 1:
            img_dirs = img_dirs * n
        if len(panoptic_dirs) == 1 and n > 1:
            panoptic_dirs = panoptic_dirs * n

        loaders = []
        for ann_path, img_dir, panoptic_dir in zip(ann_paths, img_dirs, panoptic_dirs):
            ds = COCOUnifiedDataset(
                ann_path=ann_path,
                img_dir=img_dir,
                panoptic_dir=panoptic_dir,
                cfg=self.data_cfg,
                is_training=False,
            )

            if is_dist_avail_and_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
            else:
                sampler = torch.utils.data.SequentialSampler(ds)

            collate_fn = ds.collate_fn if hasattr(ds, "collate_fn") else ds.dataset.collate_fn

            loaders.append(DataLoader(
                ds,
                batch_size=split_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=split_cfg.num_workers,
                drop_last=False,
                pin_memory=self.data_cfg.dataset.pin_memory,
                sampler=sampler,
            ))

        try:
            cfg_names = split_cfg.names
        except Exception:
            cfg_names = None
        if cfg_names is not None:
            names = _normalize_to_list(cfg_names)
            assert len(names) == n, (
                f"dataset.names has {len(names)} entries but {n} annotation files were given"
            )
            logger.info("Using configured dataset names: %s", names)
        else:
            names = _derive_dataset_names(ann_paths)
            logger.info("Using auto-derived dataset names: %s", names)
        return loaders, names

    def val_dataloader(self):
        """Create validation dataloader(s) -- one per annotation file."""
        loaders, self.val_dataset_names = self._build_eval_loaders(
            self.data_cfg.dataset.val
        )
        return loaders if len(loaders) > 1 else loaders[0]

    def test_dataloader(self):
        """Create test dataloader(s) -- one per annotation file."""
        loaders, self.test_dataset_names = self._build_eval_loaders(
            self.data_cfg.dataset.test
        )
        return loaders if len(loaders) > 1 else loaders[0]

    def predict_dataloader(self):
        """Create prediction dataloader."""
        dataset_predict = OneFormerPredictDataset(
            cfg=self.data_cfg
        )

        predict_sampler = None
        if is_dist_avail_and_initialized():
            predict_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_predict, shuffle=False
            )
        else:
            predict_sampler = torch.utils.data.SequentialSampler(dataset_predict)

        collate_fn = (
            dataset_predict.collate_fn
            if hasattr(dataset_predict, "collate_fn")
            else dataset_predict.dataset.collate_fn
        )

        predict_loader = DataLoader(
            dataset_predict,
            batch_size=self.data_cfg.dataset.test.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.data_cfg.dataset.test.num_workers,
            drop_last=False,
            pin_memory=self.data_cfg.dataset.pin_memory,
            sampler=predict_sampler,
        )
        return predict_loader

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

        collate_fn = (
            self.calib_dataset.collate_fn
            if hasattr(self.calib_dataset, "collate_fn")
            else None
        )

        calib_loader = DataLoader(
            self.calib_dataset,
            batch_size=self.data_cfg.dataset.val.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.data_cfg.dataset.val.num_workers,
            drop_last=False,
            pin_memory=self.data_cfg.dataset.pin_memory,
        )
        return calib_loader
