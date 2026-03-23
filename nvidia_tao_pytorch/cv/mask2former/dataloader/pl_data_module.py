# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Custom LightningDataModule for Mask2former."""

import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.cv.mask2former.dataloader.datasets import (
    COCODataset, COCOPanopticDataset, ADEDataset, PredictDataset
)
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized

logger = logging.getLogger(__name__)


class SemSegmDataModule(pl.LightningDataModule):
    """Mask2former data module."""

    def __init__(self, data_cfg):
        """Init."""
        super().__init__()
        self.data_cfg = data_cfg
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup the datasets for different stages."""
        # Prepare calibration dataset when stage is 'calibration'
        if stage == "calibration":
            calib_cfg = getattr(self.data_cfg, "quant_calibration_dataset", None)
            if calib_cfg is None:
                if isinstance(self.data_cfg, dict):
                    calib_cfg = self.data_cfg.get("quant_calibration_dataset", {})
                else:
                    calib_cfg = {}

            if hasattr(calib_cfg, "images_dir"):
                calib_images_dir = getattr(calib_cfg, "images_dir", "")
            else:
                calib_images_dir = calib_cfg.get("images_dir", "")

            if calib_images_dir:
                self.calib_dataset = PredictDataset(
                    calib_images_dir,
                    self.data_cfg,
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided for calibration stage."
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        if self.data_cfg.train.type == 'ade':
            dataset_train = ADEDataset(
                self.data_cfg.train.annot_file,
                self.data_cfg.train.root_dir,
                self.data_cfg,
                is_training=True,
            )
        elif self.data_cfg.train.type == 'coco':
            dataset_train = COCODataset(
                self.data_cfg.train.instance_json,
                self.data_cfg.train.img_dir,
                cfg=self.data_cfg,
                is_training=True,
            )
        elif self.data_cfg.train.type == 'coco_panoptic':
            dataset_train = COCOPanopticDataset(
                self.data_cfg.train.panoptic_json,
                self.data_cfg.train.img_dir,
                self.data_cfg.train.panoptic_dir,
                cfg=self.data_cfg,
                is_training=True,
            )
        else:
            raise NotImplementedError(
                f"The dataset type ({self.data_cfg.train.type}) is not supported."
            )

        train_sampler = None
        if is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.data_cfg.train.batch_size,
            shuffle=not train_sampler,
            collate_fn=dataset_train.collate_fn,
            num_workers=self.data_cfg.train.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler)
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        if self.data_cfg.val.type == 'ade':
            dataset_val = ADEDataset(
                self.data_cfg.val.annot_file,
                self.data_cfg.val.root_dir,
                self.data_cfg,
                is_training=False,
            )
        elif self.data_cfg.val.type == 'coco':
            dataset_val = COCODataset(
                self.data_cfg.val.instance_json,
                self.data_cfg.val.img_dir,
                cfg=self.data_cfg,
                is_training=False,
            )
        elif self.data_cfg.val.type == 'coco_panoptic':
            dataset_val = COCOPanopticDataset(
                self.data_cfg.val.panoptic_json,
                self.data_cfg.val.img_dir,
                self.data_cfg.val.panoptic_dir,
                cfg=self.data_cfg,
                is_training=False,
            )
        else:
            raise NotImplementedError(
                f"The dataset type ({self.data_cfg.val.type}) is not supported."
            )

        val_sampler = None
        if is_dist_avail_and_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val)
        else:
            val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        val_loader = DataLoader(
            dataset_val,
            batch_size=self.data_cfg.val.batch_size,
            shuffle=False,
            collate_fn=dataset_val.collate_fn,
            num_workers=self.data_cfg.val.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=val_sampler)
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        return self.val_dataloader()

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        dataset_test = PredictDataset(
            self.data_cfg.test.img_dir,
            self.data_cfg,
        )
        test_sampler = None
        if is_dist_avail_and_initialized():
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        predict_loader = DataLoader(
            dataset_test,
            batch_size=self.data_cfg.test.batch_size,
            shuffle=False,
            collate_fn=dataset_test.collate_fn,
            num_workers=self.data_cfg.test.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=test_sampler)
        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration.

        Returns:
            calib_loader: PyTorch DataLoader used for calibration.
        """
        if self.calib_dataset is None:
            raise ValueError(
                "Calibration dataset is not initialized. Call setup(stage='calibration') first."
            )
        calib_loader = DataLoader(
            self.calib_dataset,
            batch_size=self.data_cfg.val.batch_size,
            shuffle=False,
            collate_fn=self.calib_dataset.collate_fn,
            num_workers=self.data_cfg.val.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return calib_loader
