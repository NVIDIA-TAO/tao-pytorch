# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object Detection dataset."""

from typing import Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized, local_broadcast_process_authkey
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import collate_fn

from nvidia_tao_pytorch.cv.grounding_dino.dataloader.odvg import build_odvg
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.serialized_dataset import build_shm_dataset
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.coco import CocoDetection, ODPredictDataset


class ODVGDataModule(pl.LightningDataModule):
    """Lightning DataModule for Object Detection.

    Supported stages (for ``setup(stage=...)``):
    * ``fit``        – build training & validation datasets
    * ``test``       – build evaluation dataset
    * ``predict``    – build inference dataset
    * ``calibration``– build calibration dataset used for post-training quantization

    The :pyfunc:`calib_dataloader` method returns the DataLoader created for the
    calibration stage.
    """

    def __init__(self, dataset_config, subtask_config=None):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration
            subtask_config (OmegaConf): subtask configuration

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.augmentation_config = dataset_config["augmentation"]
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.max_labels = dataset_config["max_labels"]
        self.pin_memory = dataset_config["pin_memory"]
        self.subtask_config = subtask_config
        # Placeholder for calibration dataset
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).

        Args:
            stage (str): stage options from fit, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):
            # prep validation
            val_data_sources = self.dataset_config["val_data_sources"]
            val_transform = build_transforms(self.augmentation_config, dataset_mode='val')
            self.val_dataset = CocoDetection(val_data_sources["json_file"],
                                             val_data_sources["image_dir"],
                                             transforms=val_transform)
            if is_distributed:
                self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        # Assign test dataset for use in dataloader
        if stage in ('test', None):
            test_data_sources = self.dataset_config["test_data_sources"]
            test_transforms = build_transforms(self.augmentation_config, dataset_mode='eval')
            self.test_dataset = CocoDetection(test_data_sources["json_file"],
                                              test_data_sources["image_dir"],
                                              transforms=test_transforms)

        # Assign predict dataset for use in dataloader
        if stage in ('predict', None):
            pred_data_sources = self.dataset_config["infer_data_sources"]
            pred_list = pred_data_sources.get("image_dir", [])
            if isinstance(pred_list, str):
                pred_list = [pred_list]
            if "captions" not in pred_data_sources:
                raise ValueError("'captions' field needs to be passed")
            else:
                captions = pred_data_sources["captions"]
            pred_transforms = build_transforms(self.augmentation_config, subtask_config=self.subtask_config, dataset_mode='infer')
            self.pred_dataset = ODPredictDataset(pred_list, captions, transforms=pred_transforms)

        # Prepare calibration dataset
        if stage in ("calibration", None):
            calib_sources = self.dataset_config.get("quant_calibration_data_sources", None)
            image_dir = getattr(calib_sources, "image_dir", None) if calib_sources else None
            json_file = getattr(calib_sources, "json_file", None) if calib_sources else None
            if image_dir is None and isinstance(calib_sources, dict):
                image_dir = calib_sources.get("image_dir", "")
                json_file = calib_sources.get("json_file", "")

            if image_dir:
                calib_transform = build_transforms(self.augmentation_config, dataset_mode='eval')
                self.calib_dataset = CocoDetection(
                    json_file or "",
                    image_dir,
                    transforms=calib_transform,
                )
            elif stage == "calibration":
                raise ValueError("quant_calibration_data_sources.image_dir must be provided for calibration stage.")

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_transform = build_transforms(self.augmentation_config, dataset_mode='train')
        train_data_sources = self.dataset_config["train_data_sources"]

        if self.dataset_config["dataset_type"] == "serialized":
            # Torchrun has different authkey which prohibits mp.pickler to work.
            # We need to instantitate this inside train_dataloader
            # instead of setup when the multiprocessing has already been spawned.
            local_broadcast_process_authkey()
            self.train_dataset = build_shm_dataset(train_data_sources, train_transform, max_labels=self.max_labels)
        else:
            self.train_dataset = build_odvg(train_data_sources, train_transform, max_labels=self.max_labels)

        if is_dist_avail_and_initialized():
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=torch.utils.data.BatchSampler(self.train_sampler, self.batch_size, drop_last=True)
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            PyTorch DataLoader used for validation.
        """
        return DataLoader(
            self.val_dataset,
            num_workers=0,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=self.val_sampler)

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn)

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            PyTorch DataLoader used for inference.
        """
        return DataLoader(
            self.pred_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn)

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration."""
        if self.calib_dataset is None:
            raise ValueError("Calibration dataset not initialized. Call setup(stage='calibration') with proper config.")
        return DataLoader(
            self.calib_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
        )
