# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Depth Net dataset."""

from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets import build_mono_dataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.transforms import build_mono_transforms
from nvidia_tao_pytorch.cv.depth_net.dataloader.custom_collate_fn import custom_collate_fn


class MonoDepthNetDataModule(pl.LightningDataModule):
    """Lightning DataModule for Depth Estimation."""

    def __init__(self, dataset_config, subtask_config=None):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration.
            subtask_config (OmegaConf): subtask configuration.

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.subtask_config = subtask_config
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).

        Args:
            stage (str): stage options from fit, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()

        if stage == 'fit':
            # prepare train dataset
            self.train_dataset_config = self.dataset_config["train_dataset"]
            self.train_transforms = build_mono_transforms(
                self.train_dataset_config["augmentation"], split='train', resize_target=True
            )
            self.train_dataset = build_mono_dataset(
                self.train_dataset_config["data_sources"],
                self.train_transforms,
                min_depth=self.dataset_config["min_depth"],
                max_depth=self.dataset_config["max_depth"],
                normalize_depth=self.dataset_config["normalize_depth"]
            )

            if is_dist_avail_and_initialized():
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset, shuffle=True, drop_last=True
                )
            else:
                self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

            # prep validation dataset
            self.val_dataset_config = self.dataset_config["val_dataset"]
            self.val_transforms = build_mono_transforms(
                self.val_dataset_config["augmentation"], split='val', resize_target=False
            )
            self.val_dataset = build_mono_dataset(
                self.val_dataset_config["data_sources"],
                self.val_transforms,
                min_depth=self.dataset_config["min_depth"],
                max_depth=self.dataset_config["max_depth"],
                normalize_depth=self.dataset_config["normalize_depth"]
            )

            if is_distributed:
                self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        # Assign predict dataset for use in dataloader
        if stage == 'predict':
            self.infer_dataset_config = self.dataset_config["infer_dataset"]
            self.infer_transforms = build_mono_transforms(
                self.infer_dataset_config["augmentation"], split='infer', resize_target=False
            )
            self.infer_dataset = build_mono_dataset(
                self.infer_dataset_config["data_sources"],
                self.infer_transforms,
                min_depth=self.dataset_config["min_depth"],
                max_depth=self.dataset_config["max_depth"],
                normalize_depth=self.dataset_config["normalize_depth"]
            )

        if stage == 'test':
            self.test_dataset_config = self.dataset_config["test_dataset"]
            self.test_transforms = build_mono_transforms(
                self.test_dataset_config["augmentation"], split='infer', resize_target=False
            )
            self.test_dataset = build_mono_dataset(
                self.test_dataset_config["data_sources"],
                self.test_transforms,
                min_depth=self.dataset_config["min_depth"],
                max_depth=self.dataset_config["max_depth"],
                normalize_depth=self.dataset_config["normalize_depth"]
            )

        # Prepare calibration dataset when stage is 'calibration'
        if stage == 'calibration':
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            if isinstance(calib_cfg, dict):
                calib_images_dir = calib_cfg.get("images_dir", "")
            else:
                calib_images_dir = getattr(calib_cfg, "images_dir", "")

            if calib_images_dir:
                # Use infer dataset config for transforms
                self.calib_dataset_config = self.dataset_config.get(
                    "infer_dataset", self.dataset_config.get("test_dataset", {})
                )
                self.calib_transforms = build_mono_transforms(
                    self.calib_dataset_config.get("augmentation", {}),
                    split='infer',
                    resize_target=False
                )
                # Create calibration data sources from images_dir
                calib_data_sources = [{"dataset_name": "infer", "data_file": calib_images_dir}]
                self.calib_dataset = build_mono_dataset(
                    calib_data_sources,
                    self.calib_transforms,
                    min_depth=self.dataset_config["min_depth"],
                    max_depth=self.dataset_config["max_depth"],
                    normalize_depth=self.dataset_config["normalize_depth"]
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.train_dataset_config["workers"],
            pin_memory=self.train_dataset_config["pin_memory"],
            batch_size=self.train_dataset_config["batch_size"],
            sampler=self.train_sampler,
            drop_last=True,
            multiprocessing_context='spawn' if self.train_dataset_config["workers"] > 0 else None,
            persistent_workers=self.train_dataset_config["workers"] > 0
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            PyTorch DataLoader used for validation.
        """
        if self.val_dataset_config["batch_size"] > 1:
            assert self.val_dataset_config["batch_size"] == len(self.val_dataset), "batch_size must be 1 for validation"

        val_loader = DataLoader(
            self.val_dataset,
            num_workers=self.val_dataset_config["workers"],
            pin_memory=self.val_dataset_config["pin_memory"],
            batch_size=self.val_dataset_config["batch_size"],
            drop_last=False,
            sampler=self.val_sampler,
            multiprocessing_context='spawn' if self.val_dataset_config["workers"] > 0 else None
        )
        return val_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            PyTorch DataLoader used for inference.
        """
        pred_loader = DataLoader(
            self.infer_dataset,
            num_workers=self.infer_dataset_config["workers"],
            pin_memory=self.infer_dataset_config["pin_memory"],
            batch_size=self.infer_dataset_config["batch_size"],
            drop_last=False,
            shuffle=False,
            collate_fn=custom_collate_fn,
            multiprocessing_context='spawn' if self.infer_dataset_config["workers"] > 0 else None
        )
        return pred_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.test_dataset,
            num_workers=self.test_dataset_config["workers"],
            pin_memory=self.test_dataset_config["pin_memory"],
            batch_size=self.test_dataset_config["batch_size"],
            drop_last=False,
            shuffle=False,
            collate_fn=custom_collate_fn,
            multiprocessing_context='spawn' if self.test_dataset_config["workers"] > 0 else None
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
        calib_loader = DataLoader(
            self.calib_dataset,
            num_workers=self.calib_dataset_config.get("workers", 4),
            pin_memory=self.calib_dataset_config.get("pin_memory", True),
            batch_size=self.calib_dataset_config.get("batch_size", 1),
            drop_last=False,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        return calib_loader
