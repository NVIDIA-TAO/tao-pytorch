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

"""Depth Net dataset Module."""

from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.depth_net.dataloader.stereo_datasets import build_stereo_dataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.transforms import build_stereo_transforms


class StereoDepthNetDataModule(pl.LightningDataModule):
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
            multiprocessing_context='spawn' if self.test_dataset_config["workers"] > 0 else None
        )
        return test_loader

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch
            tensor datasets for each split (train, val, test).
        Args:
            stage (str): stage options from fit, test, predict or None.
        """
        is_distributed = is_dist_avail_and_initialized()
        max_disparity = self.dataset_config["max_disparity"]

        if stage in ('fit', None):
            # prepare training dataset
            self.train_dataset_config = self.dataset_config["train_dataset"]
            self.train_transforms = build_stereo_transforms(
                self.train_dataset_config["augmentation"],
                max_disparity=max_disparity,
                split='train'
            )
            self.train_dataset = build_stereo_dataset(
                self.train_dataset_config["data_sources"],
                transform=self.train_transforms,
                max_disparity=max_disparity,
            )

            if is_dist_avail_and_initialized():
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset,
                    shuffle=True,
                    drop_last=True
                )
            else:
                self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

            # Prepare validation dataset
            self.val_dataset_config = self.dataset_config["val_dataset"]
            self.val_transforms = build_stereo_transforms(
                self.val_dataset_config["augmentation"],
                max_disparity=max_disparity,
                split='val',
            )
            self.val_dataset = build_stereo_dataset(
                self.val_dataset_config['data_sources'],
                transform=self.val_transforms,
                max_disparity=max_disparity
            )

            if is_distributed:
                self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        # Assign predict dataset for use in dataloader
        if stage in ('predict', None):
            self.infer_dataset_config = self.dataset_config["infer_dataset"]
            self.infer_transforms = build_stereo_transforms(
                self.infer_dataset_config["augmentation"],
                max_disparity=max_disparity,
                split='infer',
            )
            self.infer_dataset = build_stereo_dataset(
                self.infer_dataset_config['data_sources'],
                transform=self.infer_transforms,
                max_disparity=max_disparity
            )

        # Assign test dataset for use in dataloader
        if stage in ('test', None):
            self.test_dataset_config = self.dataset_config["test_dataset"]
            self.test_transforms = build_stereo_transforms(
                self.test_dataset_config["augmentation"],
                max_disparity=max_disparity,
                split='infer',
            )
            self.test_dataset = build_stereo_dataset(
                self.test_dataset_config['data_sources'],
                transform=self.test_transforms,
                max_disparity=max_disparity
            )
