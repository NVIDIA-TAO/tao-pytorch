# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Data Module"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl
from torchvision import transforms as T

from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.transform import DinoV2Transform
from nvidia_tao_pytorch.ssl.dinov3.dataloader.dataset import (
    DinoV3Dataset,
    ShardAwareDistributedSampler,
)
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.collate import DinoV2Collate
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized


class DinoV3DataModule(pl.LightningDataModule):
    """Lightning DataModule for Object Detection."""

    def __init__(self, experiment_config):
        """ Lightning DataModule

        Args:
            experiment_config: experiment configuration
        """
        super().__init__()

        self.experiment_config = experiment_config
        self.dataset_config = self.experiment_config.dataset
        self.batch_size = self.dataset_config["batch_size"]
        self.num_workers = self.dataset_config["workers"]
        self.train_image_dir = self.dataset_config.train_dataset["images_dir"]
        self.test_image_dir = self.dataset_config.test_dataset["images_dir"]
        self.patch_size = self.experiment_config.model.backbone["patch_size"]

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.
        """
        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):
            global_crops_number = self.dataset_config.transform["n_global_crops"]
            global_crops_scale = self.dataset_config.transform["global_crops_scale"]
            global_crops_size = self.dataset_config.transform["global_crops_size"]
            global_crops_identical = False
            local_crops_number = self.dataset_config.transform["n_local_crops"]
            local_crops_scale = self.dataset_config.transform["local_crops_scale"]
            local_crops_size = self.dataset_config.transform["local_crops_size"]
            local_crops_identical = False

            transform = DinoV2Transform(
                global_crops_number=global_crops_number,
                global_crops_scale=global_crops_scale,
                global_crops_size=global_crops_size,
                global_crops_identical=global_crops_identical,
                local_crops_number=local_crops_number,
                local_crops_scale=local_crops_scale,
                local_crops_size=local_crops_size,
                local_crops_identical=local_crops_identical
            )

            self.train_dataset = DinoV3Dataset(
                root=self.train_image_dir,
                manifest_path=self.dataset_config.get("train_manifest"),
                transform=transform,
                train=True
            )

            if self.dataset_config.get("train_manifest"):
                self.train_sampler = ShardAwareDistributedSampler(
                    self.train_dataset,
                    num_replicas=(
                        torch.distributed.get_world_size() if is_distributed else 1
                    ),
                    rank=torch.distributed.get_rank() if is_distributed else 0,
                    shuffle=True,
                )
            elif is_distributed:
                self.train_sampler = distributed.DistributedSampler(
                    self.train_dataset, shuffle=True
                )
            else:
                self.train_sampler = RandomSampler(self.train_dataset)

        if stage in ('predict', None):

            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            self.predict_dataset = DinoV3Dataset(
                root=self.test_image_dir,
                transform=transform,
                train=False
            )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            torch.utils.data.DataLoader: PyTorch DataLoader used for training.
        """
        collate = DinoV2Collate(patch_size=self.patch_size)

        train_loader = DataLoader(
            self.train_dataset,
            # A refinement epoch is a complete manifest pass. DistributedSampler
            # pads uneven partitions, so retaining the final batch visits every
            # declared sample. Preserve the legacy directory-backed behavior.
            batch_sampler=BatchSampler(
                self.train_sampler,
                self.batch_size,
                drop_last=not bool(self.dataset_config.get("train_manifest")),
            ),
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        return train_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            torch.utils.data.DataLoader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        return predict_loader
