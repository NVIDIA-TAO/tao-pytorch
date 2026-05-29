# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix-3D PL data module."""

from omegaconf import open_dict

import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.datasets import (
    Front3DDataset, Matterport3DDataset, NVPanoptix3DPredictDataset
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.sampler import BatchSeededSampler
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized


class NVPanoptix3DDataModule(pl.LightningDataModule):
    """NVPanoptix-3D data module."""

    def __init__(self, config):
        """Init."""
        super().__init__()
        self.config = config
        self.data_config = config.dataset

        # Dataset factory mapping dataset types to their initialization methods
        self.DATASET = {
            "front3d": Front3DDataset,
            "matterport": Matterport3DDataset,
        }

    def common_dataloader(self, mode="train", shuffle=True):
        """
        Build the common dataloader process for all stage

        Args:
            mode (str): Mode to build the dataloader for.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            dataloader: PyTorch DataLoader used for the given mode.
        """
        subset_config = self.data_config.get(mode, None)
        if not subset_config:
            raise ValueError(f"Invalid mode '{mode}' or subset config does not include '{mode}'")

        if self.data_config.name not in self.DATASET:
            raise ValueError(
                f"Dataset type '{self.data_config.name}' not found in {list(self.DATASET.keys())}"
            )

        with open_dict(subset_config):
            subset_config.is_training = mode == "train"
            subset_config.cfg = self.data_config
            subset_config.frustum_mask_path = self.data_config.get("frustum_mask_path", None)

        dataset = self.DATASET[self.data_config.name](**subset_config)

        sampler = None
        if is_dist_avail_and_initialized():
            sampler = StatefulDistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
        sampler = BatchSeededSampler(sampler, batch_size=int(subset_config.batch_size))

        # Use persistent_workers to avoid worker respawn overhead and memory fragmentation
        num_workers = int(getattr(subset_config, "num_workers", 0))
        use_persistent = num_workers > 0
        pin_memory = bool(torch.cuda.is_available())
        drop_last = bool(getattr(subset_config, "drop_last", mode == "train"))

        # Common dataloader kwargs
        common_kwargs = {
            "batch_size": subset_config.batch_size,
            "collate_fn": dataset.collate_fn,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "pin_memory": pin_memory,
            "persistent_workers": use_persistent,
        }

        # Only add prefetch_factor if using workers
        if num_workers > 0:
            # Default to 1 in DDP to reduce host RAM spikes; can override via env var.
            default_pf = 1 if is_dist_avail_and_initialized() else 2
            common_kwargs["prefetch_factor"] = default_pf

        # Use StatefulDataLoader so training can resume with sampler/dataloader state.
        dataloader = StatefulDataLoader(
            dataset,
            sampler=sampler,
            **common_kwargs
        )
        return dataloader

    def train_dataloader(self):
        """
        Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        return self.common_dataloader(mode="train", shuffle=True)

    def val_dataloader(self):
        """
        Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        return self.common_dataloader(mode="val", shuffle=False)

    def test_dataloader(self):
        """
        Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        return self.common_dataloader(mode="test", shuffle=False)

    def predict_dataloader(self):
        """
        Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        dataset_test = NVPanoptix3DPredictDataset(
            self.config.inference.images_dir,
            self.data_config,
        )
        test_sampler = None
        if is_dist_avail_and_initialized():
            test_sampler = StatefulDistributedSampler(dataset_test)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        predict_loader = StatefulDataLoader(
            dataset_test,
            batch_size=self.data_config.test.batch_size,
            collate_fn=dataset_test.collate_fn,
            num_workers=self.data_config.test.num_workers,
            drop_last=False,
            pin_memory=bool(torch.cuda.is_available()),
            sampler=test_sampler)
        return predict_loader
