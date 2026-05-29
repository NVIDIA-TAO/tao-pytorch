# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Data Module for training and retrieval-based validation."""

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import get_custom_dataloader
from nvidia_tao_pytorch.multimodal.clip.dataloader.wds import get_train_dataloader


class CLIPDataModule(pl.LightningDataModule):
    """Lightning DataModule for CLIP with retrieval-based validation."""

    def __init__(self, dataset_config, tokenizer, resume_step,
                 preprocess, world_size):
        """Initialize the CLIPDataModule.

        Args:
            dataset_config: Configuration for the dataset.
            tokenizer: Tokenizer to process input data.
            resume_step: Step to resume training from.
            preprocess: Tuple containing preprocessing functions for training and validation.
            world_size: The number of processes for distributed training.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.preprocess_train, self.preprocess_val = preprocess
        self.resume_step = resume_step
        self.world_size = world_size

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Prepare dataloaders for each stage.

        Args:
            stage: Stage options from fit, validate, test, predict or None.
        """
        if hasattr(self, '_setup_done') and self._setup_done:
            return
        self._setup_done = True

        is_distributed = is_dist_avail_and_initialized()

        if stage in ('fit', None):
            self._setup_train_dataloader(is_distributed)

        if stage in ('fit', 'test', None):
            self._setup_val_dataloader()

    def _setup_train_dataloader(self, is_distributed: bool):
        """Setup training dataloader."""
        train_type = self.dataset_config.train.type

        if train_type == 'custom':
            self.train_dataset = get_custom_dataloader(
                datasets=self.dataset_config.train.datasets,
                transform=self.preprocess_train,
                tokenizer=self.tokenizer,
                batch_size=self.dataset_config.train.batch_size,
                num_workers=self.dataset_config.train.num_workers,
                seed=self.dataset_config.seed,
                shuffle=True,
                pin_memory=self.dataset_config.pin_memory,
                is_distributed=is_distributed,
                mode='train'
            )
        elif train_type == 'wds':
            self.train_dataset = get_train_dataloader(
                root=self.dataset_config.train.wds.root_dir,
                urls=self.dataset_config.train.wds.shard_list_file,
                samples_per_file=self.dataset_config.train.wds.samples_per_shard,
                batch_size=self.dataset_config.train.batch_size,
                seed=self.dataset_config.seed,
                num_workers=self.dataset_config.train.num_workers,
                resume_step=self.resume_step,
                transform=lambda data: (
                    self.preprocess_train(data[0]),
                    self.tokenizer(data[1])[0],
                ),
                world_size=self.world_size,
                pin_memory=self.dataset_config.pin_memory
            )
        else:
            raise ValueError(f"Unknown train dataset type: {train_type}. "
                             f"Choose from [custom, wds]")

    def _setup_val_dataloader(self):
        """Setup validation dataloader for retrieval evaluation."""
        val_cfg = self.dataset_config.val

        # Check if validation is configured
        if not val_cfg.datasets:
            logging.warning(
                "Validation not configured. Add datasets to val.datasets "
                "to enable retrieval evaluation during training."
            )
            return

        self.val_dataset = get_custom_dataloader(
            datasets=val_cfg.datasets,
            transform=self.preprocess_val,
            tokenizer=self.tokenizer,
            batch_size=val_cfg.batch_size,
            num_workers=val_cfg.num_workers,
            seed=self.dataset_config.seed,
            shuffle=False,
            pin_memory=self.dataset_config.pin_memory,
            is_distributed=None,
            mode='val'
        )
        logging.info(f"Validation dataloader: {len(val_cfg.datasets)} dataset(s), "
                     f"{len(self.val_dataset.dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Build the dataloader for training."""
        return self.train_dataset

    def val_dataloader(self) -> Optional[DataLoader]:
        """Build dataloader for validation (retrieval evaluation)."""
        return self.val_dataset

    def test_dataloader(self) -> Optional[DataLoader]:
        """Build dataloader for testing (same as validation)."""
        return self.val_dataset
