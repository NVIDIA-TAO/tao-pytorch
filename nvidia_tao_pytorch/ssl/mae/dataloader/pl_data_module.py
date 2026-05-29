# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom LightningDataModule for MAE."""

import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.ssl.mae.dataloader.datasets import PretrainDataset, FinetuneDataset, PredictDataset
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
logger = logging.getLogger(__name__)


class MAEDataModule(pl.LightningDataModule):
    """MAE data module."""

    def __init__(self, cfg):
        """Init."""
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        if self.cfg.train.stage == 'pretrain':
            dataset_train = PretrainDataset(
                self.cfg,
                is_training=True,
            )
        elif self.cfg.train.stage == 'finetune':
            dataset_train = FinetuneDataset(
                cfg=self.cfg,
                is_training=True,
            ).build()
        else:
            raise NotImplementedError(f"The train stage ({self.cfg.train.stage}) is not supported.")

        train_sampler = None
        if is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=not train_sampler,
            collate_fn=dataset_train.collate_fn if self.cfg.train.stage == 'pretrain' else None,
            num_workers=self.cfg.dataset.num_workers_per_gpu,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler)
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        if self.cfg.train.stage == 'pretrain':
            dataset_val = PretrainDataset(
                self.cfg,
                is_training=False,
            )
        elif self.cfg.train.stage == 'finetune':
            dataset_val = FinetuneDataset(
                cfg=self.cfg,
                is_training=False,
            ).build()
        else:
            raise NotImplementedError(f"The train stage ({self.cfg.train.stage}) is not supported.")

        val_sampler = None
        if is_dist_avail_and_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val)
        else:
            val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        val_loader = DataLoader(
            dataset_val,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            collate_fn=dataset_val.collate_fn if self.cfg.train.stage == 'pretrain' else None,
            num_workers=self.cfg.dataset.num_workers_per_gpu,
            drop_last=False,
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
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        if self.cfg.train.stage == 'pretrain':
            raise NotImplementedError("Inference with a pretrained model is not support!")
        elif self.cfg.train.stage == 'finetune':
            dataset_pred = PredictDataset(cfg=self.cfg)
        else:
            raise NotImplementedError(f"The train stage ({self.cfg.train.stage}) is not supported.")

        pred_sampler = None
        if is_dist_avail_and_initialized():
            pred_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_pred)
        else:
            pred_sampler = torch.utils.data.SequentialSampler(dataset_pred)
        pred_loader = DataLoader(
            dataset_pred,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            collate_fn=dataset_pred.collate_fn,
            num_workers=self.cfg.dataset.num_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=pred_sampler)

        return pred_loader
