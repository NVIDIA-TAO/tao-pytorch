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
"""PyTorch Lightning data module for OneFormer unified segmentation.

This module provides data loading and preparation functionality for training
and evaluation of OneFormer models using PyTorch Lightning framework.
"""

import logging
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.oneformer.dataloader.datasets import COCOUnifiedDataset
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.oneformer.dataloader.datasets import OneFormerPredictDataset

logger = logging.getLogger(__name__)


class SemSegmDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for semantic segmentation."""

    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

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

    def val_dataloader(self):
        """Create validation dataloader."""
        dataset_val = COCOUnifiedDataset(
            ann_path=self.data_cfg.dataset.val.annotations,
            img_dir=self.data_cfg.dataset.val.images,
            panoptic_dir=self.data_cfg.dataset.val.panoptic,
            cfg=self.data_cfg,
            is_training=False,
        )

        val_sampler = None
        if is_dist_avail_and_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False
            )
        else:
            val_sampler = torch.utils.data.SequentialSampler(dataset_val)

        collate_fn = (
            dataset_val.collate_fn
            if hasattr(dataset_val, "collate_fn")
            else dataset_val.dataset.collate_fn
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size=self.data_cfg.dataset.val.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.data_cfg.dataset.val.num_workers,
            drop_last=False,
            pin_memory=self.data_cfg.dataset.pin_memory,
            sampler=val_sampler,
        )
        return val_loader

    def test_dataloader(self):
        """Create test dataloader."""
        dataset_test = COCOUnifiedDataset(
            ann_path=self.data_cfg.dataset.test.annotations,
            img_dir=self.data_cfg.dataset.test.images,
            panoptic_dir=self.data_cfg.dataset.test.panoptic,
            cfg=self.data_cfg,
            is_training=False,
        )

        test_sampler = None
        if is_dist_avail_and_initialized():
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False
            )
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        collate_fn = (
            dataset_test.collate_fn
            if hasattr(dataset_test, "collate_fn")
            else dataset_test.dataset.collate_fn
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.data_cfg.dataset.test.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.data_cfg.dataset.test.num_workers,
            drop_last=False,
            pin_memory=self.data_cfg.dataset.pin_memory,
            sampler=test_sampler,
        )
        return test_loader

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
