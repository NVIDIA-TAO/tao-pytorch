# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom LightningDataModule for MAL."""

import logging
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.mal.datasets.coco import BoxLabelCOCO, InstSegCOCO, InstSegCOCOwithBoxInput
from nvidia_tao_pytorch.cv.mal.datasets.data_aug import data_aug_pipelines, custom_collate_fn
logger = logging.getLogger(__name__)


class WSISDataModule(pl.LightningDataModule):
    """Weakly supervised instance segmentation data module."""

    def __init__(self,
                 num_workers,
                 cfg=None):
        """Initialize train/val dataset and dataloader.

        Args:
            num_workers (int): Total number of workers
            cfg (OmegaConf): Hydra config
        """
        super().__init__()
        self.cfg = cfg
        self.num_workers = num_workers
        self.train_transform = data_aug_pipelines['train'](cfg)
        self.test_transform = data_aug_pipelines['test'](cfg)
        assert self.cfg.dataset.type == 'coco', 'only COCO format is supported.'
        self.box_inputs = None

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            logger.info("Loading train set...")
            self.train_dataset = BoxLabelCOCO(
                self.cfg.dataset.train_ann_path,
                self.cfg.dataset.train_img_dir,
                min_obj_size=self.cfg.dataset.min_obj_size,
                max_obj_size=self.cfg.dataset.max_obj_size,
                transform=self.train_transform, cfg=self.cfg)
            logger.info("Train set is loaded successfully.")

        if stage in ('fit', 'test', 'predict'):
            logger.info("Loading validation set...")
            build_dataset = InstSegCOCOwithBoxInput if self.box_inputs else InstSegCOCO

            self.val_dataset = build_dataset(
                self.cfg.dataset.val_ann_path,
                self.cfg.dataset.val_img_dir,
                min_obj_size=0,
                max_obj_size=1e9,
                load_mask=self.cfg.dataset.load_mask,
                transform=self.test_transform,
                box_inputs=self.box_inputs
            )
            logger.info("Validation set is loaded successfully.")

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True,
            num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = DataLoader(
            self.val_dataset, collate_fn=custom_collate_fn,
            batch_size=self.cfg.train.batch_size, num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.val_dataset, collate_fn=custom_collate_fn,
            batch_size=self.cfg.train.batch_size, num_workers=self.num_workers)

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.val_dataset, collate_fn=custom_collate_fn,
            batch_size=self.cfg.train.batch_size, num_workers=self.num_workers)

        return predict_loader
