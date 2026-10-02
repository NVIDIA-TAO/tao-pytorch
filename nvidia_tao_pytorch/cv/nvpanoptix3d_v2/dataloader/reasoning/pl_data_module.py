# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2-Reasoning LightningDataModule."""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.utils import (
    get_config_value,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.reasoning.reasoning_dataset import (
    ReasoningCollator, ReasoningSegDataset
)


def build_data_module(data_cfg):
    """Factory used by ``scripts/train_sam.py``."""
    return NVPanoptix3Dv2ReasoningDataModule(data_cfg)


class NVPanoptix3Dv2ReasoningDataModule(pl.LightningDataModule):
    """Builds train/val :class:`ReasoningSegDataset` loaders from the data config.

    Expected ``dataset`` config keys::

        train_manifest: /path/to/train.jsonl   # required
        val_manifest:   /path/to/val.jsonl     # optional
        resolution:     [518, 518]             # single (H, W) or bucket list
        num_views:      5
        batch_size:     1
        num_workers:    4
        depth_scale:    1000.0                 # stored units per meter
    """

    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.resolution = get_config_value(data_cfg, "resolution", [518, 518])
        self.num_views = get_config_value(data_cfg, "num_views", 5)
        self.batch_size = int(get_config_value(data_cfg, "batch_size", 1))
        self.num_workers = int(get_config_value(data_cfg, "num_workers", 4))
        self.depth_scale = float(get_config_value(data_cfg, "depth_scale", 1000.0))
        self.collator = ReasoningCollator()

        train_manifest = get_config_value(data_cfg, "train_manifest", None)
        if not train_manifest:
            raise ValueError("dataset.train_manifest is required for NVPanoptix3Dv2-Reasoning")
        self.train_ds = ReasoningSegDataset(
            manifest=train_manifest,
            resolution=self.resolution,
            num_views=self.num_views,
            require_seg_token=bool(
                get_config_value(data_cfg, "require_seg_token", True)
            ),
            depth_scale=self.depth_scale,
        )
        val_manifest = get_config_value(data_cfg, "val_manifest", None)
        self.val_ds = (
            ReasoningSegDataset(
                manifest=val_manifest,
                resolution=self.resolution,
                num_views=self.num_views,
                depth_scale=self.depth_scale,
            )
            if val_manifest
            else None
        )
        # Kept for TAO training code that expects datamodules to expose classes.
        self.classes = []

    def train_dataloader(self):
        """Training loader (shuffled)."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Validation loader (or ``None`` if no val manifest)."""
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Evaluation loader. The reasoning variant scores the val manifest."""
        return self.val_dataloader()

    def predict_dataloader(self):
        """Inference loader. The reasoning variant predicts over the val manifest."""
        return self.val_dataloader()
