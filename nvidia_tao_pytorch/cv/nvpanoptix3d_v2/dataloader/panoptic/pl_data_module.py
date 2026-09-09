# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2-Panoptic LightningDataModule."""

import json
import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.scannetpp import (
    ScanNetppCollator,
    ScanNetppDataset,
)

logger = logging.getLogger(__name__)


def build_data_module(data_cfg, seed):
    """Build the ScanNet++ panoptic data module."""
    return NVPanoptix3Dv2PanopticDataModule(data_cfg, seed=seed)


def resolve_panoptic_paths(data_cfg):
    """Resolve and validate the two preprocessed ScanNet++ split roots."""
    resolved = {}
    missing = []
    for split in ("train", "val"):
        key = f"{split}_preprocessed_root"
        configured_root = getattr(data_cfg, key, None)
        if not configured_root:
            raise ValueError(f"dataset.panoptic.{key} must be set")
        preprocessed_root = os.path.abspath(
            os.path.expanduser(str(configured_root))
        )
        if not os.path.isdir(preprocessed_root):
            missing.append(preprocessed_root)
            resolved[split] = preprocessed_root
            continue
        for filename in ("all_metadata.npz", "categories.json"):
            path = os.path.join(preprocessed_root, filename)
            if not os.path.isfile(path):
                missing.append(path)
        resolved[split] = preprocessed_root

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Invalid preprocessed ScanNet++ dataset; missing:\n  - {details}"
        )
    return resolved


def load_categories(root):
    """Read categories from one preprocessed ScanNet++ split."""
    path = os.path.join(str(root), "categories.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_vocabulary(data_cfg):
    """Load and validate the single native ScanNet++ taxonomy."""
    split_paths = resolve_panoptic_paths(data_cfg)
    categories = load_categories(split_paths["train"])
    val_categories = load_categories(split_paths["val"])
    taxonomy = [
        (category.get("id", index), category["name"], bool(category.get("isthing", True)))
        for index, category in enumerate(categories)
    ]
    val_taxonomy = [
        (category.get("id", index), category["name"], bool(category.get("isthing", True)))
        for index, category in enumerate(val_categories)
    ]
    if val_taxonomy != taxonomy:
        raise ValueError(
            "ScanNet++ train and validation roots must use the same categories.json taxonomy"
        )
    vocabulary = {
        "classes": [category["name"] for category in categories],
        "categories": categories,
    }
    logger.info("Loaded %d ScanNet++ classes", len(vocabulary["classes"]))
    return vocabulary


class NVPanoptix3Dv2PanopticDataModule(pl.LightningDataModule):
    """Data module for preprocessed ScanNet++ train and validation roots."""

    def __init__(self, data_cfg, seed=0):
        super().__init__()
        self.data_cfg = data_cfg
        self.seed = int(seed)
        self._split_paths = resolve_panoptic_paths(data_cfg)
        # Lightning's automatic set_epoch is disabled for this data module.
        self._train_sampler = None

    def set_train_epoch(self, epoch: int) -> None:
        """Forward the epoch to the distributed train sampler."""
        sampler = getattr(self, "_train_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

    def resolve_split_paths(self, split):
        """Return the validated preprocessed root for a split."""
        split_paths = getattr(self, "_split_paths", None)
        if split_paths is None:
            split_paths = resolve_panoptic_paths(self.data_cfg)
            self._split_paths = split_paths
        return split_paths[split]

    def build_dataset(self, split="train"):
        """Build the fixed ScanNet++ train or validation dataset."""
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported ScanNet++ split: {split!r}")

        resolution = [(int(size[0]), int(size[1])) for size in self.data_cfg.resolution]
        preprocessed_root = self.resolve_split_paths(split)
        return ScanNetppDataset(
            preprocessed_root=preprocessed_root,
            split=split,
            num_views=getattr(self.data_cfg, "num_views", 5),
            resolution=resolution,
            pairs_per_scene=getattr(self.data_cfg, "pairs_per_scene", 50),
            randomize_view_order=getattr(
                self.data_cfg, "randomize_view_order", True,
            ),
            seed=self.seed,
            photometric_augmentation=(
                getattr(self.data_cfg, "photometric_augmentation", None)
                if split == "train" else None
            ),
        )

    def common_dataloader(self, split="train", shuffle=True):
        """Build a train or validation dataloader."""
        dataset = self.build_dataset(split)

        batch_size = getattr(self.data_cfg, "batch_size", 1)
        num_workers = int(getattr(self.data_cfg, "num_workers", 8))

        if is_dist_avail_and_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle,
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        if split == "train":
            self._train_sampler = sampler

        collate_fn = ScanNetppCollator(classes=dataset.classes)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=(split == "train"),
            pin_memory=True,
            sampler=sampler,
            persistent_workers=(num_workers > 0),
            timeout=600 if num_workers > 0 else 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def train_dataloader(self):
        """Build the training dataloader."""
        return self.common_dataloader(split="train", shuffle=True)

    def val_dataloader(self):
        """Build the validation dataloader."""
        return self.common_dataloader(split="val", shuffle=False)

    def test_dataloader(self):
        """Evaluate on the ScanNet++ validation split."""
        return self.common_dataloader(split="val", shuffle=False)

    def predict_dataloader(self):
        """Run inference on the ScanNet++ validation split."""
        return self.common_dataloader(split="val", shuffle=False)
