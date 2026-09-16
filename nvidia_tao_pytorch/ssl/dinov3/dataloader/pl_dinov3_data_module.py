# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 data module with an opt-in canonical manifest input."""

from typing import Optional

import torch
from torch.utils.data import BatchSampler, DataLoader

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.ssl.dinov3.dataloader.dataset import (
    DinoV3Dataset,
    ShardAwareDistributedSampler,
)
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.collate import DinoV2Collate
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.transform import DinoV2Transform


class DinoV3DataModule(DinoV2DataModule):
    """Preserve DINOv2 behavior and add manifest-backed DINOv3 training."""

    def setup(self, stage: Optional[str] = None):
        """Build the legacy datasets or the opt-in manifest dataset."""
        manifest = self.dataset_config.get("train_manifest")
        if not manifest:
            super().setup(stage)
            return
        if stage in ("fit", None):
            transform_config = self.dataset_config.transform
            transform = DinoV2Transform(
                global_crops_number=transform_config["n_global_crops"],
                global_crops_scale=transform_config["global_crops_scale"],
                global_crops_size=transform_config["global_crops_size"],
                global_crops_identical=False,
                local_crops_number=transform_config["n_local_crops"],
                local_crops_scale=transform_config["local_crops_scale"],
                local_crops_size=transform_config["local_crops_size"],
                local_crops_identical=False,
            )
            self.train_dataset = DinoV3Dataset(
                root=self.train_image_dir,
                manifest_path=manifest,
                transform=transform,
                train=True,
                archive_cache_size=self.dataset_config.get("archive_cache_size", 8),
            )
            distributed = is_dist_avail_and_initialized()
            self.train_sampler = ShardAwareDistributedSampler(
                self.train_dataset,
                num_replicas=torch.distributed.get_world_size() if distributed else 1,
                rank=torch.distributed.get_rank() if distributed else 0,
                shuffle=True,
                shuffle_window=self.dataset_config.get("shard_shuffle_window", 256),
            )
        if stage in ("predict", None):
            super().setup("predict")

    def train_dataloader(self):
        """Keep all manifest rows, including the final incomplete batch."""
        if not self.dataset_config.get("train_manifest"):
            if self.num_workers == 0:
                return DataLoader(
                    self.train_dataset,
                    batch_sampler=BatchSampler(
                        self.train_sampler, self.batch_size, drop_last=True
                    ),
                    num_workers=0,
                    collate_fn=DinoV2Collate(patch_size=self.patch_size),
                    pin_memory=True,
                    persistent_workers=False,
                )
            return super().train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_sampler=BatchSampler(self.train_sampler, self.batch_size, drop_last=False),
            num_workers=self.num_workers,
            collate_fn=DinoV2Collate(patch_size=self.patch_size),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        """Permit synchronous DINOv3 prediction without altering DINOv2."""
        if self.num_workers > 0:
            return super().predict_dataloader()
        return DataLoader(
            self.predict_dataset,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
