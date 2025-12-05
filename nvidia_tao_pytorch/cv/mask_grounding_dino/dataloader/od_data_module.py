# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Object Detection dataset."""

from typing import Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset

import pytorch_lightning as pl

from nvidia_tao_pytorch.core.distributed.comm import (
    is_dist_avail_and_initialized,
    local_broadcast_process_authkey,
)
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import collate_fn

from nvidia_tao_pytorch.cv.grounding_dino.dataloader.coco import ODPredictDataset
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.serialized_dataset import build_shm_dataset
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.odvg import ODVGDataset
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.coco import CocoDetection
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.rescoco import RESPredictDataset


def build_odvg(data_sources, transforms, max_labels=50, has_mask=True):
    """Load dataset

    Args:
        data_sources (list): List of different data sources.
        transforms (dict): augmentations to apply.
        max_labels (int): Maximum number of labels.
        has_mask (bool): Whether masks are present.
    """
    if type(data_sources).__name__ == "DictConfig":
        data_sources = [data_sources]

    dataset_list = []
    for data_source in data_sources:
        image_dir = data_source.image_dir
        json_file = data_source.json_file
        label_map = data_source.label_map if "label_map" in data_source else None
        dataset_list.append(
            ODVGDataset(
                image_dir,
                json_file,
                label_map,
                max_labels=max_labels,
                transforms=transforms,
                has_mask=has_mask,
            )
        )
    if len(dataset_list) > 1:
        train_dataset = ConcatDataset(dataset_list)
    else:
        train_dataset = dataset_list[0]
    return train_dataset


class ODVGDataModule(pl.LightningDataModule):
    """Lightning DataModule for Object Detection."""

    def __init__(self, dataset_config, subtask_config=None):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration
            subtask_config (OmegaConf): subtask configuration

        """
        super().__init__()
        self.dataset_config = dataset_config
        self.augmentation_config = dataset_config["augmentation"]
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.max_labels = dataset_config["max_labels"]   # [test dataset] 0 for VG Dataset, N = num_classes for original mask grounding DINO
        self.pin_memory = dataset_config["pin_memory"]
        self.subtask_config = subtask_config

    def setup(self, stage: Optional[str] = None):
        """ Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).

        Args:
            stage (str): stage options from fit, test, predict or None.

        """
        is_distributed = is_dist_avail_and_initialized()
        has_mask = self.dataset_config["has_mask"]
        if stage in ('fit', None):
            # prep validation
            val_data_sources = self.dataset_config["val_data_sources"]
            val_transform = build_transforms(self.augmentation_config, dataset_mode='val')
            if val_data_sources["data_type"] == "OD":
                self.val_dataset = CocoDetection(
                    val_data_sources["json_file"],
                    val_data_sources["image_dir"],
                    transforms=val_transform,
                    has_mask=has_mask
                )
            elif val_data_sources["data_type"] == "VG":
                self.val_dataset = ODVGDataset(
                    val_data_sources["image_dir"],
                    val_data_sources["json_file"],
                    None,
                    max_labels=self.max_labels,
                    transforms=val_transform,
                    has_mask=has_mask
                )
            else:
                assert False, "Invalid data type"
            if is_distributed:
                self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        # Assign test dataset for use in dataloader
        if stage in ('test', None):
            test_data_sources = self.dataset_config["test_data_sources"]
            test_transforms = build_transforms(self.augmentation_config, dataset_mode='eval')
            if test_data_sources["data_type"] == "OD":
                self.test_dataset = CocoDetection(
                    test_data_sources["json_file"],
                    test_data_sources["image_dir"],
                    transforms=test_transforms,
                    has_mask=has_mask
                )
            elif test_data_sources["data_type"] == "VG":
                self.test_dataset = ODVGDataset(
                    test_data_sources["image_dir"],
                    test_data_sources["json_file"],
                    None,
                    max_labels=self.max_labels,
                    transforms=test_transforms,
                    has_mask=has_mask
                )
            else:
                assert False, "Invalid data type"

        # Assign predict dataset for use in dataloader
        if stage in ('predict', None):
            pred_data_sources = self.dataset_config["infer_data_sources"]
            pred_transforms = build_transforms(self.augmentation_config, subtask_config=self.subtask_config, dataset_mode='infer')

            if pred_data_sources["data_type"] == "OD":
                pred_list = pred_data_sources.get("image_dir", [])
                if isinstance(pred_list, str):
                    pred_list = [pred_list]
                if "captions" not in pred_data_sources:
                    raise ValueError("'captions' field needs to be passed")
                captions = pred_data_sources["captions"]
                self.pred_dataset = ODPredictDataset(pred_list, captions, transforms=pred_transforms)
            elif pred_data_sources["data_type"] == "VG":
                data_dir = pred_data_sources.get("image_dir", None)
                json_file = pred_data_sources.get("json_file", None)
                self.pred_dataset = RESPredictDataset(data_dir, json_file, transforms=pred_transforms)
            else:
                assert False, "Invalid data type"

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_transform = build_transforms(self.augmentation_config, dataset_mode='train')
        train_data_sources = self.dataset_config["train_data_sources"]
        if self.dataset_config["dataset_type"] == "serialized":
            # Torchrun has different authkey which prohibits mp.pickler to work.
            # We need to instantitate this inside train_dataloader
            # instead of setup when the multiprocessing has already been spawned.
            local_broadcast_process_authkey()
            self.train_dataset = build_shm_dataset(train_data_sources, train_transform, max_labels=self.max_labels)
        else:
            self.train_dataset = build_odvg(train_data_sources, train_transform, max_labels=self.max_labels)

        if is_dist_avail_and_initialized():
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=torch.utils.data.BatchSampler(self.train_sampler, self.batch_size, drop_last=True)
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            PyTorch DataLoader used for validation.
        """
        return DataLoader(
            self.val_dataset,
            num_workers=0,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=self.val_sampler
        )

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            PyTorch DataLoader used for evaluation.
        """
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            PyTorch DataLoader used for inference.
        """
        return DataLoader(
            self.pred_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
