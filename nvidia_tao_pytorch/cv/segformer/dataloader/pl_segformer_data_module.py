# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""SegFormer Data Module"""

from typing import Optional
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.segformer.dataloader.dataset import SFDataset
from nvidia_tao_pytorch.cv.segformer.dataloader.utils import (
    build_target_class_list, build_palette
)


class SFDataModule(pl.LightningDataModule):
    """Lightning DataModule for SegFormer."""

    def __init__(self, dataset_config):
        """Lightning DataModule Initialization.

        Args:
            dataset_config (dict): Configuration for the dataset
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = dataset_config["batch_size"]
        self.num_workers = dataset_config["workers"]
        self.root_dir = dataset_config["root_dir"]
        self.img_size = dataset_config["img_size"]
        self.dataset = dataset_config["dataset"]
        self.augmentation = dataset_config["augmentation"]
        self.label_transform = dataset_config["label_transform"]
        self.calib_dataset = None

        # This part is from mmengine, id_color_map is for visualization when n_class > 2
        target_classes = build_target_class_list(self.dataset_config)
        PALETTE, CLASSES, label_map, id_color_map = build_palette(target_classes)
        self.palette = PALETTE
        self.classes = CLASSES
        self.label_map = label_map
        self.color_map = id_color_map

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset.

        Args:
            stage (str): Stage of the dataset
        """
        is_distributed = is_dist_avail_and_initialized()

        if stage == 'fit' or stage is None:
            if self.dataset == 'SFDataset':
                self.train_dataset = SFDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split=self.dataset_config["train_split"],
                    img_size=self.img_size,
                    label_transform=self.label_transform,
                    to_tensor=True,
                    color_map=self.color_map
                )
                self.val_dataset = SFDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split=self.dataset_config["validation_split"],
                    img_size=self.img_size,
                    label_transform=self.label_transform,
                    to_tensor=True,
                    color_map=self.color_map
                )
            else:
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [SFDataset,])'
                    % self.dataset)

            if is_distributed:
                self.train_sampler = distributed.DistributedSampler(
                    self.train_dataset, shuffle=True
                )
            else:
                self.train_sampler = RandomSampler(self.train_dataset)

        if stage == 'test' or stage is None:
            if self.dataset == 'SFDataset':
                self.test_dataset = SFDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split=self.dataset_config["test_split"],
                    img_size=self.img_size,
                    label_transform=self.label_transform,
                    to_tensor=True,
                    color_map=self.color_map
                )
            else:
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [SFDataset,])'
                    % self.dataset)

        if stage == 'predict' or stage is None:
            if self.dataset == 'SFDataset':
                self.predict_dataset = SFDataset(
                    root_dir=self.root_dir,
                    augmentation=self.augmentation,
                    split=self.dataset_config["predict_split"],
                    img_size=self.img_size,
                    label_transform=self.label_transform,
                    to_tensor=True,
                    color_map=self.color_map
                )
            else:
                raise NotImplementedError(
                    'Wrong dataset name %s (choose one from [SFDataset,])'
                    % self.dataset)

        # Prepare calibration dataset when stage is 'calibration'
        # Uses the same dataset structure as test/val (expects images in {root_dir}/images/{split})
        if stage == 'calibration':
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            if isinstance(calib_cfg, dict):
                calib_root_dir = calib_cfg.get("images_dir", "")
            else:
                calib_root_dir = getattr(calib_cfg, "images_dir", "")

            if calib_root_dir:
                # Use test split structure - expects {calib_root_dir}/images/test/
                self.calib_dataset = SFDataset(
                    root_dir=calib_root_dir,
                    augmentation=self.augmentation,
                    split="test",
                    img_size=self.img_size,
                    label_transform=self.label_transform,
                    to_tensor=True,
                    color_map=self.color_map
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_sampler=BatchSampler(
                self.train_sampler, self.batch_size, drop_last=False
            ),
        )
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False
        )
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False
        )
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False
        )
        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration.

        Returns:
            calib_loader: PyTorch DataLoader used for calibration.
        """
        if self.calib_dataset is None:
            raise ValueError(
                "Calibration dataset is not initialized. "
                "Call setup(stage='calibration') first."
            )
        calib_loader = DataLoader(
            self.calib_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False
        )
        return calib_loader
