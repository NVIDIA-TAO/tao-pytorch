# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAE datasets."""
import glob
import os
import logging
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from timm.data import create_transform
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 9000000000
logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(self,
                 cfg=None,
                 is_training=False):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.is_training = is_training
        self.img_dir = cfg.dataset.train_data_sources if self.is_training else cfg.dataset.val_data_sources
        self.img_paths = [
            f
            for f in glob.glob(f"{self.img_dir}/**/*", recursive=True)
            if os.path.isfile(f) and f.lower().endswith(EXTENSIONS)
        ]
        self.num_samples = len(self.img_paths)
        self.transforms = self._get_train_transforms() if self.is_training else self._get_test_transforms()

    def __len__(self):
        """Length of the dataset."""
        return self.num_samples

    def _get_train_transforms(self):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                self.cfg.dataset.augmentation.input_size,
                scale=(self.cfg.dataset.augmentation.min_scale, self.cfg.dataset.augmentation.max_scale),
                interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
            # transforms.ColorJitter(
            #     brightness=self.brightness,
            #     contrast=self.contrast,
            #     saturation=self.saturation,
            #     hue=self.hue,
            # ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.dataset.augmentation.mean, std=self.cfg.dataset.augmentation.std)])
        return transform_train

    def _get_test_transforms(self):
        if self.cfg.dataset.augmentation.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(self.cfg.dataset.augmentation.input_size / crop_pct)
        transform_test = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.dataset.augmentation.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.dataset.augmentation.mean, std=self.cfg.dataset.augmentation.std)])
        return transform_test

    def __getitem__(self, index):
        """Per item."""
        img = Image.open(self.img_paths[index]).convert("RGB")
        img = self.transforms(img)
        data = {}
        data['image'] = img
        return data

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []

        for item in batch:
            images.append(item['image'])

        out['images'] = torch.stack(images)
        return out


class FinetuneDataset:
    """Dataset for pretraining."""

    def __init__(self,
                 cfg=None,
                 is_training=False):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.is_training = is_training
        self.img_dir = cfg.dataset.train_data_sources if self.is_training else cfg.dataset.val_data_sources
        self.transforms = self._get_train_transforms() if self.is_training else self._get_test_transforms()

    def _get_train_transforms(self):
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(
        #         self.cfg.dataset.augmentation.input_size,
        #         scale=(self.cfg.dataset.augmentation.min_scale, self.cfg.dataset.augmentation.max_scale),
        #         interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
        #     transforms.ColorJitter(
        #         brightness=self.brightness,
        #         contrast=self.contrast,
        #         saturation=self.saturation,
        #         hue=self.hue,
        #     ),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.cfg.dataset.augmentation.mean, std=self.cfg.dataset.augmentation.std)])
        transform_train = create_transform(
            input_size=self.cfg.dataset.augmentation.input_size,
            is_training=True,
            color_jitter=self.cfg.dataset.augmentation.color_jitter,
            auto_augment=self.cfg.dataset.augmentation.auto_aug,
            interpolation=self.cfg.dataset.augmentation.interpolation,
            re_prob=self.cfg.dataset.augmentation.re_prob,
            re_mode="pixel",
            re_count=1,
            scale=(self.cfg.dataset.augmentation.min_scale, self.cfg.dataset.augmentation.max_scale),
            ratio=(self.cfg.dataset.augmentation.min_ratio, self.cfg.dataset.augmentation.max_ratio),
            mean=self.cfg.dataset.augmentation.mean,
            std=self.cfg.dataset.augmentation.std,
            hflip=self.cfg.dataset.augmentation.hflip,
        )
        return transform_train

    def _get_test_transforms(self):
        if self.cfg.dataset.augmentation.input_size < 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(self.cfg.dataset.augmentation.input_size / crop_pct)
        transform_test = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.dataset.augmentation.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.dataset.augmentation.mean, std=self.cfg.dataset.augmentation.std)])
        return transform_test

    def build(self):
        """Build dataset."""
        return datasets.ImageFolder(self.img_dir, transform=self.transforms)


class PredictDataset(Dataset):
    """Dataset for prediction."""

    def __init__(self, cfg=None):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.img_dir = cfg.dataset.test_data_sources
        self.img_paths = [
            f
            for f in glob.glob(f"{self.img_dir}/**/*", recursive=True)
            if os.path.isfile(f) and f.lower().endswith(EXTENSIONS)
        ]
        self.img_paths = sorted(self.img_paths)
        self.num_samples = len(self.img_paths)
        self.transforms = self._get_test_transforms()

    def __len__(self):
        """Length of the dataset."""
        return self.num_samples

    def _get_test_transforms(self):
        if self.cfg.dataset.augmentation.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(self.cfg.dataset.augmentation.input_size / crop_pct)
        transform_test = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.dataset.augmentation.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.dataset.augmentation.mean, std=self.cfg.dataset.augmentation.std)])
        return transform_test

    def __getitem__(self, index):
        """Per item."""
        img = Image.open(self.img_paths[index]).convert("RGB")
        img = self.transforms(img)
        data = {}
        data['image'] = img
        data['image_path'] = self.img_paths[index]
        return data

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []
        paths = []
        for item in batch:
            images.append(item['image'])
            paths.append(item['image_path'])

        out['images'] = torch.stack(images)
        out['paths'] = paths
        return out
