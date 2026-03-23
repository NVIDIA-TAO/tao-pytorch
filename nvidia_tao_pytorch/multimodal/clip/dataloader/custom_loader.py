# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Custom filesystem-based image-text dataset loader for CLIP.

This module provides a DataLoader for datasets where images and their
corresponding text captions are stored as individual files on disk.
Used when dataset type is 'custom' in the config.
"""

import random
from pathlib import Path
from typing import List, Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler, Dataset

from nvidia_tao_pytorch.core.tlt_logging import logging


# New dataloader that takes a list of dataset sources
class ImageTextDataset(Dataset):
    """Image Text dataloader for fine-tuning."""

    def __init__(self, datasets: List[dict], transform: Callable = None,
                 tokenizer: Callable = None, zero_shot_eval=False, mapping=None,
                 mode='train'):
        """
        Initializes the ImageTextDataset.

        Args:
            datasets (List[dict]): List of dataset configurations.
            transform (Callable): Transform function for images.
            tokenizer (Callable): Tokenizer function for texts.
            zero_shot_eval (bool): Flag for zero-shot evaluation.
            mapping (Optional[dict]): Mapping for text transformations.
            mode (str): Dataset mode ('train' or 'val').
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.zero_shot_eval = zero_shot_eval
        self.mapping = mapping
        self.mode = mode

        self.image_text_pairs = []
        if len(datasets) > 1 and self.zero_shot_eval:
            raise NotImplementedError(
                "Validation currently only supports a single dataset as input")

        # Supported image file extensions
        supported_extensions = ['*.jpg', '*.jpeg', '*.png']
        # Iterate over each dataset configuration
        for dataset in datasets:
            image_dir = Path(dataset['image_dir'])
            caption_dir = Path(dataset.get('caption_dir') or image_dir)
            image_list_file = dataset.get('image_list_file')
            caption_file_suffix = dataset.get('caption_file_suffix', '.txt')

            # If image_list_file is provided, read image list from the text file
            if image_list_file:
                with open(image_list_file, 'r') as file:
                    image_list = [
                        line.strip() for line in file if line.strip()
                    ]
                # Trust the image list file - skip existence checks for speed
                for image_name in image_list:
                    image_path = image_dir / image_name
                    text_path = caption_dir / \
                        Path(image_name).with_suffix(caption_file_suffix)
                    self.image_text_pairs.append((image_path, text_path))
            else:
                # No image list - glob for files and verify existence
                logging.info(
                    f"image_list_file not provided. Using all images with "
                    f"extensions {supported_extensions} from {image_dir}"
                )
                image_list = [
                    p.name for ext in supported_extensions for p in image_dir.glob(ext)]
                for image_name in image_list:
                    image_path = image_dir / image_name
                    text_path = caption_dir / \
                        Path(image_name).with_suffix(caption_file_suffix)
                    if text_path.exists():
                        self.image_text_pairs.append((image_path, text_path))
        logging.info(
            f"Loaded {len(self.image_text_pairs)} image-text pairs ({self.mode})")
        if not self.image_text_pairs:
            raise ValueError("No valid image-text pairs found across datasets")

    def __len__(self):
        """Returns the number of image-text pairs in the dataset."""
        return len(self.image_text_pairs)

    def __getitem__(self, idx):
        """
        Retrieves an image-text pair from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Image, str]: A tuple containing the transformed image and the corresponding text.
        """
        image_path, text_path = self.image_text_pairs[idx]

        image = Image.open(image_path).convert('RGB')
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()

        if self.transform:
            image = self.transform(image)
        if self.zero_shot_eval and self.mapping:
            # For zero-shot eval, map text to class index BEFORE tokenization
            text = self.mapping.get(text, text)
            # If mapping found, text is now an integer class index - don't tokenize
        elif self.tokenizer:
            text = self.tokenizer(text)[0]

        return image, text


def get_custom_dataloader(
    datasets: List[dict],
    batch_size: int = 32,
    transform: Callable = None,
    tokenizer: Callable = None,
    num_workers: int = 0,
    seed: int = 42,
    zero_shot_eval: bool = False,
    mapping=None,
    shuffle=True,
    pin_memory=True,
    is_distributed=None,
    mode='train'
):
    """
    Creates a DataLoader for custom filesystem-based image-text datasets.

    Args:
        datasets (List[dict]): List of dataset configurations.
        batch_size (int): Size of batches.
        transform (Callable): Transform function for images.
        tokenizer (Callable): Tokenizer function for texts.
        num_workers (int): Number of subprocesses to use for data loading.
        seed (int): Random seed for reproducibility.
        zero_shot_eval (bool): Flag for zero-shot evaluation.
        mapping (Optional[dict]): Mapping for text transformations.
        shuffle (bool): Flag to shuffle data.
        pin_memory (bool): Flag to pin memory.
        is_distributed (Optional[bool]): Flag for distributed training.
        mode (str): Mode for the DataLoader ('train' or 'val').

    Returns:
        DataLoader: A DataLoader for the specified datasets.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = ImageTextDataset(
        datasets=datasets,
        transform=transform,
        tokenizer=tokenizer,
        zero_shot_eval=zero_shot_eval,
        mapping=mapping,
        mode=mode
    )
    dataloader_kwargs = {}

    # TODO: Check multi-gpu training if automatically supports fast training or 1 vs multi-gpu issue?
    batch_sampler = None
    if mode == 'train':
        if is_distributed:
            batch_sampler = distributed.DistributedSampler(
                dataset, shuffle=True)
        else:
            batch_sampler = RandomSampler(dataset)
    if batch_sampler:
        dataloader_kwargs['batch_sampler'] = BatchSampler(
            batch_sampler, batch_size, drop_last=True)
    else:
        dataloader_kwargs['batch_size'] = batch_size
        dataloader_kwargs['shuffle'] = shuffle

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    return dataloader
