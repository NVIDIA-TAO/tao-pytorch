
"""Referring Expression Segmentation for inference"""

import os.path as osp
import json
from typing import Any, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.odvg import clean_caption


class RESPredictDataset(Dataset):
    """Referring Expression Segmentation Dataset for inference."""

    def __init__(self, image_dir: str, json_file: str, transforms: Optional[Any] = None):
        """Initialize dataset for inference.

        Args:
            image_dir (str): Directory containing images.
            json_file (str): JSON file with annotations (list of dicts).
            transforms (callable, optional): Transformations to apply.
        """
        self.image_dir = image_dir
        self.transforms = transforms

        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.cap_lists = None
        self.captions = None

    def _load_image(self, img_path: str) -> Tuple[Image.Image, str]:
        """Load image from path.

        Args:
            img_path (str): Image filename.

        Returns:
            Tuple[Image.Image, str]: Loaded image and full path.
        """
        full_path = osp.join(self.image_dir, img_path)
        return Image.open(full_path).convert("RGB"), full_path

    def __getitem__(self, index: int) -> Tuple[Any, dict, str]:
        """Get processed dataset sample for inference.

        Args:
            index (int): Dataset index.

        Returns:
            Tuple[image, target_dict, image_path].
        """
        instance = self.data[index]
        image, path = self._load_image(instance['image_path'])
        width, height = image.size

        target = {
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "caption": clean_caption(instance['expression']),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target, path

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
