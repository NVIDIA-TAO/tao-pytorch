# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVDINOv2 Dataset"""
import logging
from pathlib import Path
from typing import Iterable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DinoV2Dataset(Dataset):
    """Dataset for NVDINOv2 to manage and transform image data for training, with support for various image formats."""

    def __init__(
        self,
        *,
        root: Union[str, Path],
        transform: Optional[callable] = None,
        train: bool = True,
        extensions: Iterable[str] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
    ):
        """Initializes the dataset with the root directory, optional transformations, and valid image extensions.

        Args:
            root (Union[str, Path]): The root directory containing the image files
            transform (Optional[callable], optional): A transformation function to apply to the images. Required for data processing. Defaults to None.
            extensions (Iterable[str], optional): A list of valid image file extensions. Defaults include common image formats. Defaults to ( ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ).
        """
        self.root = Path(root)
        self.extensions = extensions
        self.transform = transform
        self.train = train

        assert self.transform is not None, "Transform must be specified."

        self.all_images = self._list_images()

        assert len(self.all_images) > 0, f"No images found in {self.root}."

    def _list_images(self):
        """Lists all image paths in the specified root directory.

        Returns:
            List: List of all image paths relative to the root directory.
        """
        return [
            str(f.relative_to(self.root))
            for f in self.root.rglob("*")
            if f.suffix.lower() in self.extensions
        ]

    def __len__(self):
        """Returns the number of images in the dataset.

        Returns:
            Int: The total count of images in the dataset.
        """
        return len(self.all_images)

    def _get_item_internal_(self, idx):
        """Retrieves and transforms an image at a given index.

        Args:
            idx (Int): Index of the image to retrieve.

        Returns:
            Dict: A dictionary containing global and local crops of the image.
        """
        img_path = self.root / self.all_images[idx]

        image = Image.open(img_path, mode="r").convert("RGB")
        images = self.transform(image)
        if self.train:
            return {
                "global_crops": images["global_crops"],
                "local_crops": images["local_crops"],
            }
        return {
            "images": images,
            "input_path": self.all_images[idx]
        }

    def __getitem__(self, idx):
        """Retrieves an item (image) at a given index with error handling.

        Args:
            idx (Int): Index of the image to retrieve.

        Returns:
            Dict: A dictionary containing global and local crops of the image.
        """
        try:
            return self._get_item_internal_(idx)
        except Exception as e:
            logger.error(f"Error retrieving image at {self.root / self.all_images[idx]}: {e}")
            raise
