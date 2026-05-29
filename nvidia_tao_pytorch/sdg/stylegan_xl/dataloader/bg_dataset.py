# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigDatasetGAN Dataset"""

import os
import torch
import numpy as np
from PIL import Image


class LabelDataset(torch.utils.data.Dataset):
    """A custom dataset for loading grayscale images with corresponding binary masks.

    This dataset assumes that each image file is stored in a specified directory
    (`data_dir`) and is named in a format that includes a unique identifier in the
    filename (e.g., 'seed[XXXX].png'). Each image file is expected to represent a binary
    mask, where pixel values are either 0 or 255.

    Attributes:
        data_dir (str): Directory containing image files with binary masks.
        label_list (List[str]): Sorted list of filenames within `data_dir`.
    """

    def __init__(self, data_dir):
        """Initialize the dataset by setting the data directory and loading file names.

        Args:
            data_dir (str): Path to the directory containing binary mask images.
        """
        super().__init__()
        self.data_dir = data_dir
        self.label_list = sorted(os.listdir(self.data_dir))

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            int: Total number of image files in the directory.
        """
        return len(self.label_list)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding binary mask from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple:
                - int: A unique integer identifier extracted from the image filename (seed).
                - torch.Tensor: A 2D tensor representing the binary mask of the image,
                  with pixel values 0 (background) and 1 (foreground).

        Example:
            For an image file named 'seed[0010].png', this function will return
            the integer seed 10 and the binary mask as a tensor.
        """
        label_p = os.path.join(
            self.data_dir, self.label_list[idx]
        )

        img = Image.open(label_p).convert('L')
        label_np = np.array(img.getdata()).reshape((img.size)) // 255

        label_tensor = torch.tensor(label_np, dtype=torch.long)
        seed = int(os.path.basename(label_p).split('.')[0][4:])  # seed[XXXX].png

        return (seed, label_tensor)
