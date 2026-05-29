# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset for Metric Learning Recognition inference."""

import os

import torch
from torchvision import transforms

from nvidia_tao_pytorch.cv.ml_recog.dataloader.datasets.image_datasets import MetricLearnImageFolder
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import read_image

VALID_IMAGE_EXT = ['.jpg', '.jpeg', '.png']


class InferenceImageFolder(MetricLearnImageFolder):
    """This class inherits from :class:`MetricLearnImageFolder`. It prepares the
    data loader from the a classification dataset folder.

    In __getitem__ instead of returning image tensor and the target, it returns
    image tensor and the image path.
    """

    def __getitem__(self, index):
        """Retrieves the (image, image path) from data index.

        Args:
            index (int): Index of the data to retrieve

        Returns:
            sample (torch.Tensor): the image tensor
            path (String): the directory of the image file
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)

        return sample, path


class InferenceImageDataset(torch.utils.data.Dataset):
    """This class inherits from :class:`torch.utils.data.Dataset`. It prepares
    data loader from an image folder.

    In __getitem__, it returns image tensor and the image path.
    """

    def __init__(self, image_folder, transform=None):
        """Initiates Dataset for inference image folder input.

        Args:
            image_folder(String): path of image folder
            transform(torchvision.transorms.Compose): the composed transforms
        """
        self.paths = [os.path.join(image_folder, imgname)
                      for imgname in sorted(os.listdir(image_folder))
                      if os.path.splitext(imgname)[1].lower()
                      in VALID_IMAGE_EXT]

        self.transform = transform

    def __len__(self):
        """Gets the length of datasets"""
        return len(self.paths)

    def __getitem__(self, index):
        """Retrieves the (image, image path) from data index."""
        image = read_image(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, self.paths[index]
