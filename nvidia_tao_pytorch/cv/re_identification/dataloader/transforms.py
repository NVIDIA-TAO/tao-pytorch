# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builds Transforms based on training and validation."""

import torchvision.transforms as T
from timm.data import random_erasing
import random
import math


def build_transforms(model_config, is_train=True):
    """Return transforms for images based on the training and validation context.

    This function generates different sets of transformation operations for training and validation processes.
    For training, the operations include resizing, horizontal flip, padding, random crop, normalization and random erasing.
    For validation, the operations include only resizing and normalization.

    Args:
        model_config (dict): A dictionary containing the model and dataset configurations.
        is_train (bool): Indicates if the transformations are for training. If False, transformations are for validation.
                         Defaults to True.

    Returns:
        torchvision.transforms.Compose: A compose object that contains the list of transformations to be applied on an image.
    """
    normalize_transform = T.Normalize(mean=model_config['dataset']['pixel_mean'], std=model_config['dataset']['pixel_std'])
    if is_train:
        if "resnet" in model_config["model"]["backbone"]:
            transform = T.Compose([
                T.Resize([model_config['model']['input_height'], model_config['model']['input_width']]),
                T.RandomHorizontalFlip(p=model_config['dataset']['prob']),
                T.Pad(model_config['dataset']['padding']),
                T.RandomCrop([model_config['model']['input_height'], model_config['model']['input_width']]),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=model_config['dataset']['re_prob'], mean=model_config['dataset']['pixel_mean'])
            ])
        elif "swin" in model_config["model"]["backbone"]:
            transform = T.Compose([
                T.Resize([model_config['model']['input_height'], model_config['model']['input_width']], interpolation=3),
                T.RandomHorizontalFlip(p=model_config['dataset']['prob']),
                T.Pad(model_config['dataset']['padding']),
                T.RandomCrop([model_config['model']['input_height'], model_config['model']['input_width']]),
                T.ToTensor(),
                normalize_transform,
                random_erasing.RandomErasing(probability=model_config['dataset']['re_prob'], mode='pixel', max_count=1, device='cpu'),
            ])
    else:
        transform = T.Compose([
            T.Resize([model_config['model']['input_height'], model_config['model']['input_width']]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


class RandomErasing(object):
    """A data augmentation technique that randomly selects a rectangle region in an image and erases its pixels.

    This technique can help in the training process by introducing a form of noise. The target area is computed based on
    a set of pre-defined probability and aspect ratio parameters. The pixel values of the erased region are replaced
    by the mean pixel values of the image.

    Args:
        probability (float, optional): The probability that the random erasing operation will be performed. Defaults to 0.5.
        sl (float, optional): The lower bound of the range from which the area of the erase region is randomly sampled. Defaults to 0.02.
        sh (float, optional): The upper bound of the range from which the area of the erase region is randomly sampled. Defaults to 0.4.
        r1 (float, optional): The lower bound of the range from which the aspect ratio of the erase region is randomly sampled. Defaults to 0.3.
        mean (tuple, optional): The pixel mean values for each channel. Defaults to (0.4914, 0.4822, 0.4465).

    Methods:
        __call__(img): Performs the random erasing operation on the input image tensor.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        """
        Constructor to initialize random erasing technique for augmentation.

        Args:
            probability (float): Configuration file.
            sl (float): Lower interval of uniform distribution for target area.
            sh (float): Higher interval of uniform distribution for target area.
            r1 (float): Lower interval of uniform distribution for aspect ratio.
            mean (tuple): Pixel mean in 3 channels for normalization.
        Returns:
            transform (transforms): Image Transform for traning, testing & validation data.

        """
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """Perform the random erasing operation on the input image tensor.

        The function computes a target erase region on the image based on the initialized parameters.
        If the region is valid, the pixel values in this region will be replaced by the initialized mean values.

        Args:
            img (torch.Tensor): The input image tensor, expected in (C, H, W) format.

        Returns:
            torch.Tensor: The image tensor after random erasing operation.
        """
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
