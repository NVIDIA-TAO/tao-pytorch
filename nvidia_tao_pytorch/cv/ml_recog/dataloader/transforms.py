# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builds Transforms based on training and validation."""

import torchvision.transforms as T

from nvidia_tao_pytorch.cv.re_identification.dataloader.transforms import RandomErasing


def build_transforms(model_config, is_train=True):
    """Returns training or validation dataloader transforms. The transforms include
    transferring the Image file to torch.Tensor, random crop, random horizontal flip,
    color jitter, gaussian blur, normalization and random erasing.

    Whether to use random rotation, color jitter and gaussian blur is specified
    in the config. The mean and std of normalization is specified in the config.
    The size of random crop is specified in the config. The probability of random
    horizontal flip is specified in the config. The kernel size and sigma of
    gaussian blur is specified in the config. The brightness, contrast, saturation
    and hue of color jitter is specified in the config. The probability of random
    erasing is specified in the config.

    Args:
        model_config (DictConfig): Configuration file
        is_train (Boolean): True for training, False for Testing & Validation

    Returns:
        transform (torchvision.transforms.Compose): Image transform for traning, testing & validation data

    """
    normalize_transform = T.Normalize(mean=model_config['dataset']['pixel_mean'],
                                      std=model_config['dataset']['pixel_std'])
    input_size = (model_config['model']['input_width'], model_config['model']['input_height'])

    if is_train:
        transforms = [
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(p=model_config['dataset']['prob']),
        ]

        if model_config['dataset']['random_rotation']:
            transforms.append(T.RandomRotation(degrees=(0, 180), expand=False))

        if model_config['dataset']['color_augmentation']['enabled'] is True:
            color_aug_params = model_config['dataset']['color_augmentation']
            transforms.append(T.ColorJitter(brightness=color_aug_params['brightness'],
                                            contrast=color_aug_params['contrast'],
                                            saturation=color_aug_params['saturation'],
                                            hue=color_aug_params['hue']))

        if model_config['dataset']['gaussian_blur']['enabled'] is True:
            gauss_params = model_config['dataset']['gaussian_blur']
            transforms.append(T.GaussianBlur(kernel_size=list(gauss_params['kernel']),
                                             sigma=gauss_params['sigma']))

        transforms += [
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=model_config['dataset']['re_prob'],
                          mean=model_config['dataset']['pixel_mean'])
        ]

        transform = T.Compose(transforms)
    else:
        w, h = input_size

        # takes 1/1.14 of the image after cropping
        init_resize = (int(w * 1.14), int(h * 1.14))
        transform = T.Compose([
            T.Resize(init_resize),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
