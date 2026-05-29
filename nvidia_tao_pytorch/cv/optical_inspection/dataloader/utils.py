# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChangeNet Data Module utilities"""
import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


def to_tensor_and_norm(imgs, labels, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Convert a list of images to PyTorch tensors and normalize them.

    Args:
        imgs (list): A list of PIL images to be converted to PyTorch tensors and normalized.
        labels (list): A list of label images as numpy arrays.
        mean (list): The mean values for normalization (default is [0.5, 0.5, 0.5]).
        std (list): The standard deviation values for normalization (default is [0.5, 0.5, 0.5]).

    Returns:
        tuple: A tuple containing two lists - the first list contains the converted and normalized image tensors,
               and the second list contains the label tensors as PyTorch tensors.
    """
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=mean, std=std)
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:
    """
    Class for applying data augmentation to images.

    Args:
        img_size (int): The size of the images after resizing.
        with_random_hflip (bool, optional): Apply random horizontal flip.
        with_random_vflip (bool, optional): Apply random vertical flip.
        with_random_rot (bool, optional): Apply random rotation (90, 180, or 270 degrees).
        with_random_crop (bool, optional): Apply random resized crop.
        with_scale_random_crop (bool, optional): Apply scale random crop.
        with_random_blur (bool, optional): Apply random Gaussian blur.
        random_color_tf (bool, optional): Apply random color jitter.
        mean (list, optional): Mean values for normalization (default is [0.5, 0.5, 0.5]).
        std (list, optional): Standard deviation values for normalization (default is [0.5, 0.5, 0.5]).

    """

    def __init__(
            self,
            img_size,
            random_flip=None,
            random_rotate=None,
            random_color=None,
            # with_scale_random_crop=None,
            with_random_crop=False,
            with_random_blur=False,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        """Initialize"""
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_color = random_color
        self.with_random_crop = with_random_crop
        # self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.mean = mean
        self.std = std

    def transform(self, imgs, imgs1, to_tensor=True):
        """
        Apply a sequence of data augmentation routines to a list of images and labels.

        Args:
            imgs (list): A list of PIL images for test images to apply data augmentation to.
            imgs1 (list): A list of PIL images for golden images to apply similar data augmentation as imgs.
            labels (list): A list of label images as numpy arrays.
            to_tensor (bool, optional): Convert images to PyTorch tensors (default is True).

        Returns:
            tuple: A tuple containing augmented images and labels.

        Notes:
            This function performs a series of image augmentation operations on the input images and labels. The following steps are performed in sequence:

            1. Resize images to the specified image size (if not using dynamic resizing).
            2. Resize labels to match the image size.
            3. Apply horizontal and vertical flips based on the given probabilities.
            4. Apply random rotation based on the given probability and angle list.
            5. Apply random resized crop if specified.
            6. Apply scale and random crop transformations if enabled.
            7. Apply random Gaussian blur if specified.
            8. Apply random color jitter transformations.
            9. Convert images to PyTorch tensors and normalize them.

        """
        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size[0], self.img_size[1]):
                imgs = [TF.resize(img, [self.img_size[0], self.img_size[1]], interpolation=3)
                        for img in imgs]
            if imgs1[0].size != (self.img_size[0], self.img_size[1]):
                imgs1 = [TF.resize(img, [self.img_size[0], self.img_size[1]], interpolation=3)
                         for img in imgs1]
        else:
            self.img_size = imgs[0].size[0]

        if self.random_flip is not None and self.random_flip.enable:
            hflip_probability = 1 - self.random_flip.hflip_probability
            vflip_probability = 1 - self.random_flip.vflip_probability

            if random.random() > hflip_probability:
                imgs = [TF.hflip(img) for img in imgs]
                imgs1 = [TF.hflip(img) for img in imgs1]

            if random.random() > vflip_probability:
                imgs = [TF.vflip(img) for img in imgs]
                imgs1 = [TF.vflip(img) for img in imgs1]

        if self.random_rotate is not None and self.random_rotate.enable:
            random_base = 1 - self.random_rotate.rotate_probability

            if random.random() > random_base:
                angles = self.random_rotate.angle_list
                index = random.randint(0, 2)
                angle = angles[index]
                imgs = [TF.rotate(img, angle) for img in imgs]
                imgs1 = [TF.rotate(img, angle) for img in imgs1]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=(self.img_size[0], self.img_size[1])). \
                get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size[0], self.img_size[1]),
                                    interpolation=Image.BICUBIC)
                    for img in imgs]

            imgs1 = [TF.resized_crop(img, i, j, h, w,
                                     size=(self.img_size[0], self.img_size[1]),
                                     interpolation=Image.BICUBIC) for img in imgs1]

        # if self.with_scale_random_crop is not None and self.with_scale_random_crop.enable:
        #     # rescale
        #     scale_range = self.with_scale_random_crop.scale_range
        #     target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

        #     imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
        #     imgs1 = [pil_rescale(img, target_scale, order=3) for img in imgs1]
        #     # crop
        #     image_size = imgs[0].size  # h, w
        #     box = get_random_crop_box(image_size=image_size, crop_size=self.img_size)
        #     imgs = [pil_crop(img, box, crop_size=self.img_size, default_value=0)
        #             for img in imgs]
        #     imgs1 = [pil_crop(img, box, crop_size=self.img_size, default_value=0)
        #               for img in imgs1]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]
            imgs1 = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                     for img in imgs1]

        if self.random_color is not None and self.random_color.enable:
            if random.random() > self.random_color.color_probability:
                color_jitter = transforms.ColorJitter(brightness=self.random_color.brightness,
                                                      contrast=self.random_color.contrast,
                                                      saturation=self.random_color.saturation,
                                                      hue=self.random_color.hue)
                imgs_tf, imgs1_tf = [], []
                brightness = random.uniform(*color_jitter.brightness)
                contrast = random.uniform(*color_jitter.contrast)
                saturation = random.uniform(*color_jitter.saturation)
                hue = random.uniform(*color_jitter.hue)
                k = len(imgs1) // len(imgs)
                for i in range(len(imgs)):
                    tf = transforms.ColorJitter(
                        (brightness, brightness),
                        (contrast, contrast),
                        (saturation, saturation),
                        (hue, hue))
                    imgs_tf.append(tf(imgs[i]))
                    imgs1_tf.extend([tf(imgs1[i * k + j]) for j in range(k)])
                imgs, imgs1 = imgs_tf, imgs1_tf

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs1 = [TF.to_tensor(img) for img in imgs1]
            imgs = [TF.normalize(img, mean=self.mean, std=self.std)
                    for img in imgs]
            imgs1 = [TF.normalize(img, mean=self.mean, std=self.std)
                     for img in imgs1]

        return imgs, imgs1


def pil_crop(image, box, cropsize, default_value):
    """
    Crop an image using the specified box coordinates.

    Args:
        image (PIL.Image.Image): The input image to be cropped.
        box: A tuple containing the box coordinates
        cropsize (int): The size of the cropped image.
        default_value (int): The default value to fill the cropped region.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    assert isinstance(image, Image.Image), "Input image must be a PIL.Image.Image object."
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(image_size, crop_size):
    """
    Generate random crop box coordinates for cropping an image.

    Args:
        image_size (Tuple[int, int]): The size of the original image (height, width).
        crop_size (int): The desired size of the crop.

    Returns:
        Tuple: A tuple containing the crop box coordinates.
    """
    h, w = image_size
    ch = min(crop_size, h)
    cw = min(crop_size, w)

    w_space = w - crop_size
    h_space = h - crop_size

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be rescaled.
        scale (float): The scaling factor.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The rescaled image.
    """
    assert isinstance(img, Image.Image), "Input image must be a PIL.Image.Image object."
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be resized.
        size (Tuple[int, int]): The target size (height, width) of the resized image.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The resized image.
    """
    assert isinstance(img, Image.Image), "Input image must be a PIL.Image.Image object."
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
