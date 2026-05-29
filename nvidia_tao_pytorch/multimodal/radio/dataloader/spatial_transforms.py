# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Spatial (geometric) transforms ported from EVFM.

Includes patch-aligned cropping for Vision Transformers and
other geometric augmentations. Compatible with TAO's PIL-based pipeline.
"""

import random
import math
from PIL import Image
import torchvision.transforms.functional as TF


def get_patch_aligned_random_crop_params(img_width, img_height, target_size,
                                         patch_size, scale=(0.08, 1.0)):
    """
    Generate random crop parameters aligned to patch boundaries.

    Critical for Vision Transformers; misaligned crops can cause issues
    with positional embeddings.

    Args:
        img_width: Width of input image.
        img_height: Height of input image.
        target_size: Target size after resize (square output).
        patch_size: Patch size used by ViT (e.g., 14 for ViT-L/14).
        scale: Range for random crop area as fraction of image (min, max).

    Returns:
        tuple: (i, j, h, w) crop parameters for TF.crop.
    """
    area = img_width * img_height

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        log_ratio = random.uniform(-0.3, 0.3)
        aspect_ratio = math.exp(log_ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= img_width and h <= img_height:
            crop_h = (h // patch_size) * patch_size
            crop_w = (w // patch_size) * patch_size

            if (0 < crop_h <= img_height and 0 < crop_w <= img_width):
                i = random.randint(0, img_height - crop_h)
                j = random.randint(0, img_width - crop_w)
                return i, j, crop_h, crop_w

    # Fallback: center crop
    w = min(img_width, img_height)
    h = w
    i = (img_height - h) // 2
    j = (img_width - w) // 2
    return i, j, h, w


def patch_aligned_random_resized_crop(img, size, patch_size, scale=(0.08, 1.0)):
    """
    Random resized crop with patch alignment for Vision Transformers.

    Args:
        img: PIL.Image input.
        size: Target output size (square).
        patch_size: Patch size for ViT alignment.
        scale: Scale range for random crop.

    Returns:
        PIL.Image: Cropped and resized image.
    """
    width, height = img.size
    i, j, h, w = get_patch_aligned_random_crop_params(
        width, height, size, patch_size, scale
    )
    img = TF.crop(img, i, j, h, w)
    img = TF.resize(img, [size, size], interpolation=Image.BILINEAR)
    return img


def continuous_random_rotation(img, angle_range=(-15, 15)):
    """
    Random rotation with continuous angles.

    Unlike discrete angle lists, samples from a continuous distribution.

    Args:
        img: PIL.Image input.
        angle_range: Range of angles in degrees (min, max).

    Returns:
        PIL.Image: Rotated image.
    """
    angle = random.uniform(*angle_range)
    return TF.rotate(img, angle, interpolation=Image.BILINEAR)


def random_perspective_transform(img, distortion_scale=0.1, probability=0.5):
    """
    Random perspective transformation.

    Args:
        img: PIL.Image input.
        distortion_scale: Distortion amount (0-1). EVFM uses ~0.01-0.05.
        probability: Probability of applying transform.

    Returns:
        PIL.Image: Transformed image.
    """
    if random.random() > probability:
        return img

    width, height = img.size
    half_h = height // 2
    half_w = width // 2

    def _off(mag):
        return int(random.uniform(-distortion_scale, distortion_scale) * mag)

    topleft = [_off(half_w), _off(half_h)]
    topright = [width + _off(half_w), _off(half_h)]
    botright = [width + _off(half_w), height + _off(half_h)]
    botleft = [_off(half_w), height + _off(half_h)]

    startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
    endpoints = [topleft, topright, botright, botleft]

    return TF.perspective(
        img, startpoints, endpoints, interpolation=Image.BILINEAR
    )
