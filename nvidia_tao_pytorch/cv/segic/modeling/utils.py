# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General utility functions for segmentation model."""

import cv2
import numpy as np


def overlay_masks(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Args:
        image(np.ndarray): Training image. np.ndarray, (C, H, W)
        mask(np.ndarray): Segmentation mask. (H, W)
        color(Tuple): Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha(float): Segmentation mask's transparency. float = 0.5
        resize(Tuple): Resize the image if not None.

    Returns:
        image_combined(np.ndarray): The combined image.

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
