# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Colorization Utilities for BigDatasetGAN."""

import numpy as np


def color_map(N, normalized=False):
    """Generate a color map with N distinct colors.

    Args:
        N (int): Number of colors to generate.
        normalized (bool, optional): Whether to normalize the colors to [0, 1]. Default is False.

    Returns:
        np.ndarray: An array of shape (N, 3) containing the generated colors.
    """
    def bitget(byteval, idx):
        """Get the bit value at the specified index.

        Args:
            byteval (int): The byte value.
            idx (int): The index of the bit to get.

        Returns:
            int: The bit value (0 or 1).
        """
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class VOCColorize(object):
    """Class to colorize grayscale images using a predefined color map."""

    def __init__(self, n):
        """Initialize the VOCColorize object.

        Args:
            n (int): Number of colors in the color map.
        """
        self.cmap = color_map(n)

    def __call__(self, gray_image):
        """Colorize a grayscale image.

        Args:
            gray_image (np.ndarray): Grayscale image to colorize.

        Returns:
            np.ndarray: Colorized image.
        """
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
