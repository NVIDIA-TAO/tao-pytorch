# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data augmentation."""
import numpy as np
import random
import cv2
from fvcore.transforms.transform import Transform


class RandomRotation(Transform):
    """Random rotation transformation."""

    def __init__(self, orig_size, angle_range=(-10, 10)):
        super().__init__()
        self.angle = random.uniform(angle_range[0], angle_range[1])
        h, w = orig_size
        center = (w / 2, h / 2)
        self.transform_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)

    def apply_coords(self, coords):
        """Apply rotation to coordinates."""
        return coords

    def apply_image(self, img):
        """Apply rotation to image."""
        h, w = img.shape[:2]
        return cv2.warpAffine(
            img,
            self.transform_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def apply_segmentation(self, segmentation):
        """Apply rotation to segmentation."""
        h, w = segmentation.shape[:2]
        return cv2.warpAffine(
            segmentation,
            self.transform_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )


class GaussianBlur(Transform):
    """Gaussian blur transformation."""

    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = random.randrange(
            kernel_size_range[0], kernel_size_range[1] + 1, 2
        )
        self.sigma = random.uniform(sigma_range[0], sigma_range[1])

    def apply_coords(self, coords):
        """Apply blur to coordinates."""
        return coords

    def apply_image(self, img):
        """Apply blur to image."""
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)

    def apply_segmentation(self, segmentation):
        """Apply blur to segmentation."""
        return segmentation


class RandomErasing(Transform):
    """Random erasing transformation."""

    def __init__(self, orig_size, scale=(0.02, 0.2), ratio=(0.3, 3.3), seg_value=255):
        super().__init__()
        self.rect = None
        self.seg_value = seg_value
        img_h, img_w = orig_size
        area = img_h * img_w
        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])
            h = int(round(np.sqrt(erase_area * aspect_ratio)))
            w = int(round(np.sqrt(erase_area / aspect_ratio)))
            if w < img_w and h < img_h:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                self.rect = (x1, y1, w, h)
                break

    def apply_coords(self, coords):
        """Apply erasing to coordinates."""
        return coords

    def apply_image(self, img):
        """Apply erasing to image."""
        if self.rect is None:
            return img
        x1, y1, w, h = self.rect
        img[y1:y1 + h, x1:x1 + w] = img.mean(axis=(0, 1))
        return img

    def apply_segmentation(self, segmentation):
        """Apply erasing to segmentation."""
        if self.rect is None:
            return segmentation
        x1, y1, w, h = self.rect
        segmentation[y1:y1 + h, x1:x1 + w] = self.seg_value
        return segmentation
