# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data augmentation."""
import sys
import numpy as np
import random
import cv2
from fvcore.transforms.transform import (
    CropTransform, HFlipTransform, NoOpTransform, Transform)


def apply_transform(img, mask, T):
    """Apply tranformation to image and mask."""
    img_T = T.apply_image(img)
    mask_T = T.apply_segmentation(mask)

    return img_T, mask_T


def get_output_shape(oldh: int,
                     oldw: int,
                     short_edge_length: int,
                     max_size: int):
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class RandomHorizontalFlip(Transform):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, orig_size, prob=0.5):
        """Init.
        Args:
            prob (float): probability of flip.
        """
        super().__init__()
        self.orig_size = orig_size
        self.prob = prob
        self.hflip_transform = self._get_transform()

    def _get_transform(self):
        _, w = self.orig_size
        if random.random() < self.prob:
            return HFlipTransform(w)
        return NoOpTransform()

    def apply_coords(self, coords):
        """Apply to coords."""
        return self.hflip_transform.apply_coords(coords)

    def apply_image(self, img):
        """Apply to image."""
        return self.hflip_transform.apply_image(img)


class RandomCrop(Transform):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, orig_size, crop_size, segm=None, max_ratio=0.6, ignored_category=0):
        """Init.
        Args:
            orig_size (tuple[int, int]): two ints.
            crop_size (tuple[float, float]): two floats.
            max_ratio (float): the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        super().__init__()
        self.orig_size = orig_size
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignored_category = ignored_category
        self.crop_transform = self._get_transform(segm)

    def _get_transform(self, segm):
        h, w = self.orig_size
        croph, cropw = (min(self.crop_size[0], h), min(self.crop_size[1], w))
        if self.max_ratio >= 1.0:
            h, w = self.orig_size
            croph, cropw = (min(self.crop_size[0], h), min(self.crop_size[1], w))
            assert h >= croph and w >= cropw, f"Shape computation in {self} has bugs."
            h0 = np.random.randint(h - croph + 1)
            w0 = np.random.randint(w - cropw + 1)
        else:
            for _ in range(10):
                h0 = np.random.randint(h - croph + 1)
                w0 = np.random.randint(w - cropw + 1)
                sem_seg_temp = segm[h0: h0 + croph, w0: w0 + cropw]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.max_ratio:
                    break
        return CropTransform(w0, h0, cropw, croph)

    def apply_coords(self, coords):
        """Apply to coords."""
        return self.crop_transform.apply_coords(coords)

    def apply_image(self, img):
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        return self.crop_transform.apply_image(img)

    def apply_polygons(self, polygons):
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        return self.crop_transform.apply_polygons(polygons)


class ResizeShortestEdge(Transform):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        orig_size,
        short_edge_length,
        max_size=sys.maxsize,
        interp=cv2.INTER_LINEAR,
        prob=1.0
    ):
        """Init."""
        super().__init__()
        self.orig_size = orig_size
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp
        self.prob = prob  # TODO(@yuw): enable?
        self._get_output_shape()

    def _get_output_shape(self):
        h, w = self.orig_size
        self.new_size = None
        size = np.random.choice(self.short_edge_length)
        if size != 0:
            hh, ww = get_output_shape(h, w, size, self.max_size)
            self.new_size = (ww, hh)

    def apply_coords(self, coords):
        """Apply to coords.."""
        return coords

    def apply_image(self, img, interp=None):
        """Apply to image."""
        return cv2.resize(img, self.new_size, interpolation=self.interp)

    def apply_segmentation(self, segmentation):
        """Apply to segmentation mask."""
        return cv2.resize(segmentation, self.new_size, interpolation=cv2.INTER_NEAREST)


class ColorAugSSDTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        """Init."""
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        """Apply to coords."""
        return coords

    def apply_segmentation(self, segmentation):
        """Apply to segmentation."""
        return segmentation

    def apply_image(self, img, interp=None):
        """Apply to image."""
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        """Convert."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Random brightness."""
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        """Random contrast."""
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        """Random saturation."""
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Random hue."""
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img
