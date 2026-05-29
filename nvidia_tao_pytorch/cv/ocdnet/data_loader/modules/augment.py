#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augment module."""
import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    """Random Noise class."""

    def __init__(self, random_rate):
        """Initialize."""
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """Add random noise."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        data['img'] = (random_noise(data['img'], mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data


class RandomScale:
    """Random Scale class."""

    def __init__(self, scales, random_rate):
        """Initialize."""
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """Add random scale."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data


class RandomRotateImgBox:
    """Random Rotate Image Box."""

    def __init__(self, degrees, random_rate, same_size=False):
        """Initialize."""
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, np.ndarray, tuple)):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise TypeError('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """Add random rotate image box."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        # rotate
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            rangle = np.deg2rad(angle)
            # calculate w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # getRotationMatrix2D
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # offset between original center point and new center point
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # update mat
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # warpaffine
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data['img'] = rot_img
        data['text_polys'] = np.array(rot_text_polys)
        return data


class RandomResize:
    """Random Resize."""

    def __init__(self, size, random_rate, keep_ratio=False):
        """Initialize."""
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, (list, np.ndarray, tuple)):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise TypeError('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """Add random resize."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['text_polys'] = text_polys
        return data


class Resize2D:
    """Resize 2D."""

    def __init__(self, short_size, resize_text_polys=True):
        """Initialize."""
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """Resize images and texts"""
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        if isinstance(self.short_size, (list, tuple)):
            target_width = self.short_size[0]
            target_height = self.short_size[1]
            scale = (target_width / w, target_height / h)
            im = cv2.resize(im, dsize=None, fx=scale[0], fy=scale[1])
            if self.resize_text_polys:
                text_polys[:, :, 0] *= scale[0]
                text_polys[:, :, 1] *= scale[1]
        else:
            short_edge = min(h, w)
            if short_edge < self.short_size:
                scale = self.short_size / short_edge
                im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
                scale = (scale, scale)
                if self.resize_text_polys:
                    text_polys[:, :, 0] *= scale[0]
                    text_polys[:, :, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data


class HorizontalFlip:
    """Horizontal Flip class."""

    def __init__(self, random_rate):
        """Initialize."""
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """Add horizontal flip."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        __, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class VerticalFlip:
    """Vertical Flip class."""

    def __init__(self, random_rate):
        """Initialize."""
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """Add Vertical flip."""
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, __, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data
