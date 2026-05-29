#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Random crop data."""
import cv2
import numpy as np


class EastRandomCropData():
    """Random crop images."""

    def __init__(self, size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, require_original_image=False, keep_ratio=True):
        """Initialize.

        Args:
            size (set): The cropped target size.
            max_tries (int): The max times to try to crop since the cropped area
                is too small or failed to crop.
            min_crop_side_ratio (float): The minimum ratio to crop in terms of each line.
            require_original_image: Whether require original image.
            keep_ratio (bool) : Whether to keep ratio.
        """
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """Random crop images and ensure the text is not cropped"""
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        # Get the bbox/polygon whose tag are not ignored
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        # Calculate the crop area
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        # Crop image, keep aspect ratio
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
            padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))
        # Crop texts
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        # Calculate the new coordinates
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data

    def is_poly_outside_rect(self, poly, x, y, w, h):
        """Check if the text is outside the cropped area."""
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        """Generate the regions which can be split."""
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        """Select two split lines during one region."""
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        """Select two split lines during two regions."""
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        """Crop area."""
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for _ in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h
