#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make shrink map."""
import numpy as np
import cv2


def shrink_polygon_py(polygon, shrink_ratio):
    """Shrink the polygon to 1/shrink_ratio.

    Args:
        polygon (list): The original polygon.
        shrink_ratio (float): The shrink_raio.

    Returns:
        shrinked (list): The shrinked polygon.
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    """Shrink polygon pyclipper.

    Args:
        polygon (list): The original polygon.
        shrink_ratio (float): The shrink_raio.

    Returns:
        shrinked (list): The shrinked polygon.
    """
    from shapely.geometry import Polygon
    import pyclipper
    # Generate polygon object
    polygon_shape = Polygon(polygon)
    # The distance during shrinking
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(p) for p in polygon]
    padding = pyclipper.PyclipperOffset()  # pylint: disable=I1101
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # pylint: disable=I1101
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class MakeShrinkMap():
    """Generate probability map."""

    def __init__(self, min_text_size=8, shrink_ratio=0.4, shrink_type='pyclipper'):
        """Initialize.

        Args:
            min_text_size (int): The minimum text size.
            shrink_ratio (float): The shrink ratio of polygon.
        """
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data: dict) -> dict:
        """Generate shrinked polygon."""
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        """Align the coordinate order of the polygons, ignore the polygon whose area is zero.

        Args:
            polygons (list): The polygons in text data.
            ignore_tags: (list): The tags which are marked ignored.
            h (int): The height of image.
            w (int): The width of image.

        Returns:
            polygons: The new polygons.
            ignore_tags: The new ignore_tags.
        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        # Clip the polygon coordinates inside the image
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            # Calculate the area of polygon
            area = self.polygon_area(polygons[i])
            # Ignore the polygon whose area is small
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        """Calculate the area of polygon."""
        return cv2.contourArea(polygon)
