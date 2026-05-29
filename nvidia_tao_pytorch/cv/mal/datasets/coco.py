# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""COCO dataset."""

from nvidia_tao_pytorch.cv.mal.datasets.voc import InstSegVOC, BoxLabelVOC, InstSegVOCwithBoxInput


class BoxLabelCOCO(BoxLabelVOC):
    """Dataset to load COCO box labels."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}


class InstSegCOCO(InstSegVOC):
    """Dataset to load COCO instance segmentation labels."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}


class InstSegCOCOwithBoxInput(InstSegVOCwithBoxInput):
    """Dataset to load COCO labels with only box input."""

    def get_category_mapping(self):
        """Category mapping."""
        categories = self.coco.dataset['categories']
        self.cat_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(categories)}
