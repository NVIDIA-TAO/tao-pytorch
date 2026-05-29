# Original source taken from https://github.com/open-mmlab/mmsegmentation/mmseg/datasets/transforms/loading.py
#
# Copyright (c) OpenMMLab. All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom LoadAnnotations Module."""

import warnings

import numpy as np
import mmcv
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
import mmengine.fileio as fileio
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TAOLoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset."""

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
        input_type='rgb'
    ) -> None:
        """Constructor.

        Args:
            reduce_zero_label (bool, optional): Whether reduce all label value
                by 1. Usually used for datasets where 0 is background label.
                Defaults to None.
            imdecode_backend (str): The image decoding backend type. The backend
                argument for :func:``mmcv.imfrombytes``.
                See :fun:``mmcv.imfrombytes`` for details.
                Defaults to 'pillow'.
            backend_args (dict): Arguments to instantiate a file backend.
                See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
                for details. Defaults to None.
                Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
        """
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend
        self.input_type = input_type

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # @sean this is the difference between ours and the default
        if self.input_type == "grayscale":
            gt_semantic_seg = gt_semantic_seg / 255
            gt_semantic_seg = np.where(gt_semantic_seg > 0.5, 1, 0)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        """Returns object with params"""
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
