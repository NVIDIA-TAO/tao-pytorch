# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion TAO3D Dataset module. """

from typing import Callable, List, Union, Tuple
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import Box3DMode

import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


@DATASETS.register_module()
class TAO3DDataset(Det3DDataset):
    """TAO3DDataset Dataset.

    This class serves as the API for experiments on the TAO3D Dataset [Single LiDAR + Single Camera]

    """

    METAINFO = {
        'classes': ('person',),
        'palette': [(0, 230, 0),]
    }

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = {'use_camera': False, 'use_lidar': True},
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 default_cam_key: str = 'CAM0',
                 origin: Tuple[float, float, float] = (0.5, 0.5, 0.0),
                 **kwargs) -> None:
        """
        Initialize TAO3DDataset
        Args:
            ann_file (str): Path of annotation file.
            pipeline (list[dict]): Pipeline used for data processing.
                Defaults to [].
            box_type_3d (str): Type of 3D box of this dataset.
                Based on the `box_type_3d`, the dataset will encapsulate the box
                to its original format then converted them to `box_type_3d`.
                Defaults to 'LiDAR' in this dataset. Available options includes:
                - 'LiDAR': Box in LiDAR coordinates.
                - 'Depth': Box in depth coordinates, usually for indoor dataset.
                - 'Camera': Box in camera coordinates.
            load_type (str): Type of loading mode. Defaults to 'frame_based'.
                - 'frame_based': Load all of the instances in the frame.
            modality (dict): Modality to specify the sensor data used as input.
                Defaults to dict(use_camera=False, use_lidar=True).
            filter_empty_gt (bool): Whether to filter the data with empty GT.
                If it's set to be True, the example with empty annotations after
                data pipeline will be dropped and a random example will be chosen
                in `__getitem__`. Defaults to True.
            test_mode (bool): Whether the dataset is in test mode.
                Defaults to False.
        """
        #  Support only frame-based load for TAO
        self.load_type = load_type
        self.origin = origin
        assert load_type in ('frame_based'), "load_type is only supported for \
                             frame_based"

        super().__init__(
            ann_file=ann_file,
            modality=modality,
            default_cam_key=default_cam_key,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_eval_anns=load_eval_anns,
            **kwargs)

        assert self.modality is not None, "input_modality must be set"
        assert box_type_3d.lower() in ('lidar', 'camera'), \
               "box_type_3d needs to be set to lidar or camera"

        # override box_type with TAO3D data type
        self.default_cam_key = default_cam_key
        if box_type_3d.lower() == 'lidar':
            box_type_3d = tao_structures.TAOLiDARInstance3DBoxes
            box_mode_3d = Box3DMode.LIDAR
        elif box_type_3d.lower() == 'camera':
            box_type_3d = tao_structures.TAOCameraInstance3DBoxes
            box_mode_3d = Box3DMode.CAM
        else:
            raise NotImplementedError(f'box_type_3d type {box_type_3d} is not supported yet')
        self.box_type_3d = box_type_3d
        self.box_mode_3d = box_mode_3d
        self.metainfo['box_type_3d'] = self.box_type_3d

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            ann_info = {}
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        ann_info = self._remove_dontcare(ann_info)

        # This dataset assumes that your 3D bounding box labels are in lidar coordinate
        if self.box_mode_3d == Box3DMode.CAM:
            # if box_mode is set to Camera, convert labels from lidar to camera coordainte
            lidar2cam = np.array(info['images'][self.default_cam_key]['lidar2cam'])
            gt_bboxes_3d = tao_structures.TAOLiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'], origin=self.origin).convert_to(self.box_mode_3d, np.transpose(lidar2cam), yaw_dim=2)
        else:
            # If box_mode is set to lidar, we keep the lidar coordinate labels.
            gt_bboxes_3d = tao_structures.TAOLiDARInstance3DBoxes(ann_info['gt_bboxes_3d'], origin=self.origin)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info
