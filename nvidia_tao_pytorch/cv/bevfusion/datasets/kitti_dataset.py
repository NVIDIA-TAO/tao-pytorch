# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion kitti-person dataset module. """

from typing import Callable, List, Union, Tuple
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import Box3DMode

import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


@DATASETS.register_module(force=True)
class KittiPersonDataset(Det3DDataset):
    """KITTI-Person Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    """

    METAINFO = {
        'classes': ('person',),
        'palette': [
            (0, 230, 0),  # green
        ]
    }

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = {'use_camera': False, 'use_lidar': True},
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 origin: Tuple[float, float, float] = (0.5, 1.0, 0.5),
                 **kwargs) -> None:
        """
        Initialize Kitti Person Dataset
        Args:
            ann_file (str): Path of annotation file.
            pipeline (List[dict]): Pipeline used for data processing.
                Defaults to [].
            modality (dict): Modality to specify the sensor data used as input.
                Defaults to dict(use_lidar=True).
            default_cam_key (str): The default camera name adopted.
                Defaults to 'CAM2'.
            load_type (str): Type of loading mode. Defaults to 'frame_based'.

                - 'frame_based': Load all of the instances in the frame.
                - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
                - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
            box_type_3d (str): Type of 3D box of this dataset.
                Based on the `box_type_3d`, the dataset will encapsulate the box
                to its original format then converted them to `box_type_3d`.
                Defaults to 'LiDAR' in this dataset. Available options includes:

                - 'LiDAR': Box in LiDAR coordinates.
                - 'Depth': Box in depth coordinates, usually for indoor dataset.
                - 'Camera': Box in camera coordinates.
            filter_empty_gt (bool): Whether to filter the data with empty GT.
                If it's set to be True, the example with empty annotations after
                data pipeline will be dropped and a random example will be chosen
                in `__getitem__`. Defaults to True.
            test_mode (bool): Whether the dataset is in test mode.
                Defaults to False.
            pcd_limit_range (List[float]): The range of point cloud used to filter
                invalid predicted boxes.
                Defaults to [0, -40, -3, 70.4, 40, 0.0].
        """
        self.pcd_limit_range = pcd_limit_range
        self.origin = origin
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based'), "load_type is only supported for \
                             frame_based, mv_image_based and fov_image_based"
        self.load_type = load_type
        # override box_type with synthetic data type
        self.default_cam_key = default_cam_key
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert self.modality is not None, "input_modality must be set"
        assert box_type_3d.lower() in ('lidar', 'camera'), \
               "box_type_3d needs to be set to lidar or camera"

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
        self.origin = origin

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
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']

        # noqa pylint: disable=R1705
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = {}

            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if 'label' in ann_name and ann_name != 'attr_label':
                    temp_anns = [
                        self.label_mapping[item] for item in temp_anns
                    ]

                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping.get(ann_name, None)
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)

                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

            for label in ann_info['gt_labels_3d']:
                if label != -1:
                    self.num_ins_per_cat[label] += 1

        if ann_info is None:
            ann_info = {}
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['images'][self.default_cam_key]['lidar2cam'])
        cam2lidar = np.transpose(np.linalg.inv(lidar2cam))
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        if self.box_mode_3d == Box3DMode.CAM:
            gt_bboxes_3d = tao_structures.TAOCameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'], origin=self.origin)
        else:
            gt_bboxes_3d = tao_structures.TAOCameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'], origin=self.origin).convert_to(self.box_mode_3d, cam2lidar, yaw_dim=1)

            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
