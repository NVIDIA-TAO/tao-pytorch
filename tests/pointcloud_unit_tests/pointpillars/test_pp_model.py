# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from easydict import EasyDict
import yaml
import numpy as np
import pytest
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models.detectors.pointpillar import (
    PointPillar
)

MODEL_CONFIG = """
model:
    name: PointPillar
    vfe:
        name: PillarVFE
        with_distance: False
        use_absolue_xyz: True
        use_norm: True
        num_filters: [64]
    map_to_bev:
        name: PointPillarScatter
        num_bev_features: 64
    backbone_2d:
        name: BaseBEVBackbone
        layer_nums: [3, 5, 5]
        layer_strides: [2, 2, 2]
        num_filters: [64, 128, 256]
        upsample_strides: [1, 2, 4]
        num_upsample_filters: [128, 128, 128]
    dense_head:
        name: AnchorHeadSingle
        class_agnostic: False
        use_direction_classifier: True
        dir_offset: 0.78539
        dir_limit_offset: 0.0
        num_dir_bins: 2
        anchor_generator_config: [
            {
                'class_name': 'Vehicle',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]
        target_assigner_config:
            name: AxisAlignedTargetAssigner
            pos_fraction: -1.0
            sample_size: 512
            norm_by_num_examples: False
            match_height: False
            box_coder: ResidualCoder
        loss_config:
            loss_weights: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
    post_processing:
        recall_thresh_list: [0.3, 0.5, 0.7]
        score_thresh: 0.1
        output_raw_score: False
        eval_metric: kitti
        nms_config:
            multi_classes_nms: False
            nms_type: nms_gpu
            nms_thresh: 0.01
            nms_pre_max_size: 4096
            nms_post_max_size: 500
    sync_bn: False
"""

class DATASET:
    pass


@pytest.mark.pointcloud_unit
def test_model():
    """Test model builder."""
    model_config = EasyDict(yaml.safe_load(MODEL_CONFIG)["model"])
    dataset = DATASET()
    dataset.class_names = ["Vehicle", "Pedestrian", "Cyclist"]
    dataset.point_feature_encoder = DATASET()
    dataset.point_feature_encoder.num_point_features = 4
    dataset.voxel_size = np.array([0.2, 0.2, 5.8])
    dataset.point_cloud_range = np.array(
        [-51.2, -51.2, -1.4, 51.2, 51.2, 4.4]
    )
    dataset.grid_size = np.round((
        dataset.point_cloud_range[3:6] - dataset.point_cloud_range[0:3]
    ) / dataset.voxel_size)
    model = PointPillar(
        model_config, 3, dataset
    )
    assert len(model.module_list) == 4
