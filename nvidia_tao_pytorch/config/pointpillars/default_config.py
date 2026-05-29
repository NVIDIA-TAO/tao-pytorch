# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class PPVFEConfig:
    """VFE config."""

    name: str = STR_FIELD(
        value="PillarVFE",
        default_value="PillarVFE",
        display_name="VFE",
        description="The VFE module for PointPillars model.",
        valid_options="PillarVFE"
    )
    with_distance: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="with_distancce",
        description="Flag to enable with_distance for VFE or not."
    )
    use_absolue_xyz: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use_absolue_xyz",
        description="Flag to use absolute xyz or not."
    )
    use_norm: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use_norm",
        description="Flag to use norm or not."
    )
    num_filters: List[int] = LIST_FIELD(
        [64],
        default_value=[64],
        display_name="num_filters",
        description="Number of filters for VFE module."
    )


@dataclass
class PPMapToBEVConfig:
    """Map to BEV config."""

    name: str = STR_FIELD(
        value="PointPillarScatter",
        default_value="PointPillarScatter",
        display_name="PointPillarScatter",
        description="PointPillarScatter module for PointPillars.",
        valid_options="PointPillarScatter"
    )
    num_bev_features: int = INT_FIELD(
        value=64,
        default_value=64,
        valid_min=64,
        valid_max=64,
        display_name="num_bev_features",
        description="Number of BEV features for MapToBEV module."
    )


@dataclass
class PPBackbone2DConfig:
    """Backbone 2d config."""

    name: str = STR_FIELD(
        value="BaseBEVBackbone",
        default_value="BaseBEVBackbone",
        display_name="BaseBEVBackbone",
        description="BaseBEVBackbone module for PointPillars model.",
        valid_options="BaseBEVBackbone"
    )
    layer_nums: List[int] = LIST_FIELD(
        [3, 5, 5],
        default_value=[3, 5, 5],
        display_name="layer_nums",
        description="Number of layers for BaseBEVBackbone module."
    )
    layer_strides: List[int] = LIST_FIELD(
        [2, 2, 2],
        default_value=[2, 2, 2],
        display_name="layer_strides",
        description="layer strides for BaseBEVBackbone module."
    )
    num_filters: List[int] = LIST_FIELD(
        [64, 128, 256],
        default_value=[64, 128, 256],
        display_name="num_filters",
        description="Number of filters for each layer of BaseBEVBackbone module."
    )
    upsample_strides: List[int] = LIST_FIELD(
        [1, 2, 4],
        default_value=[1, 2, 4],
        display_name="upsample_strides",
        description="Upsample strides for each layer of BaseBEVBackbone module."
    )
    num_upsample_filters: List[int] = LIST_FIELD(
        [128, 128, 128],
        default_value=[128, 128, 128],
        display_name="num_upsample_filters",
        description="Number of upsample filters for each layer of BaseBEVBackbone module."
    )


@dataclass
class PPTargetAssignerConfig:
    """Target assigner config."""

    name: str = STR_FIELD(
        value="AxisAlignedTargetAssigner",
        default_value="AxisAlignedTargetAssigner",
        display_name="name",
        description="Name of target assigner module of PointPillars.",
        valid_options="AxisAlignedTargetAssigner"
    )
    pos_fraction: float = FLOAT_FIELD(
        value=-1.0,
        default_value=-1.0,
        display_name="pos_fraction",
        description="Positive fraction of target assigner.",
        valid_min=-1.0,
        valid_max=-1.0
    )
    sample_size: int = INT_FIELD(
        value=512,
        default_value=512,
        display_name="sample_size",
        description="Sample size of target assigner.",
        valid_min=512,
        valid_max=512
    )
    norm_by_num_examples: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="norm_by_num_examples",
        description="Flag to enable normalization by number of examples or not."
    )
    match_height: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="match_height",
        description="Flag to enable match height or not."
    )
    box_coder: str = STR_FIELD(
        value="ResidualCoder",
        default_value="ResidualCoder",
        display_name="box_coder",
        description="Type of the box coder.",
        valid_options="ResidualCoder"
    )


@dataclass
class PPLossConfig:
    """Loss config."""

    loss_weights: Dict[str, Any] = DICT_FIELD(
        {
            'cls_weight': 1.0,
            'loc_weight': 2.0,
            'dir_weight': 0.2,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        default_value={
            'cls_weight': 1.0,
            'loc_weight': 2.0,
            'dir_weight': 0.2,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        },
        display_name="loss_weights",
        description="Weighting factors for loss functions."
    )


@dataclass
class PPDenseHeadConfig:
    """Dense head config."""

    name: str = STR_FIELD(
        value="AnchorHeadSingle",
        default_value="AnchorHeadSingle",
        display_name="name",
        description="Name of the DenseHead module.",
        valid_options="AnchorHeadSingle"
    )
    class_agnostic: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="class_agnostic",
        description="Flag to enable class agnostic or not."
    )
    use_direction_classifier: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use_direction_classifier",
        description="Flag to use direction classifier or not."
    )
    dir_offset: float = FLOAT_FIELD(
        value=0.78539,
        default_value=0.78539,
        display_name="dir_offset",
        description="Direction offset.",
        valid_min=0.78539,
        valid_max=0.78539
    )
    dir_limit_offset: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        display_name="dir_limit_offset",
        description="Direction limit offset.",
        valid_min=0.0,
        valid_max=0.0
    )
    num_dir_bins: int = INT_FIELD(
        value=2,
        default_value=2,
        display_name="num_dir_bins",
        description="Number of direction bins.",
        valid_min=2,
        valid_max=2
    )
    anchor_generator_config: List[Dict[str, Any]] = LIST_FIELD(
        [
            {
                'class_name': 'Car',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Truck',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Van',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Tram',
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
            },
            {
                'class_name': 'Misc',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ],
        default_value=[
            {
                'class_name': 'Car',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Truck',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Van',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Tram',
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
            },
            {
                'class_name': 'Misc',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ],
        display_name="anchor_generator_config",
        description="Config for anchor generation."
    )
    target_assigner_config: PPTargetAssignerConfig = DATACLASS_FIELD(PPTargetAssignerConfig())
    loss_config: PPLossConfig = DATACLASS_FIELD(PPLossConfig())


@dataclass
class PPNMSConfig:
    """NMS config."""

    multi_classes_nms: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="multi_classes_nms",
        description="Flag to enable multi-class NMS or not."
    )
    nms_type: str = STR_FIELD(
        value="nms_gpu",
        default_value="nms_gpu",
        display_name="nms_type",
        description="Type of NMS operation.",
        valid_options="nms_gpu"
    )
    nms_thresh: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        display_name="nms_thresh",
        description="NMS threshold.",
        valid_min=0.0,
        valid_max=1.0
    )
    nms_pre_max_size: int = INT_FIELD(
        value=4096,
        default_value=4096,
        display_name="nms_pre_max_size",
        description="Maximum number of inputs for NMS operation.",
        valid_min=1,
        valid_max="inf"
    )
    nms_post_max_size: int = INT_FIELD(
        value=500,
        default_value=500,
        display_name="nms_post_max_size",
        description="Maximum number of outputs for NMS operation.",
        valid_min=1,
        valid_max="inf"
    )


@dataclass
class PPPostprocessingConfig:
    """Postprocessing config."""

    recall_thresh_list: List[float] = LIST_FIELD(
        [0.3, 0.5, 0.7, 0.3, 0.3, 0.3, 0.3],
        default_value=[0.3, 0.5, 0.7, 0.3, 0.3, 0.3, 0.3],
        display_name="recall_thresh_list",
        description="List of recall thresholds."
    )
    score_thresh: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="score_thresh",
        description="Score threshold.",
        valid_min=0.0,
        valid_max=1.0
    )
    output_raw_score: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="output_raw_score",
        description="Flag to output raw score or not."
    )
    eval_metric: str = STR_FIELD(
        value="kitti",
        default_value="kitti",
        display_name="eval_metric",
        description="Evaluation metric.",
        valid_options="kitti"
    )
    nms_config: PPNMSConfig = DATACLASS_FIELD(PPNMSConfig())


@dataclass
class PPModelConfig:
    """Model config."""

    name: str = STR_FIELD(
        value="PointPillar",
        default_value="PointPillar",
        display_name="name",
        description="Name of the PointPillars model.",
        valid_options="PointPillar"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained_model_path",
        description="Path to pretrained model."
    )
    vfe: PPVFEConfig = DATACLASS_FIELD(PPVFEConfig())
    map_to_bev: PPMapToBEVConfig = DATACLASS_FIELD(PPMapToBEVConfig())
    backbone_2d: PPBackbone2DConfig = DATACLASS_FIELD(PPBackbone2DConfig())
    dense_head: PPDenseHeadConfig = DATACLASS_FIELD(PPDenseHeadConfig())
    post_processing: PPPostprocessingConfig = DATACLASS_FIELD(PPPostprocessingConfig())
    sync_bn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="sync_bn",
        description="Flag to use sync BN or not."
    )


@dataclass
class PPDataAugmentorConfig:
    """Augmentation config."""

    disable_aug_list: List[str] = LIST_FIELD(
        ['placeholder'],
        default_value=['placeholder'],
        display_name="disable_aug_list",
        description="List of disabled augmentations"
    )
    aug_config_list: Optional[List[Any]] = LIST_FIELD(
        arrList=[{
            "db_info_path": ["dbinfos_train.pkl"],
            "disable_with_fake_lidar": False,
            "limit_whole_scene": False,
            "name": "gt_sampling",
            "num_point_features": 4,
            "preface": {
                "filter_by_min_points": ["Car:5", "Truck:5", "Van:5", "Tram:5", "Pedestrian:5", "Cyclist:5", "Misc:5"]
            },
            "remove_extra_width": [0.0, 0.0, 0.0],
            "sample_groups": ["Car:15", "Truck:15", "Van:15", "Tram:15", "Pedestrian:15", "Cyclist:15", "Misc:15"]
        }],
        display_name="aug_config_list",
        description="List of configurations of augmentations."
    )


@dataclass
class PPDatasetConfig:
    """Dataset config."""

    class_names: List[str] = LIST_FIELD(
        ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist', 'Misc'],
        default_value=['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist', 'Misc'],
        display_name="class_names",
        description="List of names of object classes."
    )
    type: str = STR_FIELD(
        value="GeneralPCDataset",
        default_value="GeneralPCDataset",
        display_name="type",
        description="Type of dataset.",
        valid_options="GeneralPCDataset"
    )
    data_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="data_path",
        description="Path to data."
    )
    data_info_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="data_info_path",
        description="Path to data info."
    )
    data_split: Optional[Dict[str, str]] = DICT_FIELD(
        {"test": "val", "train": "train"},
        display_name="data_split",
        description="Split of data."
    )
    info_path: Optional[Dict[str, List[str]]] = DICT_FIELD(
        {"test": ["infos_val.pkl"], "train": ["infos_train.pkl"]},
        display_name="info_path",
        description="Path to info."
    )
    balanced_resampling: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="balanced_resampling",
        description="Flag to enable balanced resampling or not."
    )
    point_feature_encoding: Optional[Dict[str, Any]] = DICT_FIELD(
        {
            "encoding_type": "absolute_coordinates_encoding",
            "src_feature_list": ["x", "y", "z", "intensity"],
            "used_feature_list": ["x", "y", "z", "intensity"]
        },
        display_name="point_feature_encoding",
        description="Point feature encoding configurations."
    )
    point_cloud_range: Optional[List[float]] = LIST_FIELD(
        arrList=[5.245, -25.983, -3.854, 79.485, 48.257, 2.908],
        display_name="point_cloud_range",
        description="Point cloud's coordinate range."
    )
    data_augmentor: PPDataAugmentorConfig = DATACLASS_FIELD(PPDataAugmentorConfig())
    data_processor: Optional[List[Any]] = LIST_FIELD(
        arrList=[
            {
                "name": "mask_points_and_boxes_outside_range",
                "remove_outside_boxes": True
            },
            {
                "name": "shuffle_points",
                "shuffle": {"test": False, "train": True}
            },
            {
                "max_number_of_voxels": {"test": 10000, "train": 16000},
                "max_points_per_voxel": 32,
                "name": "transform_points_to_voxels",
                "voxel_size": [0.16, 0.16, 6.762]
            }
        ],
        display_name="data_processor",
        description="Data processor configurations."
    )
    num_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="num_workers",
        description="Number of workers."
    )


@dataclass
class PPTrainConfig:
    """Train experiment config."""

    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="batch_size",
        description="Batch size.",
        valid_min=1,
        valid_max="inf"
    )
    num_epochs: int = INT_FIELD(
        value=80,
        default_value=80,
        display_name="num_epochs",
        description="Number of epochs to train for.",
        valid_min=1,
        valid_max="inf"
    )
    optimizer: str = STR_FIELD(
        value="adam_onecycle",
        default_value="adam_onecycle",
        display_name="optimizer",
        description="Type of optimizer.",
        valid_options="adam_onecycle"
    )
    lr: float = FLOAT_FIELD(
        value=0.003,
        default_value=0.003,
        display_name="lr",
        description="Learning rate.",
        valid_min=0.0,
        valid_max="inf",
        automl_enabled="True"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        display_name="weight_decay",
        description="Weighting decay factor.",
        valid_min=0.01,
        valid_max=1.0,
        automl_enabled="True"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        display_name="momentum",
        description="Momentum.",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="True"
    )
    moms: List[float] = LIST_FIELD(
        [0.95, 0.85],
        default_value=[0.95, 0.85],
        display_name="moms",
        description="Moms."
    )
    pct_start: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        display_name="pct_start",
        description="pct_start.",
        valid_min=0.0,
        valid_max=1.0
    )
    div_factor: float = FLOAT_FIELD(
        value=10.0,
        default_value=10.0,
        display_name="div_factor",
        description="div_factor.",
        valid_min=1.0,
        valid_max="inf"
    )
    decay_step_list: List[int] = LIST_FIELD(
        [35, 45],
        default_value=[35, 45],
        display_name="decay_step_list",
        description="List of steps for decaying learning rate.",
        automl_enabled="True",
        value_type="list_2",
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="lr_decay",
        description="Learning rate decay.",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="True"
    )
    lr_clip: float = FLOAT_FIELD(
        value=0.0000001,
        default_value=0.0000001,
        display_name="lr_clip",
        description="Learning rate clip.",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="True"
    )
    lr_warmup: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="lr_warmup",
        description="Flag to enable learning rate warmup or not.",
        automl_enabled="True"
    )
    warmup_epoch: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="warmup_epoch",
        description="Number of epochs for warming up the learning rate.",
        valid_min=0,
        valid_max="inf"
    )
    grad_norm_clip: float = FLOAT_FIELD(
        value=10.0,
        default_value=10.0,
        display_name="grad_norm_clip",
        description="Grad norm clip.",
        valid_min=0.0,
        valid_max="inf"
    )
    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="resume_training_checkpoint_path",
        description="Path to checkpoint for resuming the training."
    )
    pruned_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pruned_model_path",
        description="Path to pruned model."
    )
    tcp_port: int = INT_FIELD(
        value=18888,
        default_value=18888,
        display_name="tcp_port",
        description="TCP port number.",
        valid_min=49152,
        valid_max=65535
    )
    random_seed: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="random_seed",
        description="Random seed."
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="checkpoint_interval",
        description="Interval of epochs to save checkpoints.",
        valid_min=1,
        valid_max="inf"
    )
    validation_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="checkpoint_interval",
        description="Interval of epochs to save checkpoints.",
        valid_min=1,
        valid_max="inf"
    )
    max_checkpoint_save_num: int = INT_FIELD(
        value=30,
        default_value=30,
        display_name="max_checkpoint_save_num",
        description="Maximum number of checkpoints to save.",
        valid_min=1,
        valid_max="inf"
    )
    merge_all_iters_to_one_epoch: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="merge_all_iters_to_one_epoch",
        description="Flag to merge all iterations into one epoch or not."
    )
    num_gpus: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="num_gpus",
        description="Number of GPUs.",
        valid_min=1,
        valid_max="inf"
    )
    gpu_ids: List[int] = LIST_FIELD(
        [0],
        default_value=[0],
        display_name="gpu_ids",
        description="GPU IDs."
    )


@dataclass
class PPEvalConfig:
    """Eval config."""

    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="batch_size",
        description="Batch size.",
        valid_min=1,
        valid_max="inf"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="checkpoint",
        description="Path to checkpoint to evaluate on."
    )
    save_to_file: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="save_to_file",
        description="Flag to save evaluation result to file or not."
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="trt_engine",
        description="Path to TensorRT engine to evaluate on."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="results_dir",
        description="Path to evaluation results directory."
    )


@dataclass
class PPInferConfig:
    """Infer config."""

    max_points_num: int = INT_FIELD(
        value=25000,
        default_value=25000,
        display_name="max_points_num",
        description="Maximum number of points."
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="batch_size",
        description="Batch size.",
        valid_min=1,
        valid_max="inf"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="checkpoint",
        description="Path to checkpoint to do inference on."
    )
    viz_conf_thresh: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="viz_conf_thresh",
        description="Confidence threshold for visualization.",
        valid_min=0.0,
        valid_max=1.0
    )
    save_to_file: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="save_to_file",
        description="Flag to save inference result to file or not."
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="trt_engine",
        description="Path to TensorRT engine to do inference on."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="results_dir",
        description="Path to inference results directory."
    )


@dataclass
class PPExportConfig:
    """Export config."""

    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        display_name="gpu_id",
        description="GPU ID.",
        valid_min=0,
        valid_max="inf"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="checkpoint",
        description="Path to checkpoint to export from."
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="onnx_file",
        description="Path to ONNX file to save to."
    )
    cal_data_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="cal_data_path",
        description="Path to calibration data for INT8 TensorRT engine."
    )
    cal_cache_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="cal_cache_file",
        description="Path to INT8 calibration cache file to save to."
    )
    data_type: Optional[str] = STR_FIELD(
        value="fp32",
        default_value="fp32",
        display_name="data_type",
        description="Data type of TensorRT engine.",
        valid_options="fp32,fp16"
    )
    save_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="save_engine",
        description="Path to TensorRT engine to save to."
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="batch_size",
        description="Batch size to export.",
        valid_min=1,
        valid_max="inf"
    )
    cal_num_batches: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="cal_num_batches",
        description="Number of batches of data used for INT8 calibration."
    )
    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        display_name="workspace_size",
        description="Workspace size in MB for building TensorRT engine.",
        valid_min=1,
        valid_max="inf"
    )


@dataclass
class PPPruneConfig:
    """Prune config."""

    model: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="model",
        description="Path to model to be pruned."
    )
    pruning_thresh: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="pruning_thresh",
        description="Pruning threshold.",
        valid_min=0.0,
        valid_max=1.0
    )


@dataclass
class ExperimentConfig:
    """Experiment config."""

    dataset: PPDatasetConfig = DATACLASS_FIELD(PPDatasetConfig())
    model: PPModelConfig = DATACLASS_FIELD(PPModelConfig())
    train: PPTrainConfig = DATACLASS_FIELD(PPTrainConfig())
    evaluate: PPEvalConfig = DATACLASS_FIELD(PPEvalConfig())
    inference: PPInferConfig = DATACLASS_FIELD(PPInferConfig())
    export: PPExportConfig = DATACLASS_FIELD(PPExportConfig())
    gen_trt_engine: PPExportConfig = DATACLASS_FIELD(PPExportConfig())
    prune: PPPruneConfig = DATACLASS_FIELD(PPPruneConfig())
    key: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="key",
        description="Key to encoding/decoding models."
    )
    local_rank: int = INT_FIELD(
        value=0,
        default_value=0,
        display_name="local_rank",
        description="Local rank ID.",
        valid_min=0,
        valid_max="inf"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="results_dir",
        description="Path to directory of results"
    )
