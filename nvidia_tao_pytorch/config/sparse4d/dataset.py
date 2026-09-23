# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import Dict, List
from dataclasses import dataclass
from omegaconf import MISSING

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
class QuantCalibrationDataset:
    """Quantization calibration dataset config."""

    images_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the directory containing calibration images.",
        display_name="calibration images directory",
    )


@dataclass
class Sparse4DTrainDatasetConfig:
    """Training dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file",
    )
    test_mode: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Test mode",
        display_name="Test mode",
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag",
    )
    with_seq_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="With sequence flag",
        display_name="With sequence flag",
    )
    sequences_split_num: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        description="Number of sequences",
        display_name="Number of sequences",
    )
    keep_consistent_seq_aug: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep consistent sequence augmentation",
        display_name="Keep consistent sequence augmentation",
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch",
    )


@dataclass
class Sparse4DValDatasetConfig:
    """Validation dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file",
    )
    test_mode: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Test mode",
        display_name="Test mode",
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag",
    )
    tracking: bool = BOOL_FIELD(
        value=True, default_value=True, description="Tracking", display_name="Tracking"
    )
    tracking_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold",
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch",
    )


@dataclass
class Sparse4DTestDatasetConfig:
    """Test dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file",
    )
    test_mode: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Test mode",
        display_name="Test mode",
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag",
    )
    tracking: bool = BOOL_FIELD(
        value=True, default_value=True, description="Tracking", display_name="Tracking"
    )
    tracking_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold",
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch",
    )


@dataclass
class Sparse4DAugmentationConfig:
    """Augmentation configuration for Sparse4D."""

    resize_lim: List[float] = LIST_FIELD(
        arrList=[0.7, 0.77],
        default_value=[0.7, 0.77],
        description="Resize limits",
        display_name="Resize limits",
    )
    final_dim: List[int] = LIST_FIELD(
        arrList=[512, 1408],
        default_value=[512, 1408],
        description="Final dimensions",
        display_name="Final dimensions",
    )
    bot_pct_lim: List[float] = LIST_FIELD(
        arrList=[0.0, 0.0],
        default_value=[0.0, 0.0],
        description="Bottom percentage limits",
        display_name="Bottom percentage limits",
    )
    rot_lim: List[float] = LIST_FIELD(
        arrList=[-5.4, 5.4],
        default_value=[-5.4, 5.4],
        description="Rotation limits in degrees",
        display_name="Rotation limits in degrees",
    )
    image_size: List[int] = LIST_FIELD(
        arrList=[1080, 1920],
        default_value=[1080, 1920],
        description="Original image size",
        display_name="Original image size",
    )
    rand_flip: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Random flip",
        display_name="Random flip",
    )
    rot3d_range: List[float] = LIST_FIELD(
        arrList=[-0.3925, 0.3925],
        default_value=[-0.3925, 0.3925],
        description="3D rotation range in radians",
        display_name="3D rotation range in radians",
    )


@dataclass
class Sparse4DNormalizeConfig:
    """Normalization configuration for Sparse4D."""

    mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        default_value=[123.675, 116.28, 103.53],
        description="Mean values for normalization",
        display_name="Mean values for normalization",
    )
    std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        default_value=[58.395, 57.12, 57.375],
        description="Standard deviation values for normalization",
        display_name="Standard deviation values for normalization",
    )
    to_rgb: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Convert to RGB",
        display_name="Convert to RGB",
    )


@dataclass
class Sparse4DSequencesConfig:
    """Sequences configuration for Sparse4D."""

    split_num: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        description="Number of sequence splits",
        display_name="Number of sequence splits",
    )
    keep_consistent_aug: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep consistent augmentation",
        display_name="Keep consistent augmentation",
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep same scene in batch",
        display_name="Keep same scene in batch",
    )


@dataclass
class Sparse4DTrackingConfig:
    """Tracking configuration for Sparse4D."""

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable tracking",
        display_name="Enable tracking",
    )
    threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold",
    )


@dataclass
class Omniverse3DDetTrackDatasetConfig:
    """Dataset configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="omniverse_3d_det_track",
        default_value="omniverse_3d_det_track",
        description="Dataset type",
        display_name="Dataset type",
    )
    batch_size: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch size",
    )
    use_h5_file_for_rgb: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Read RGB camera images from HDF5 files instead of image directories",
        display_name="Use HDF5 for RGB images",
    )
    use_h5_file_for_depth: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use H5 file for depth maps",
        display_name="Use H5 file for depth",
    )
    lazy_load: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Defer pkl loading to __getitem__ using a pre-built index",
        display_name="Lazy load annotations",
    )
    lazy_load_cache_size: int = INT_FIELD(
        value=50,
        default_value=50,
        valid_min=1,
        valid_max="inf",
        description="Maximum pkl files held in the lazy-loading LRU cache",
        display_name="Lazy load cache size",
    )
    pkl_sample_size: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Number of PKLs sampled per epoch; zero disables sampling. Requires lazy_load=True",
        display_name="PKL sample size per epoch",
    )
    pkl_cam_counts_path: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the pkl-to-camera-count mapping used for balanced sampling",
        display_name="PKL camera counts path",
    )
    fps_drop_prob: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability of downsampling a training scene to a lower FPS",
        display_name="FPS drop probability",
    )
    target_fps_choices: List[int] = LIST_FIELD(
        arrList=[30, 20, 15, 10, 6, 5, 3, 2, 1],
        default_value=[30, 20, 15, 10, 6, 5, 3, 2, 1],
        description="Candidate target FPS values for FPS-drop augmentation",
        display_name="Target FPS choices",
    )
    max_cameras: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        valid_max="inf",
        description="Limit training cameras per frame to reduce GPU memory; a non-positive value disables sampling",
        display_name="Maximum cameras per frame",
    )
    eval_dist_fcn: str = STR_FIELD(
        value="center_distance",
        default_value="center_distance",
        description="Evaluation distance function",
        display_name="Evaluation distance function",
        valid_options="center_distance,iou_3d,both",
    )
    eval_hota: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Run HOTA tracking evaluation in addition to AMOTA",
        display_name="Enable HOTA evaluation",
    )
    ltt_2dgt_sidecar_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory of per-scene LooseToTight 2D ground-truth NPZ sidecars",
        display_name="LTT 2D ground-truth sidecar directory",
    )
    ltt_2dgt_frame_regex: str = STR_FIELD(
        value="",
        default_value="",
        description="Optional regex used to extract frame IDs for LTT sidecar joins",
        display_name="LTT frame regex",
    )
    ltt_2dgt_dedup_regex: str = STR_FIELD(
        value=r"^CT[\w.]+?__",
        default_value=r"^CT[\w.]+?__",
        description="Regex removed from scene names before resolving LTT sidecars",
        display_name="LTT scene deduplication regex",
    )
    ltt_2dgt_cache_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of per-scene LTT sidecar indices cached per worker",
        display_name="LTT sidecar cache size",
    )
    ltt_2dgt_warn_on_miss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Warn once per scene when an LTT sidecar join has no matches",
        display_name="Warn on LTT sidecar miss",
    )
    rtdetr_2d_cache_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory of per-scene RT-DETR 2D pseudo-label NPZ caches",
        display_name="RT-DETR 2D cache directory",
    )
    rtdetr_2d_cache_path: str = STR_FIELD(
        value="",
        default_value="",
        description="Optional single RT-DETR 2D pseudo-label NPZ cache",
        display_name="RT-DETR 2D cache path",
    )
    rtdetr_2d_dedup_regex: str = STR_FIELD(
        value=r"^CT[\w.]+?__",
        default_value=r"^CT[\w.]+?__",
        description="Regex removed from scene names before resolving RT-DETR caches",
        display_name="RT-DETR scene deduplication regex",
    )
    rtdetr_2d_score_thr: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Global score threshold for RT-DETR pseudo-labels",
        display_name="RT-DETR score threshold",
    )
    rtdetr_2d_per_class_score_thr: Dict[str, float] = DICT_FIELD(
        hashMap={},
        description="Optional class-name to score-threshold overrides",
        display_name="RT-DETR per-class score thresholds",
    )
    rtdetr_2d_cache_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="Number of RT-DETR scene indices cached per worker",
        display_name="RT-DETR cache size",
    )
    rtdetr_2d_mark_real: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Mark samples with an RT-DETR scene cache as lacking 3D ground truth",
        display_name="Treat RT-DETR-cached samples as 2D-only",
    )
    resize_to_canonical_2d: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Resize explicitly 2D-only training images to the canonical SV2D size",
        display_name="Enable canonical SV2D resize",
    )
    canonical_2d_height: int = INT_FIELD(
        value=1080,
        default_value=1080,
        valid_min=1,
        valid_max="inf",
        description="Canonical SV2D image height",
        display_name="Canonical SV2D height",
    )
    canonical_2d_width: int = INT_FIELD(
        value=1920,
        default_value=1920,
        valid_min=1,
        valid_max="inf",
        description="Canonical SV2D image width",
        display_name="Canonical SV2D width",
    )
    sync_route: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Keep distributed ranks on the same 3D or 2D supervision route",
        display_name="Synchronize co-training route",
    )
    real_scene_keywords: List[str] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Scene-name substrings identifying real 2D-supervised data",
        display_name="Real scene keywords",
    )
    real_block_prob: float = FLOAT_FIELD(
        value=-1.0,
        default_value=-1.0,
        valid_min=-1.0,
        valid_max=1.0,
        description="Probability in [0, 1], or exactly -1 for automatic scene-count weighting; (-1, 0) is invalid",
        display_name="Real route block probability",
    )
    scene_switch_iters: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Iterations per scene before a synchronized route switch; zero uses legacy cadence",
        display_name="Scene switch iterations",
    )
    num_frames: int = INT_FIELD(
        value=200,
        default_value=200,
        valid_min=1,
        valid_max="inf",
        description="Number of frames",
        display_name="Number of frames",
    )
    num_bev_groups: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of BEV groups",
        display_name="Number of BEV groups",
    )
    data_root: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to data root",
        display_name="Path to data root",
    )
    classes: List[str] = LIST_FIELD(
        arrList=[
            "person",
            "gr1_t2",
            "agility_digit",
            "nova_carter",
        ],
        default_value=[
            "person",
            "gr1_t2",
            "agility_digit",
            "nova_carter",
        ],
        description="Classes to detect",
        display_name="Classes to detect",
    )
    num_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        valid_max="inf",
        description="Number of workers",
        display_name="Number of workers",
    )
    num_ids: int = INT_FIELD(
        value=70,
        default_value=70,
        valid_min=1,
        valid_max="inf",
        description="Number of IDs",
        display_name="Number of IDs",
    )
    augmentation: Sparse4DAugmentationConfig = DATACLASS_FIELD(
        Sparse4DAugmentationConfig(),
        description="Augmentation config",
        display_name="Augmentation config",
    )
    normalize: Sparse4DNormalizeConfig = DATACLASS_FIELD(
        Sparse4DNormalizeConfig(),
        description="Normalize config",
        display_name="Normalize config",
    )
    sequences: Sparse4DSequencesConfig = DATACLASS_FIELD(
        Sparse4DSequencesConfig(),
        description="Sequences config",
        display_name="Sequences config",
    )
    train_dataset: Sparse4DTrainDatasetConfig = DATACLASS_FIELD(
        Sparse4DTrainDatasetConfig(),
        description="Train dataset config",
        display_name="Train dataset config",
    )
    val_dataset: Sparse4DValDatasetConfig = DATACLASS_FIELD(
        Sparse4DValDatasetConfig(),
        description="Val dataset config",
        display_name="Val dataset config",
    )
    test_dataset: Sparse4DTestDatasetConfig = DATACLASS_FIELD(
        Sparse4DTestDatasetConfig(),
        description="Test dataset config",
        display_name="Test dataset config",
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for quantization calibration dataset.",
    )
