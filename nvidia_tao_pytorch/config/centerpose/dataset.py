# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from dataclasses import dataclass
from typing import List

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class CenterPoseDatasetConfig:
    """Dataset config."""

    train_data: str = STR_FIELD(
        value="",
        description="Path to training data.",
        display_name="Training Data"
    )
    test_data: str = STR_FIELD(
        value="",
        description="Path to testing data.",
        display_name="Testing Data"
    )
    val_data: str = STR_FIELD(
        value="",
        description="Path to validation data.",
        display_name="Validation Data"
    )
    inference_data: str = STR_FIELD(
        value="",
        description="Path to inference data.",
        display_name="Inference Data"
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="Batch size.",
        display_name="Batch Size",
        popular="32"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of workers.",
        display_name="Workers",
        popular="8"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        description="Pin memory.",
        display_name="Pin Memory"
    )
    num_classes: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of classes.",
        display_name="Number of Classes"
    )
    num_joints: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=8,
        valid_max=8,
        description="Number of 3D bounding box keypoints.",
        display_name="Number of Keypoints",
        popular="8"
    )
    max_objs: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        valid_max="inf",
        description="Maximum detected number of objects.",
        display_name="Maximum Detected Objects",
        popular="10"
    )
    mean: List[float] = LIST_FIELD(
        arrList=[0.40789654, 0.44719302, 0.47026115],
        description="Mean values for normalization.",
        display_name="Mean"
    )
    std: List[float] = LIST_FIELD(
        arrList=[0.28863828, 0.27408164, 0.27809835],
        description="Standard deviation values for normalization.",
        display_name="Standard Deviation"
    )
    _eig_val: List[float] = LIST_FIELD(
        arrList=[0.2141788, 0.01817699, 0.00341571],
        description="Eigenvalues for color data augmentation from CenterNet."
    )
    _eig_vec: List[List[float]] = LIST_FIELD(
        arrList=[
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ],
        description="Eigenvectors for color data augmentation from CenterNet."
    )
    category: str = STR_FIELD(
        value="cereal_box",
        description="Category of the object.",
        display_name="Category",
        popular="cereal_box"
    )
    num_symmetry: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of the object symmetries.",
        display_name="Number of Symmetries",
        popular="1"
    )
    mse_loss: bool = BOOL_FIELD(
        value=False,
        description="Use mean squared error loss.",
        display_name="Mean Squared Error Loss"
    )
    center_3D: bool = BOOL_FIELD(
        value=False,
        description="Use 3D center loss for the object.",
        display_name="3D Center"
    )
    obj_scale: bool = BOOL_FIELD(
        value=True,
        description="Use object scale loss.",
        display_name="Object Scale"
    )
    use_absolute_scale: bool = BOOL_FIELD(
        value=False,
        description="Use absolute scale loss.",
        display_name="Absolute Scale"
    )
    obj_scale_uncertainty: bool = BOOL_FIELD(
        value=False,
        description="Use object scale uncertainty loss.",
        display_name="Object Scale Uncertainty"
    )
    dense_hp: bool = BOOL_FIELD(
        value=False,
        description="Use dense heatmaps.",
        display_name="Dense Heatmaps"
    )
    hps_uncertainty: bool = BOOL_FIELD(
        value=False,
        description="Use heatmaps uncertainty loss.",
        display_name="Heatmaps Uncertainty"
    )
    reg_bbox: bool = BOOL_FIELD(
        value=True,
        description="Use bounding box regression loss.",
        display_name="Bounding Box Regression"
    )
    reg_offset: bool = BOOL_FIELD(
        value=True,
        description="Use offset regression loss.",
        display_name="Offset Regression"
    )
    hm_hp: bool = BOOL_FIELD(
        value=True,
        description="Use heatmaps for keypoints.",
        display_name="Heatmaps for Keypoints"
    )
    reg_hp_offset: bool = BOOL_FIELD(
        value=True,
        description="Use offset regression loss for keypoints.",
        display_name="Offset Regression for Keypoints"
    )
    flip_idx: List[List[int]] = LIST_FIELD(
        arrList=[[1, 5], [3, 7], [2, 6], [4, 8]],
        description="Flipping indices for keypoints."
    )

    # Data augmentation
    no_color_aug: bool = BOOL_FIELD(
        value=False,
        description="No color augmentation.",
        display_name="No Color Augmentation"
    )
    not_rand_crop: bool = BOOL_FIELD(
        value=False,
        description="No random cropping.",
        display_name="No Random Cropping"
    )
    aug_rot: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Rotation angle for data augmentation.",
        display_name="Rotation Angle"
    )
    flip: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.1,
        valid_max=1.0,
        description="Flip probability for data augmentation.",
        display_name="Flip Probability"
    )
    input_res: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=512,
        valid_max=512,
        description="Input resolution.",
        display_name="Input Resolution"
    )
    output_res: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=128,
        valid_max=128,
        description="Output resolution.",
        display_name="Output Resolution"
    )
