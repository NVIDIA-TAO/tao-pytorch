# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO preprocessing configuration types.

This module provides dataclass definitions for configuring Grounding DINO preprocessing
in DeepStream/Inference Microservices applications.
"""

from typing import List

from dataclasses import dataclass, field, is_dataclass
from nvidia_tao_pytorch.core.types.nvdsinfer import BaseDSType


@dataclass
class GroundingDINOPreprocessPropertyConfig(BaseDSType):
    """Configuration for Grounding DINO preprocessing in DeepStream/Inference Microservices.

    This class defines the preprocessing configuration for Visual ChangeNet inference,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.

    Attributes:
        tensor_name (str): Name of the input tensor. Defaults to "input_1".
        processing_width (int): Width of the processed image. Defaults to 224.
        processing_height (int): Height of the processed image. Defaults to 224.
        network_input_shape (str): Shape of the network input in format "batch;channels;height;width".
            Defaults to "2;3;224;224".
        network_color_format (int): Color format of the network input. Defaults to 0.
        network_input_order (int): Order of the network input. Defaults to 0.
        tensor_data_type (int): Data type of the tensor. Defaults to 0.
        maintain_aspect_ratio (int): Flag to maintain aspect ratio during preprocessing.
            Defaults to 1.
        symmetric_padding (int): Flag to enable symmetric padding. Defaults to 1.
        scaling_pool_memory_type (int): Memory type for scaling pool. Defaults to 0.
        scaling_pool_compute_hw (int): Compute hardware for scaling pool. Defaults to 0.
        scaling_filter (int): Scaling filter type. Defaults to 0.
        scaling_buf_pool_size (int): Size of the scaling buffer pool. Defaults to 6.
        tensor_buf_pool_size (int): Size of the tensor buffer pool. Defaults to 6.
        target_unique_ids (int): Flag for target unique IDs. Defaults to 1.
        process_on_frame (int): Flag to process on frame. Defaults to 1.
        unique_id (int): Unique identifier for the preprocessing configuration. Defaults to 5.
        custom_lib_path (str): Path to the custom preprocessing library.
            Defaults to "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so".
        custom_tensor_preparation_function (str): Name of the custom tensor preparation function.
            Defaults to "CustomTensorPreparation".
        enable (int): Flag to enable preprocessing. Defaults to 1.
    """

    tensor_name: str = "inputs"
    processing_width: int = 960
    processing_height: int = 544
    network_input_shape: List[int] = field(default_factory=lambda: [1, 3, 544, 960])
    network_color_format: int = 0
    network_input_order: int = 0
    tensor_data_type: int = 0
    maintain_aspect_ratio: int = 1
    symmetric_padding: int = 1
    scaling_pool_memory_type: int = 0
    scaling_pool_compute_hw: int = 0
    scaling_filter: int = 0
    scaling_buf_pool_size: int = 6
    tensor_buf_pool_size: int = 6
    target_unique_ids: int = 1
    process_on_frame: int = 1
    unique_id: int = 5
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so"
    custom_tensor_preparation_function: str = "CustomTensorPreparation"
    enable: int = 1

    def validate(self):
        """Validates the preprocessing configuration.

        Currently a placeholder for future validation logic.
        """
        pass


@dataclass
class GroundingDINOUserConfigs(BaseDSType):
    """User configuration for Grounding DINO preprocessing.

    This class defines the user configuration for Visual ChangeNet preprocessing,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.
    """

    pixel_normalization_factor: float = 0.007843137
    offsets: List = field(default_factory=lambda: [127.5, 127.5, 127.5])

    def validate(self):
        """Validates the user configuration.

        Currently a placeholder for future validation logic.
        """
        pass


@dataclass
class GroundingDINOGroupConfig(BaseDSType):
    """Configuration for Grounding DINO preprocessing in DeepStream/Inference Microservices.

    This class defines the preprocessing configuration for Visual ChangeNet inference,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.
    """

    src_ids: int = -1
    custom_input_transformation_function: str = "CustomAsyncTransformation"
    process_on_roi: int = 0

    def validate(self):
        """Validates the user configuration.

        Currently a placeholder for future validation logic.
        """
        assert len(self.src_ids) == 8, "src_ids must be a list of 8 integers"
        assert all(item < 16 for item in self.src_ids), (
            "src_ids must be less than 16, by forced requirement from Inference Microservices"
        )


@dataclass
class GroundingDINOPreprocessConfig(BaseDSType):
    """Configuration for Grounding DINO preprocessing in DeepStream/Inference Microservices.

    This class defines the preprocessing configuration for Grounding DINO inference,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.
    """

    property: GroundingDINOPreprocessPropertyConfig = field(
        default_factory=lambda: GroundingDINOPreprocessPropertyConfig()
    )
    user_configs: GroundingDINOUserConfigs = field(
        default_factory=lambda: GroundingDINOUserConfigs()
    )
    group: List[GroundingDINOGroupConfig] = field(
        default_factory=lambda: [GroundingDINOGroupConfig()]
    )

    def validate(self):
        """Validates the preprocessing configuration.

        Currently a placeholder for future validation logic.
        """
        pass


if __name__ == "__main__":
    grounding_dino_config = GroundingDINOPreprocessConfig()
    assert is_dataclass(grounding_dino_config), "The instance of grounding_dino_config is not a dataclass."
    print(str(grounding_dino_config))
