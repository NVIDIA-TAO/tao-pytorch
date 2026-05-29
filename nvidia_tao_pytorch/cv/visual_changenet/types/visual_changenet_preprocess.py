# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet preprocessing configuration types.

This module provides dataclass definitions for configuring Visual ChangeNet preprocessing
in DeepStream/Inference Microservices applications.
"""

from typing import List

from dataclasses import dataclass, field, is_dataclass
from nvidia_tao_pytorch.core.types.nvdsinfer import BaseDSType


@dataclass
class VisualChangeNetPreprocessPropertyConfig(BaseDSType):
    """Configuration for Visual ChangeNet preprocessing in DeepStream/Inference Microservices.

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

    tensor_name: str = "input_1"
    processing_width: int = 224
    processing_height: int = 224
    network_input_shape: List[int] = field(default_factory=lambda: [2, 3, 224, 224])
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
class VisualChangeNetUserConfigs(BaseDSType):
    """User configuration for Visual ChangeNet preprocessing.

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
class VisualChangeNetGroupConfig(BaseDSType):
    """Configuration for Visual ChangeNet preprocessing in DeepStream/Inference Microservices.

    This class defines the preprocessing configuration for Visual ChangeNet inference,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.
    """

    src_ids: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14])
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
class VisualChangeNetPreprocessConfig(BaseDSType):
    """Configuration for Visual ChangeNet preprocessing in DeepStream/Inference Microservices.

    This class defines the preprocessing configuration for Visual ChangeNet inference,
    including input tensor specifications, image processing parameters, and custom
    preprocessing settings.
    """

    property: VisualChangeNetPreprocessPropertyConfig = field(
        default_factory=lambda: VisualChangeNetPreprocessPropertyConfig()
    )
    user_configs: VisualChangeNetUserConfigs = field(
        default_factory=lambda: VisualChangeNetUserConfigs()
    )
    group: VisualChangeNetGroupConfig = field(
        default_factory=lambda: VisualChangeNetGroupConfig()
    )

    def validate(self):
        """Validates the preprocessing configuration.

        Currently a placeholder for future validation logic.
        """
        pass


if __name__ == "__main__":
    visual_changenet_config = VisualChangeNetPreprocessConfig()
    assert is_dataclass(visual_changenet_config), "The instance of visual_changenet_config is not a dataclass."
    print(str(visual_changenet_config))
