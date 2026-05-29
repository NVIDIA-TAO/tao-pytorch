# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet DeepStream inference configuration types.

This module provides dataclass definitions for configuring Visual ChangeNet inference
in DeepStream/Inference Microservices applications. It includes configurations for properties,
class attributes, and overall inference settings.
"""

from dataclasses import dataclass, field, is_dataclass
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class VisualChangeNetNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Configuration for Visual ChangeNet inference properties in DeepStream/Inference Microservices.

    This class defines the schema for nvdsinfer property element specific to Visual ChangeNet.
    It extends BaseNVDSPropertyConfig with Visual ChangeNet-specific configurations.

    Attributes:
        classifier_threshold (float): Confidence threshold for classification predictions.
            Defaults to 0.5.
        input_tensor_from_meta (int): Flag to determine input tensor source.
            Must be either 0 or 1. Defaults to 1.
        parse_classifier_func_name (str): Name of the classifier parsing function.
            Defaults to "NvDsInferClassiferParseNonSoftmax".
        custom_lib_path (str): Path to the custom parser library.
            Defaults to "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so".
    """

    classifier_threshold: float = 0.5
    input_tensor_from_meta: int = 1
    parse_classifier_func_name: str = "NvDsInferClassiferParseNonSoftmax"
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"

    def validate(self):
        """Validates the configuration parameters.

        Ensures that input_tensor_from_meta is either 0 or 1 and calls the parent class's
        validation method.

        Raises:
            AssertionError: If input_tensor_from_meta is not 0 or 1.
        """
        assert self.input_tensor_from_meta in [0, 1], "input_tensor_from_meta must be 0 or 1"
        super().validate()


@dataclass
class VisualChangeNetNvDSInferConfig(BaseDSType):
    """Main configuration class for Visual ChangeNet inference in DeepStream/Inference Microservices.

    This class combines all necessary configurations for Visual ChangeNet inference,
    including property configurations and class attributes. It provides a complete
    configuration structure for the inference pipeline.

    Attributes:
        property_field (VisualChangeNetNvDSPropertyConfig): Configuration for inference properties.
            Defaults to a pre-configured instance with the following settings:
            - cluster_mode: 2
            - classifier_threshold: 0.5
            - offsets: [127.5, 127.5, 127.5]
            - net_scale_factor: 0.00392156862745098
            - input_tensor_from_meta: 1
            - network_type: 1
            - network_mode: 2
            - num_detected_classes: 2
            - model_color_format: 0
    """

    property_field: VisualChangeNetNvDSPropertyConfig = field(
        default_factory=lambda: VisualChangeNetNvDSPropertyConfig(
            cluster_mode=2,
            classifier_threshold=0.5,
            offsets=[127.5, 127.5, 127.5],
            net_scale_factor=0.00392156862745098,
            input_tensor_from_meta=1,
            network_type=1,
            network_mode=2,
            num_detected_classes=2,
            model_color_format=0
        )
    )

    def validate(self):
        """Validates the configuration.

        Currently a placeholder for future validation logic.
        """
        pass


if __name__ == "__main__":
    visual_changenet_config = VisualChangeNetNvDSInferConfig()
    assert is_dataclass(visual_changenet_config), "The instance of visual_changenet_config is not a dataclass."
    print(str(visual_changenet_config))
