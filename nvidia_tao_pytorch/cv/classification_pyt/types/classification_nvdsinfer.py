# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GST-Nvinfer config file for RT-DETR."""

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class ClassificationNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for Classification."""

    classification_threshold: float = 0.5

    def validate(self):
        """Validate the NVConfig."""
        super().validate()
        assert self.cluster_mode == 4, (
            "Cluster mode should be 4 since this is strictly a classification model"
        )
        assert self.network_type == 1, (
            "Network type should be 1, since this is a classification model."
        )


@dataclass
class ClassificationNvDSInferConfig(BaseDSType):
    """ClassificationNvDSInfer config element."""

    property_field: ClassificationNvDSPropertyConfig = field(default_factory=lambda: ClassificationNvDSPropertyConfig(
        cluster_mode=4,
        gie_unique_id=1,
        net_scale_factor=0.0173520735728,
        offsets=[123.675, 116.28, 103.53],
        network_type=1,
        network_mode=1,
        output_blob_names=None,
        model_color_format=0,
        classification_threshold=0.0,
    ))

    def validate(self):
        """Function to validate the dataclass."""
        self.property_field.validate()


if __name__ == "__main__":
    classification_config = ClassificationNvDSInferConfig()
    assert is_dataclass(classification_config), "The instance of base_config is not a dataclass."
    print(str(classification_config))
