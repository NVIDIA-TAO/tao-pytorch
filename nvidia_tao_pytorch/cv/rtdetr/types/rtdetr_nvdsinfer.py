# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GST-Nvinfer config file for RT-DETR."""

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNVDSClassAttributes,
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class RTDETRNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for RT-DETR."""

    parse_bbox_func_name: str = "NvDsInferParseCustomDDETRTAO"
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"


@dataclass
class RTDETRNvDSClassAttribute(BaseNVDSClassAttributes):
    """Structured configuration defining the schema for nvdsinfer class-attr element for RT-DETR."""

    topk: int = 20


@dataclass
class RTDETRNvDSInferConfig(BaseDSType):
    """RTDETRNvDSInfer config element."""

    class_attrs_all: RTDETRNvDSClassAttribute = field(default_factory=lambda: RTDETRNvDSClassAttribute())
    property_field: RTDETRNvDSPropertyConfig = field(default_factory=lambda: RTDETRNvDSPropertyConfig(
        cluster_mode=4,
        offsets=[0, 0, 0],
        net_scale_factor=1 / 255.0,
        network_type=0,
        network_mode=2,
        output_blob_names=["pred_boxes", "pred_logits"],
        model_color_format=0
    ))

    def validate(self):
        """Function to validate the dataclass."""
        pass


if __name__ == "__main__":
    rtdetr_config = RTDETRNvDSInferConfig()
    assert is_dataclass(rtdetr_config), "The instance of base_config is not a dataclass."
    print(str(rtdetr_config))
