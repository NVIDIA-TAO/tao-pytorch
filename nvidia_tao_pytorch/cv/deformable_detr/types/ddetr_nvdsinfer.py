# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GST-Nvinfer config file for DDETR."""

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseDSType,
    BaseNVDSClassAttributes,
    BaseNvDSPropertyConfig
)


@dataclass
class DDETRNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for RT-DETR."""

    parse_bbox_func_name: str = "NvDsInferParseCustomDDETRTAO"
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"


@dataclass
class DDETRNvDSClassAttribute(BaseNVDSClassAttributes):
    """Structured configuration defining the schema for nvdsinfer class-attr element for RT-DETR."""

    topk: int = 20


@dataclass
class DDETRNvDSInferConfig(BaseDSType):
    """RTDETRNvDSInfer config element."""

    property_field: DDETRNvDSPropertyConfig = field(default_factory=lambda: DDETRNvDSPropertyConfig(
        cluster_mode=4,
        net_scale_factor=0.0173520735728,
        network_type=0,
        network_mode=2,
        output_blob_names=["pred_boxes", "pred_logits"],
        model_color_format=0
    ))
    class_attrs_all: DDETRNvDSClassAttribute = field(default_factory=lambda: DDETRNvDSClassAttribute())

    def validate(self):
        """Function to validate the dataclass."""
        pass


if __name__ == "__main__":
    ddetr_config = DDETRNvDSInferConfig(
        cluster_mode=4,
    )
    assert is_dataclass(ddetr_config), "The instance of base_config is not a dataclass."
    print(str(ddetr_config))
