# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone Init Module."""

from nvidia_tao_pytorch.cv.backbone_v2.mit import (
    mit_b0,
    mit_b1,
    mit_b2,
    mit_b3,
    mit_b4,
    mit_b5,
)
from nvidia_tao_pytorch.cv.segformer.model.backbones.dino_v2 import vit_giant_nvdinov2, vit_large_nvdinov2
from nvidia_tao_pytorch.cv.segformer.model.backbones.fan import (
    fan_base_16_p4_hybrid,
    fan_large_16_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_tiny_8_p4_hybrid,
)
from nvidia_tao_pytorch.cv.segformer.model.backbones.open_clip import (
    vit_base_nvclip_16_siglip,
    vit_huge_nvclip_14_siglip,
)
from nvidia_tao_pytorch.cv.segformer.model.backbones.radio import (
    c_radio_v2_vit_base_patch16_224,
    c_radio_v2_vit_huge_patch16_224,
    c_radio_v2_vit_large_patch16_224,
    c_radio_v3_vit_large_patch16_reg4_dinov2,
)


vit_adapter_model_dict = {
    "vit_large_nvdinov2": vit_large_nvdinov2,
    "vit_giant_nvdinov2": vit_giant_nvdinov2,
    "vit_base_nvclip_16_siglip": vit_base_nvclip_16_siglip,
    "vit_huge_nvclip_14_siglip": vit_huge_nvclip_14_siglip,
}

cradio_vit_adapter_model_dict = {
    "c_radio_v2_vit_huge_patch16_224": c_radio_v2_vit_huge_patch16_224,
    "c_radio_v2_vit_large_patch16_224": c_radio_v2_vit_large_patch16_224,
    "c_radio_v2_vit_base_patch16_224": c_radio_v2_vit_base_patch16_224,
    "c_radio_v3_vit_large_patch16_reg4_dinov2": c_radio_v3_vit_large_patch16_reg4_dinov2,
}

fan_model_dict = {
    "fan_tiny_8_p4_hybrid": fan_tiny_8_p4_hybrid,
    "fan_small_12_p4_hybrid": fan_small_12_p4_hybrid,
    "fan_base_16_p4_hybrid": fan_base_16_p4_hybrid,
    "fan_large_16_p4_hybrid": fan_large_16_p4_hybrid,
}

mit_model_dict = {
    "mit_b0": mit_b0,
    "mit_b1": mit_b1,
    "mit_b2": mit_b2,
    "mit_b3": mit_b3,
    "mit_b4": mit_b4,
    "mit_b5": mit_b5,
}
