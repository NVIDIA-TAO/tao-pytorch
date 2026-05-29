# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO ViT Model Module"""

from nvidia_tao_pytorch.cv.backbone_v2.radio import (
    c_radio_p1_vit_huge_patch16_mlpnorm,
    c_radio_p2_vit_huge_patch16_mlpnorm,
    c_radio_p3_vit_huge_patch16_mlpnorm,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_huge_patch16,
    c_radio_v2_vit_large_patch16,
)


radio_model_dict = {
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": c_radio_p1_vit_huge_patch16_mlpnorm,
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": c_radio_p2_vit_huge_patch16_mlpnorm,
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": c_radio_p3_vit_huge_patch16_mlpnorm,
    "c_radio_v2_vit_huge_patch16_224": c_radio_v2_vit_huge_patch16,
    "c_radio_v2_vit_large_patch16_224": c_radio_v2_vit_large_patch16,
    "c_radio_v2_vit_base_patch16_224": c_radio_v2_vit_base_patch16,
}
