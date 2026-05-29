# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Model Parameters Mapping Module """

radio_model_cfg = {
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 3,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2, 3],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16
    },
    "c_radio_v2_vit_base_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    },
    "c_radio_v2_vit_large_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    },
    "c_radio_v2_vit_huge_patch16_224": {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8
    }
}
