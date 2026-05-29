# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for RADIO model
"""

from typing import Dict, Any


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    """
    Function to parse target parameters from pretrained radio weights

    Args:
        state_dict (dict): pretrained pytorch model weights
        prefix (str): prefix used to parse target weights
    """
    mod_state_dict = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict
