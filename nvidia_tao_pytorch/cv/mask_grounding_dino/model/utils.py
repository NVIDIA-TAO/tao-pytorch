# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Misc functions. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter

ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("grounding_dino", "model.")


def mask_gdino_parser(original):
    """Parse public Mask Grounding DINO checkpoints."""
    state_dict = {}
    for key, value in list(original.items()):
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        elif key.startswith("backbone."):
            # MMLab compatible weight loading
            new_key = key[9:]
            state_dict[new_key] = value
        elif key.startswith("ema_"):
            # Do not include ema params from MMLab
            continue
        else:
            state_dict[key] = value
    return state_dict
