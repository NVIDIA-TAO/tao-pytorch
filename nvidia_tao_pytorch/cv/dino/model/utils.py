# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINO model utils. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter

ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("dino", "model.")


def dino_parser(original):
    """Parse public DINO checkpoints."""
    state_dict = {}
    for key, value in list(original.items()):
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        elif key.startswith("backbone."):
            # MMLab compatible weight loading
            new_key = key[9:]
            state_dict[new_key] = value
        elif key.startswith("model.model.backbone."):
            new_key = key[len("model.model.backbone."):]
            state_dict[new_key] = value
        elif key.startswith("model.backbone."):
            new_key = key[len("model.backbone."):]
            state_dict[new_key] = value
        elif key.startswith("model.encoder."):
            new_key = key[len("model.encoder."):]
            state_dict[new_key] = value
        elif key.startswith("model.decoder."):
            new_key = key[len("model.decoder."):]
            state_dict[new_key] = value
        elif key.startswith("model."):
            # MAE compatible weight loading
            new_key = key[len("model."):]
            state_dict[new_key] = value
        elif key.startswith("ema_"):
            # Do not include ema params from MMLab
            continue
        elif 'grn' in key:
            # Reshape GRN parameters from 6D to 4D if needed
            if value.dim() == 6:  # If parameter is 6D [1, 1, 1, 1, 1, C]
                state_dict[key] = value.squeeze(3).squeeze(3)  # Reshape to 4D [1, 1, 1, C]
            elif value.dim() == 2:
                state_dict[key] = value.unsqueeze(0).unsqueeze(1)

        else:
            state_dict[key] = value
    return state_dict
