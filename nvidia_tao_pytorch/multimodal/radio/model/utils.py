# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Misc functions. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter


ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("rtdetr", "model.model.backbone.")
ptm_adapter.add("mask2former", "model.backbone.")
ptm_adapter.add("mal", "student.backbone.")
ptm_adapter.add("mask_grounding_dino", "model.model.backbone.0.body.")
ptm_adapter.add("grounding_dino", "model.model.backbone.0.body.")
ptm_adapter.add("dino", "model.model.backbone.0.body.")
ptm_adapter.add("visual_changenet_classify", "model.backbone.")
ptm_adapter.add("visual_changenet_classify", "model.backbone.radio.", model_type="radio_learnable")
ptm_adapter.add("visual_changenet_segment", "model.backbone.")
ptm_adapter.add("visual_changenet_segment", "model.backbone.radio.", model_type="radio")


def cls_parser(original):
    """Parse public classification checkpoints."""
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
        else:
            state_dict[key] = value
    return state_dict
