# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Misc functions."""

import torch

from nvidia_tao_pytorch.core.tlt_logging import logging


def load_pretrained_weights(pretrained_path):
    """Load pretrained weights from a checkpoint file."""
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = {}
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        logging.warning("Warning: Checkpoint does not contain 'state_dict' or 'model' key. Assuming it's the state_dict itself.")
        state_dict = checkpoint
    processed_state_dict = {}

    # load a model trained on tao pipeline
    if "criterion.instance_bank.anchor" in state_dict and "criterion.instance_bank.instance_feature" in state_dict:
        logging.info("Detected TAO pipeline checkpoint format.")
        return state_dict

    # load a model trained on mmcv pipeline
    if "head.instance_bank.anchor" in state_dict and "head.instance_bank.instance_feature" in state_dict:
        logging.info("Detected MMCV pipeline checkpoint format. Mapping keys...")
        for k, v in state_dict.items():
            if "instance_bank.anchor" in k:
                processed_state_dict["model." + k] = v
                processed_state_dict["criterion." + k.replace("head.", "")] = v
            elif "instance_bank.instance_feature" in k:
                processed_state_dict["model." + k] = v
                processed_state_dict["criterion." + k.replace("head.", "")] = v
            elif not k.startswith("model."):
                processed_state_dict["model." + k] = v
            else:
                processed_state_dict[k] = v
        return processed_state_dict

    # Heuristic check for backbone-only model (lacks common head/neck keys)
    is_likely_backbone_only = True
    for k in state_dict.keys():
        if k.startswith("head.") or k.startswith("neck.") or k.startswith("model.head.") or k.startswith("model.img_neck."):
            is_likely_backbone_only = False
            break

    # Load a backbone only model
    if is_likely_backbone_only:
        logging.info("Attempting to load as a backbone-only checkpoint.")
        processed_state_dict = {}
        prefix_to_remove = ""
        if any(k.startswith("model.") for k in state_dict.keys()):
            prefix_to_remove = "model."
        elif any(k.startswith("backbone.") for k in state_dict.keys()):
            prefix_to_remove = "backbone."
        target_prefix = "model.img_backbone."
        for k, v in state_dict.items():
            k_no_prefix = k
            if prefix_to_remove and k.startswith(prefix_to_remove):
                k_no_prefix = k[len(prefix_to_remove):]
            if k_no_prefix.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
                target_key = target_prefix + k_no_prefix
                processed_state_dict[target_key] = v
        if not processed_state_dict:
            logging.warning("Warning: No keys were mapped during backbone-only loading attempt. Checkpoint might be incompatible or empty.")
            return state_dict
        else:
            logging.info(f"Mapped {len(processed_state_dict)} keys for backbone.")
            return processed_state_dict
    return state_dict
