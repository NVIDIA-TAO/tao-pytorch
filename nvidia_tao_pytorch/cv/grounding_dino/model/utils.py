# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO model utils. """
from nvidia_tao_pytorch.core.utils.ptm_utils import StateDictAdapter

ptm_adapter = StateDictAdapter()
ptm_adapter.add("mae", "model.encoder.")
ptm_adapter.add("classification", "model.")
ptm_adapter.add("grounding_dino", "model.")


def grounding_dino_parser(original):
    """Parse public Grounding DINO checkpoints.

    Download checkpoints from https://github.com/IDEA-Research/GroundingDINO/releases.
    """
    final = {}
    for k, v in original.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        if k.startswith('backbone.0'):
            k = f"model.model.{k}"
            k = k.replace("backbone.0", "backbone.0.body")
        elif k == "bert.embeddings.position_ids":
            continue
        elif "label_enc.weight" in k:
            continue
        else:
            k = f"model.model.{k}"
        final[k] = v
    return final
