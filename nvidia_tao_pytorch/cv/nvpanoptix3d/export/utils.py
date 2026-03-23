# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for NVPanoptix3D Model."""

import torch


def load_2d_model(model: torch.nn.Module, checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a checkpoint into the model, handling common Lightning/DDP prefixes.

    Attempts to load weights using several common key-prefix conventions
    (``model.``, ``module.``, etc.). For ``MaskFormerModelWrapper`` instances
    the state dict is additionally split between the inner ``model`` and
    ``projector`` sub-modules.

    Args:
        model: Target model to load weights into
        checkpoint_path: Path to the checkpoint file (.pth or .ckpt)
        device: Device to map the checkpoint tensors onto (e.g. "cpu" or "cuda")

    Returns:
        The model with weights loaded in-place
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Try common wrapper prefixes (Lightning, DDP, etc.)
    for prefix in ["model.", "module.", "model.module.", "module.model.", ""]:
        state_stripped = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        } if prefix else state

        # Check if model has separate model and projector submodules (MaskFormerModelWrapper)
        if hasattr(model, "model") and hasattr(model, "projector"):
            # Split weights: projector.* goes to projector, everything else goes to model
            projector_state = {k[10:]: v for k, v in state_stripped.items() if k.startswith("projector.")}
            model_state = {k: v for k, v in state_stripped.items() if not k.startswith("projector.")}

            if model_state or projector_state:
                if model_state:
                    _, _ = model.model.load_state_dict(model_state, strict=False)
                if projector_state:
                    _, _ = model.projector.load_state_dict(projector_state, strict=False)
                return model

    return model
