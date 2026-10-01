# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for NVPanoptix3D Model."""

import torch

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.attention import (
    Attention,
    MemEffAttention,
)


def patch_xformers_attention_for_export(model: torch.nn.Module) -> None:
    """Patch xformers attention modules for ONNX export tracing.

    xformers memory_efficient_attention relies on CUTLASS ops with
    c10::SymInt arguments that the legacy JIT tracer cannot export.
    For each MemEffAttention module in the model, this function replaces
    ``forward`` with ``Attention.forward`` so export uses the
    scaled-dot-product-attention path that is traceable.

    This helper is shared by the NVPanoptix3D and NVPanoptix3Dv2 ONNX
    exporters so both trace the same attention path.

    Args:
        model: Model tree whose MemEffAttention modules are patched in-place.
    """
    for module in model.modules():
        if isinstance(module, MemEffAttention):
            module.forward = Attention.forward.__get__(module, Attention)


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
