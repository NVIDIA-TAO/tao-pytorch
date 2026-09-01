# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for the Qwen reasoning module."""

from __future__ import annotations

import torch
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.qwen_vlm import (
    DEFAULT_MODEL_ID,
    Qwen3VLReasoner,
)


def cfg_get(cfg, key, default=None):
    """Get a value from OmegaConf, argparse-style objects, or dicts."""
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def build_qwen(qwen_cfg) -> Qwen3VLReasoner:
    """Build the Qwen reasoner from the ``model.reasoning.qwen`` config block."""
    lora_cfg = cfg_get(qwen_cfg, "lora", {})
    dtype_str = str(cfg_get(qwen_cfg, "dtype", "float32")).lower()
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype_str, torch.float32)
    return Qwen3VLReasoner(
        model_id=cfg_get(qwen_cfg, "model_id", DEFAULT_MODEL_ID),
        seg_token=cfg_get(qwen_cfg, "seg_token", "[SEG]"),
        freeze_vision=bool(cfg_get(qwen_cfg, "freeze_vision", True)),
        dtype=dtype,
        attn_implementation=cfg_get(qwen_cfg, "attn_implementation", "sdpa"),
        lora_r=int(cfg_get(lora_cfg, "r", 16)),
        lora_alpha=int(cfg_get(lora_cfg, "alpha", 32)),
        lora_dropout=float(cfg_get(lora_cfg, "dropout", 0.05)),
        use_lora=bool(cfg_get(qwen_cfg, "use_lora", True)),
    )
