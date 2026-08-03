# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity of DINOv3 RoPE2D against timm's RotaryEmbeddingDinoV3 (CPU).

This is the cheap, deterministic guard on the highest-risk piece of the port: the rotary
frequency schedule, coordinate normalization, and sin/cos layout must match the timm
``vit_base_patch16_dinov3`` reference exactly. The full feature-parity smoke test
(``test_feature_parity.py``) needs the weights + a GPU; this one needs neither.
"""
import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.layers.rope import RoPE2D

# timm's DINOv3 rope lives in pos_embed_sincos; skip cleanly on older timm.
timm_rope = pytest.importorskip("timm.layers.pos_embed_sincos")
RotaryEmbeddingDinoV3 = getattr(timm_rope, "RotaryEmbeddingDinoV3", None)
requires_timm_rope = pytest.mark.skipif(
    RotaryEmbeddingDinoV3 is None, reason="timm too old: no RotaryEmbeddingDinoV3"
)

HEAD_DIM = 64
THETA = 100.0


@requires_timm_rope
@pytest.mark.ssl_unit
@pytest.mark.parametrize("H,W", [(16, 16), (8, 12)])
def test_rope2d_matches_timm(H, W):
    """RoPE2D's patch sin/cos tables equal timm RotaryEmbeddingDinoV3.get_embed."""
    ref = RotaryEmbeddingDinoV3(
        dim=HEAD_DIM, temperature=THETA, normalize_coords="separate",
        grid_indexing="ij", rotate_half=True,
    ).eval()
    emb = ref.get_embed((H, W))            # [H*W, 2*head_dim] = cat([sin, cos])
    sin_ref, cos_ref = emb.chunk(2, dim=-1)  # each [H*W, head_dim]

    rope = RoPE2D(head_dim=HEAD_DIM, theta=THETA, num_prefix_tokens=1, num_register_tokens=4)
    sin, cos = rope(H, W, device=torch.device("cpu"), dtype=torch.float32)

    # Patch rows live after the single [CLS] prefix; registers are the identity tail.
    sin_patch = sin[1:1 + H * W]
    cos_patch = cos[1:1 + H * W]

    assert torch.allclose(sin_patch, sin_ref, atol=1e-5), "RoPE sin table diverges from timm"
    assert torch.allclose(cos_patch, cos_ref, atol=1e-5), "RoPE cos table diverges from timm"


@requires_timm_rope
@pytest.mark.ssl_unit
def test_rope2d_periods_match_timm():
    """The period schedule matches timm's (theta ** (2i / (head_dim/2)))."""
    ref = RotaryEmbeddingDinoV3(dim=HEAD_DIM, temperature=THETA).eval()
    rope = RoPE2D(head_dim=HEAD_DIM, theta=THETA)
    assert torch.allclose(rope.periods, ref.periods, atol=1e-5)
