# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 2D axial RoPE unit tests (CPU; pure-torch, no GPU/xformers needed).

Covers the rotation math and, critically, that the ``[CLS]`` token and trailing register
tokens are *not* rotated (identity rows) while patch tokens are.
"""
import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.layers.rope import RoPE2D, apply_rope, rotate_half

HEAD_DIM = 64
NUM_REGISTERS = 4


@pytest.mark.ssl_unit
def test_rotate_half():
    """rotate_half splits the last dim in half: [a, b] -> [-b, a]."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = rotate_half(x)
    assert torch.equal(out, torch.tensor([-3.0, -4.0, 1.0, 2.0]))


@pytest.mark.ssl_unit
def test_apply_rope_identity_is_noop():
    """sin=0, cos=1 (the identity rotation) leaves the tensor unchanged."""
    x = torch.randn(2, 5, 3, HEAD_DIM)
    sin = torch.zeros(5, HEAD_DIM)
    cos = torch.ones(5, HEAD_DIM)
    out = apply_rope(x, sin, cos)
    assert torch.allclose(out, x, atol=1e-6)


@pytest.mark.ssl_unit
def test_apply_rope_preserves_norm():
    """RoPE is an orthogonal rotation, so it preserves the per-token vector norm."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, 3, HEAD_DIM)
    # Arbitrary non-trivial angles duplicated across the two halves (rotate_half convention).
    half = torch.randn(7, HEAD_DIM // 2)
    emb = torch.cat([half, half], dim=-1)
    out = apply_rope(x, emb.sin(), emb.cos())
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


@pytest.mark.ssl_unit
def test_rope2d_excludes_cls_and_registers():
    """CLS (index 0) and the trailing register rows must be identity; patches must rotate."""
    H = W = 8
    rope = RoPE2D(head_dim=HEAD_DIM, theta=100.0, num_prefix_tokens=1,
                  num_register_tokens=NUM_REGISTERS)
    sin, cos = rope(H, W, device=torch.device("cpu"), dtype=torch.float32)

    n_patches = H * W
    seq_len = 1 + n_patches + NUM_REGISTERS
    assert sin.shape == (seq_len, HEAD_DIM)
    assert cos.shape == (seq_len, HEAD_DIM)

    # [CLS] at index 0 is identity.
    assert torch.allclose(sin[0], torch.zeros(HEAD_DIM))
    assert torch.allclose(cos[0], torch.ones(HEAD_DIM))

    # Trailing register tokens are identity.
    reg = slice(1 + n_patches, seq_len)
    assert torch.allclose(sin[reg], torch.zeros(NUM_REGISTERS, HEAD_DIM))
    assert torch.allclose(cos[reg], torch.ones(NUM_REGISTERS, HEAD_DIM))

    # Patch tokens are rotated: position 0 of the grid has zero angle (still identity), but
    # later patches must differ from identity.
    patch = slice(1, 1 + n_patches)
    assert not torch.allclose(cos[patch], torch.ones(n_patches, HEAD_DIM)), \
        "patch tokens should carry non-identity rotation"


@pytest.mark.ssl_unit
def test_rope2d_requires_head_dim_divisible_by_four():
    """2D axial RoPE needs head_dim % 4 == 0 (two axes, rotary pairs each)."""
    with pytest.raises(AssertionError):
        RoPE2D(head_dim=66, theta=100.0)


@pytest.mark.ssl_unit
def test_rope2d_grid_distinguishes_positions():
    """Different patch positions get different rotation tables."""
    H = W = 4
    rope = RoPE2D(head_dim=HEAD_DIM, theta=100.0, num_prefix_tokens=1, num_register_tokens=0)
    sin, cos = rope(H, W, device=torch.device("cpu"), dtype=torch.float32)
    # patch index 1 (grid (0,1)) vs patch index 5 (grid (1,1)) should differ.
    assert not torch.allclose(sin[1 + 1], sin[1 + 5])
