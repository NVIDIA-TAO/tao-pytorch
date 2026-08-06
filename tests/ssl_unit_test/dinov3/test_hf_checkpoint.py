# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading HuggingFace-format DINOv3 checkpoints.

The HF ``DINOv3ViTModel`` export holds the same weights as the timm release but serializes
them differently: ``embeddings.*`` / ``layer.N.*`` names, and attention stored as separate
``q_proj``/``k_proj``/``v_proj`` rather than a fused ``qkv``. Under the timm rules alone exactly
two keys coincide (``norm.weight``/``norm.bias``), so an HF checkpoint used to load 2 of 211
tensors and train from a near-random backbone while still logging a successful load.
"""

import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    convert_hf_to_tao,
    hf_to_tao,
    is_hf_dinov3_state_dict,
)

EMBED_DIM = 64
N_BLOCKS = 2


def _fake_hf_state_dict(n_blocks=N_BLOCKS, dim=EMBED_DIM, qv_bias_value=0.0):
    """Build a minimal state dict in HuggingFace ``DINOv3ViTModel`` layout."""
    torch.manual_seed(0)
    sd = {
        "embeddings.cls_token": torch.randn(1, 1, dim),
        "embeddings.mask_token": torch.randn(1, 1, dim),
        "embeddings.register_tokens": torch.randn(1, 4, dim),
        "embeddings.patch_embeddings.weight": torch.randn(dim, 3, 16, 16),
        "embeddings.patch_embeddings.bias": torch.randn(dim),
        "norm.weight": torch.randn(dim),
        "norm.bias": torch.randn(dim),
    }
    for b in range(n_blocks):
        p = f"layer.{b}."
        for proj in ("q_proj", "k_proj", "v_proj"):
            sd[f"{p}attention.{proj}.weight"] = torch.randn(dim, dim)
        for proj in ("q_proj", "v_proj"):
            sd[f"{p}attention.{proj}.bias"] = torch.full((dim,), qv_bias_value)
        sd[f"{p}attention.o_proj.weight"] = torch.randn(dim, dim)
        sd[f"{p}attention.o_proj.bias"] = torch.randn(dim)
        sd[f"{p}layer_scale1.lambda1"] = torch.randn(dim)
        sd[f"{p}layer_scale2.lambda1"] = torch.randn(dim)
        sd[f"{p}mlp.up_proj.weight"] = torch.randn(4 * dim, dim)
        sd[f"{p}mlp.up_proj.bias"] = torch.randn(4 * dim)
        sd[f"{p}mlp.down_proj.weight"] = torch.randn(dim, 4 * dim)
        sd[f"{p}mlp.down_proj.bias"] = torch.randn(dim)
        sd[f"{p}norm1.weight"] = torch.randn(dim)
        sd[f"{p}norm1.bias"] = torch.randn(dim)
        sd[f"{p}norm2.weight"] = torch.randn(dim)
        sd[f"{p}norm2.bias"] = torch.randn(dim)
    return sd


@pytest.mark.ssl_unit
def test_hf_format_detection():
    """HF exports are recognized; timm ones are not."""
    assert is_hf_dinov3_state_dict(_fake_hf_state_dict())
    timm_like = {"cls_token": torch.zeros(1), "blocks.0.attn.qkv.weight": torch.zeros(1),
                 "norm.weight": torch.zeros(1)}
    assert not is_hf_dinov3_state_dict(timm_like)


@pytest.mark.ssl_unit
def test_hf_key_renames():
    """Every HF name maps onto its TAO counterpart."""
    cases = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "register_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "layer.3.attention.o_proj.bias": "blocks.3.attn.proj.bias",
        "layer.3.layer_scale1.lambda1": "blocks.3.ls1.gamma",
        "layer.3.layer_scale2.lambda1": "blocks.3.ls2.gamma",
        "layer.3.mlp.up_proj.weight": "blocks.3.mlp.fc1.weight",
        "layer.3.mlp.down_proj.bias": "blocks.3.mlp.fc2.bias",
        "layer.3.norm1.weight": "blocks.3.norm1.weight",
        "norm.weight": "norm.weight",
    }
    for src, want in cases.items():
        assert hf_to_tao(src) == want, f"{src} -> {hf_to_tao(src)}, expected {want}"


@pytest.mark.ssl_unit
def test_hf_qkv_fusion_is_concatenation():
    """q/k/v fuse into one ``attn.qkv`` in q,k,v order — the transform a rename cannot do."""
    hf = _fake_hf_state_dict()
    tao = convert_hf_to_tao(hf)

    fused = tao["blocks.0.attn.qkv.weight"]
    assert fused.shape == (3 * EMBED_DIM, EMBED_DIM)
    q, k, v = (hf[f"layer.0.attention.{p}.weight"] for p in ("q_proj", "k_proj", "v_proj"))
    torch.testing.assert_close(fused, torch.cat([q, k, v], dim=0), rtol=0, atol=0)

    # The separate projections must not survive alongside the fused tensor.
    assert not [key for key in tao if "q_proj" in key or "k_proj" in key or "v_proj" in key]


@pytest.mark.ssl_unit
def test_hf_zero_qv_bias_is_dropped():
    """DINOv3 ships all-zero q/v biases, which TAO's bias-free attention simply drops."""
    tao = convert_hf_to_tao(_fake_hf_state_dict(qv_bias_value=0.0))
    assert not [k for k in tao if k.endswith("attn.qkv.bias")]
    # The output projection's bias, which TAO *does* have, is preserved.
    assert "blocks.0.attn.proj.bias" in tao


@pytest.mark.ssl_unit
def test_hf_nonzero_qv_bias_refuses_to_convert():
    """A nonzero q/v bias cannot be represented, so it must fail loudly rather than vanish."""
    with pytest.raises(ValueError, match="nonzero"):
        convert_hf_to_tao(_fake_hf_state_dict(qv_bias_value=0.5))


@pytest.mark.ssl_unit
def test_hf_conversion_covers_every_key():
    """No HF tensor is silently lost apart from the deliberately dropped zero biases."""
    hf = _fake_hf_state_dict()
    tao = convert_hf_to_tao(hf)
    dropped = [k for k in hf if k.endswith(("q_proj.bias", "v_proj.bias"))]
    fused_away = [k for k in hf if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))]
    # every source key is either renamed, fused, or an intentionally dropped bias
    assert len(tao) == len(hf) - len(dropped) - len(fused_away) + N_BLOCKS
    assert "mask_token" in tao, "HF ships mask_token; timm does not, so it must survive"
