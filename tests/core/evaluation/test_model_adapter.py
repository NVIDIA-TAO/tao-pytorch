# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the (summary, features) model-adapter contract."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.evaluation import ADAPTER_REGISTRY, build_adapter
from nvidia_tao_pytorch.core.evaluation.model_adapter import (
    BackboneV2Adapter,
    ModelAdapter,
    features_to_map,
    load_tao_state_dict,
)


@pytest.mark.unit
def test_features_to_map_reshapes_tokens():
    """[B,T,D] tokens reshape to [B,D,h,w] using ceil(H/patch)."""
    feats = torch.randn(2, 16, 8)            # 16 = 4x4 patches
    images = torch.randn(2, 3, 64, 64)       # 64/16 = 4
    out = features_to_map(feats, images, patch_size=16)
    assert out.shape == (2, 8, 4, 4)


@pytest.mark.unit
def test_features_to_map_idempotent_on_bchw():
    """An already-spatial [B,C,H,W] map passes through unchanged."""
    spatial = torch.randn(2, 8, 4, 4)
    out = features_to_map(spatial, torch.randn(2, 3, 64, 64), patch_size=16)
    assert out.shape == spatial.shape
    assert torch.allclose(out, spatial)


@pytest.mark.unit
def test_model_adapter_is_abstract():
    """ModelAdapter cannot be instantiated directly (abstract forward)."""
    with pytest.raises(TypeError):
        ModelAdapter()  # pylint: disable=abstract-class-instantiated


@pytest.mark.unit
def test_backbone_v2_adapter_passthrough():
    """BackboneV2Adapter forwards to backbone(return_features=True) → (summary, features)."""

    class _FakeBackboneV2(nn.Module):
        patch_size = 16

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 12, 16, stride=16)

        def forward(self, x, return_features=False, return_logits=True):
            f = self.conv(x)
            return f.mean((2, 3)), f

    adapter = BackboneV2Adapter(_FakeBackboneV2(), feature_dim=12).eval()
    summary, feats = adapter(torch.randn(2, 3, 64, 64))
    assert summary.shape == (2, 12)
    assert feats.shape == (2, 12, 4, 4)


@pytest.mark.unit
def test_load_tao_state_dict_strip_and_rename(tmp_path):
    """TAO ckpt loader strips model.radio.radio. and renames ls*.gamma → grandma.

    The gamma → grandma rename is intentional: it mirrors upstream RADIO's
    hubconf.py (which renames the LayerScale ``gamma`` parameter to ``grandma``)
    so the re-keyed state dict loads cleanly against a torch.hub RADIO model.
    See ``load_tao_state_dict`` for the full rationale.
    """
    sd = {
        "model.radio.radio.blocks.0.ls1.gamma": torch.zeros(4),
        "model.radio.radio.blocks.0.ls2.gamma": torch.zeros(4),
        "model.radio.radio.patch_embed.weight": torch.zeros(2),
        "teacher.ignore_me": torch.zeros(1),
    }
    path = tmp_path / "ckpt.pth"
    torch.save({"state_dict": sd}, path)
    out = load_tao_state_dict(str(path))
    assert set(out) == {
        "blocks.0.ls1.grandma", "blocks.0.ls2.grandma", "patch_embed.weight",
    }


@pytest.mark.unit
def test_mae_adapter_contract():
    """MAEViTAdapter returns (CLS summary, unshuffled patch features) with correct shapes."""
    from nvidia_tao_pytorch.ssl.mae.model.mae import MaskedAutoencoderViT

    assert "mae" in ADAPTER_REGISTRY
    model = MaskedAutoencoderViT(
        img_size=64, patch_size=16, embed_dim=48, depth=2, num_heads=4,
        decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4, mlp_ratio=4.0,
    ).eval()
    adapter = build_adapter("mae", model)
    assert adapter.patch_size == 16 and adapter.feature_dim == 48

    x = torch.randn(2, 3, 64, 64)
    summary, feats = adapter(x)
    assert summary.shape == (2, 48)
    assert feats.shape == (2, 16, 48)              # 4x4 patches
    assert features_to_map(feats, x, adapter.patch_size).shape == (2, 48, 4, 4)
    # eval (mask_ratio=0) is deterministic — no stochastic masking leak
    assert torch.allclose(summary, adapter(x)[0], atol=1e-5)


@pytest.mark.unit
def test_dinov3_adapter_contract():
    """The dinov3 registry entry reuses DinoV2Adapter (same teacher-backbone token dict)."""
    from nvidia_tao_pytorch.core.evaluation.model_adapter import DinoV2Adapter

    assert ADAPTER_REGISTRY["dinov3"] is DinoV2Adapter

    class _FakeBackbone(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return {
                "x_norm_clstoken": torch.randn(b, 48),
                "x_norm_patchtokens": torch.randn(b, 16, 48),
            }

    pl_model = nn.Module()
    pl_model.teacher = nn.ModuleDict({"backbone": _FakeBackbone()})
    adapter = build_adapter("dinov3", pl_model, patch_size=16, feature_dim=48)
    assert adapter.patch_size == 16 and adapter.feature_dim == 48

    x = torch.randn(2, 3, 64, 64)
    summary, feats = adapter(x)
    assert summary.shape == (2, 48)
    assert feats.shape == (2, 16, 48)
    assert features_to_map(feats, x, adapter.patch_size).shape == (2, 48, 4, 4)


@pytest.mark.unit
def test_build_adapter_unknown_network():
    """Unknown network raises a helpful KeyError."""
    with pytest.raises(KeyError):
        build_adapter("does_not_exist", nn.Identity())
