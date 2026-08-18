# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 checkpoint-remap unit tests: SwiGLU fc1 fuse/split round-trip (ViT-H+/7B)."""
import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    extract_backbone_state_dict,
    fuse_timm_swiglu_fc1,
    is_full_checkpoint,
    split_fused_swiglu_fc1,
)


def _swiglu_split_state_dict(in_dim=8, hidden=6, n_blocks=2):
    """Build a tiny timm-style SwiGLU state dict (split fc1_g/fc1_x) plus some non-MLP keys."""
    sd = {"cls_token": torch.randn(1, 1, in_dim)}
    for b in range(n_blocks):
        p = f"blocks.{b}.mlp."
        sd[p + "fc1_g.weight"] = torch.randn(hidden, in_dim)
        sd[p + "fc1_g.bias"] = torch.randn(hidden)
        sd[p + "fc1_x.weight"] = torch.randn(hidden, in_dim)
        sd[p + "fc1_x.bias"] = torch.randn(hidden)
        sd[p + "fc2.weight"] = torch.randn(in_dim, hidden)
        sd[p + "fc2.bias"] = torch.randn(in_dim)
        sd[f"blocks.{b}.norm1.weight"] = torch.randn(in_dim)  # non-MLP passthrough
    return sd


@pytest.mark.ssl_unit
@pytest.mark.parametrize("wrapped", [False, True])
def test_full_checkpoint_detection_and_teacher_extraction(wrapped):
    """Full-checkpoint detection and extraction share the same wrapped-state handling."""
    teacher_weight = torch.ones(1)
    state_dict = {
        "student.backbone.cls_token": torch.zeros(1),
        "teacher.backbone.cls_token": teacher_weight,
        "teacher.dino_head.weight": torch.randn(1),
    }
    checkpoint = {"state_dict": state_dict} if wrapped else state_dict

    assert is_full_checkpoint(checkpoint)
    extracted = extract_backbone_state_dict(checkpoint, source="teacher")
    assert set(extracted) == {"cls_token"}
    assert torch.equal(extracted["cls_token"], teacher_weight)


@pytest.mark.ssl_unit
def test_stripped_checkpoint_is_not_full():
    """A stripped backbone checkpoint stays on the existing restore path."""
    checkpoint = {"cls_token": torch.ones(1), "mask_token": torch.zeros(1)}

    assert not is_full_checkpoint(checkpoint)
    extracted = extract_backbone_state_dict(checkpoint)
    assert set(extracted) == set(checkpoint)
    assert all(torch.equal(extracted[key], checkpoint[key]) for key in checkpoint)


@pytest.mark.ssl_unit
def test_swiglu_fc1_fuse_is_gate_first():
    """Fusing concatenates [fc1_g, fc1_x] along dim 0 (gate first), doubling fc1 width."""
    torch.manual_seed(0)
    orig = _swiglu_split_state_dict()
    fused = fuse_timm_swiglu_fc1(orig)

    # No split projections remain; fused fc1 present at 2x width.
    assert not any(k.endswith(("fc1_g.weight", "fc1_x.weight")) for k in fused)
    for b in range(2):
        p = f"blocks.{b}.mlp."
        h = orig[p + "fc1_g.weight"].shape[0]
        for suffix in ("weight", "bias"):
            assert fused[p + f"fc1.{suffix}"].shape[0] == 2 * h
            assert torch.equal(fused[p + f"fc1.{suffix}"][:h], orig[p + f"fc1_g.{suffix}"])
            assert torch.equal(fused[p + f"fc1.{suffix}"][h:], orig[p + f"fc1_x.{suffix}"])


@pytest.mark.ssl_unit
def test_swiglu_fc1_fuse_split_roundtrip():
    """split(fuse(x)) == x for a SwiGLU checkpoint (load -> export is lossless)."""
    torch.manual_seed(1)
    orig = _swiglu_split_state_dict()
    fused = fuse_timm_swiglu_fc1(orig)
    # `reference` carries fc1_g keys (ViT-H+/7B target) -> split is applied.
    restored = split_fused_swiglu_fc1(fused, reference=orig)
    assert set(restored) == set(orig)
    for k in orig:
        assert torch.equal(restored[k], orig[k]), k


@pytest.mark.ssl_unit
def test_swiglu_fc1_split_fuse_roundtrip():
    """fuse(split(y)) == y starting from a fused checkpoint (export -> load is lossless)."""
    torch.manual_seed(2)
    fused = fuse_timm_swiglu_fc1(_swiglu_split_state_dict())
    refused = fuse_timm_swiglu_fc1(split_fused_swiglu_fc1(fused, reference=_swiglu_split_state_dict()))
    assert set(refused) == set(fused)
    for k in fused:
        assert torch.equal(refused[k], fused[k]), k


@pytest.mark.ssl_unit
def test_plain_mlp_is_untouched():
    """Plain-MLP (ViT-B/L) ``fc1`` must NOT be split/fused: it has no fc1_g, and a plain-MLP
    target reference (no fc1_g) leaves the single fc1 intact."""
    torch.manual_seed(3)
    plain = {
        "blocks.0.mlp.fc1.weight": torch.randn(6, 8),
        "blocks.0.mlp.fc1.bias": torch.randn(6),
        "blocks.0.mlp.fc2.weight": torch.randn(8, 6),
        "blocks.0.mlp.fc2.bias": torch.randn(8),
    }
    # fuse: no fc1_g in source -> unchanged.
    fused = fuse_timm_swiglu_fc1(plain)
    assert set(fused) == set(plain)
    for k in plain:
        assert torch.equal(fused[k], plain[k])

    # split: reference has no fc1_g (plain-MLP target) -> fc1 must stay a single tensor.
    plain_reference = {"blocks.0.mlp.fc1.weight": torch.zeros(6, 8)}
    split = split_fused_swiglu_fc1(plain, reference=plain_reference)
    assert set(split) == set(plain)
    for k in plain:
        assert torch.equal(split[k], plain[k])
