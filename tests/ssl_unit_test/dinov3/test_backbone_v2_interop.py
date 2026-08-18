# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 SSL <-> cv/backbone_v2 interop tests.

CPU tests cover the key-rename round-trip and the checkpoint-extraction logic (no GPU, no
weights). The GPU test is the real gate: an SSL-trained backbone, after ``convert``, loads
into the existing ``dinov3_vitb16`` registry entry and produces matching features.
"""
import os

import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    timm_to_tao,
    tao_to_timm,
    extract_backbone_state_dict,
    remap_tao_backbone_to_timm,
    convert_ssl_to_timm,
)

EMBED = 8
REG = 4


@pytest.mark.ssl_unit
def test_rename_is_invertible():
    """timm -> TAO -> timm is the identity on representative keys."""
    keys = [
        "cls_token", "reg_token", "patch_embed.proj.weight", "patch_embed.proj.bias",
        "blocks.0.gamma_1", "blocks.11.gamma_2", "blocks.5.attn.qkv.weight",
        "blocks.5.attn.proj.bias", "blocks.0.norm1.weight", "blocks.7.mlp.fc1.weight",
        "norm.weight",
    ]
    for k in keys:
        assert tao_to_timm(timm_to_tao(k)) == k


@pytest.mark.ssl_unit
def test_specific_renames():
    """The non-identity renames map both directions."""
    assert timm_to_tao("reg_token") == "register_tokens"
    assert timm_to_tao("blocks.3.gamma_1") == "blocks.3.ls1.gamma"
    assert timm_to_tao("blocks.3.gamma_2") == "blocks.3.ls2.gamma"
    assert tao_to_timm("register_tokens") == "reg_token"
    assert tao_to_timm("blocks.3.ls1.gamma") == "blocks.3.gamma_1"
    assert tao_to_timm("blocks.3.ls2.gamma") == "blocks.3.gamma_2"
    # identity for everything else
    assert tao_to_timm("blocks.3.attn.qkv.weight") == "blocks.3.attn.qkv.weight"


@pytest.mark.ssl_unit
def test_extract_from_full_lightning_ckpt():
    """A full checkpoint selects the requested source's backbone and de-prefixes it."""
    raw = {"state_dict": {
        "student.backbone.cls_token": torch.zeros(1, 1, EMBED),
        "teacher.backbone.cls_token": torch.ones(1, 1, EMBED),
        "student.dino_head.last_layer": torch.zeros(2),
        "gram_teacher.cls_token": torch.full((1, 1, EMBED), 2.0),
    }}
    teacher_sd = extract_backbone_state_dict(raw, source="teacher")
    assert set(teacher_sd) == {"cls_token"}
    assert torch.equal(teacher_sd["cls_token"], torch.ones(1, 1, EMBED))
    student_sd = extract_backbone_state_dict(raw, source="student")
    assert torch.equal(student_sd["cls_token"], torch.zeros(1, 1, EMBED))


@pytest.mark.ssl_unit
def test_extract_from_stripped_backbone_file():
    """A stripped backbone file (no prefixes) passes through, dropping stray head keys."""
    raw = {"cls_token": torch.zeros(1, 1, EMBED), "blocks.0.ls1.gamma": torch.zeros(EMBED)}
    sd = extract_backbone_state_dict(raw, source="teacher")
    assert set(sd) == {"cls_token", "blocks.0.ls1.gamma"}


@pytest.mark.ssl_unit
def test_remap_drops_tao_only_and_renames():
    """remap drops mask_token, renames registers/gammas, keeps the rest."""
    tao_sd = {
        "mask_token": torch.zeros(1, 1, EMBED),
        "register_tokens": torch.zeros(1, REG, EMBED),
        "blocks.0.ls1.gamma": torch.zeros(EMBED),
        "cls_token": torch.zeros(1, 1, EMBED),
    }
    out = remap_tao_backbone_to_timm(tao_sd)
    assert "mask_token" not in out
    assert "reg_token" in out and "blocks.0.gamma_1" in out and "cls_token" in out


# ---------------------------------------------------------------------------------------------
# GPU + weights: the real SSL -> backbone_v2 round-trip gate.
# ---------------------------------------------------------------------------------------------

_WEIGHT_CANDIDATES = [
    os.environ.get("DINOV3_VITB_WEIGHTS", ""),
    "/data/weights/dinov3/vitb16",
    os.path.expanduser("~/weights/dinov3/vitb16"),
]


def _find_weights():
    for cand in _WEIGHT_CANDIDATES:
        if cand and os.path.exists(cand):
            return cand
    return None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
_weights = _find_weights()
requires_weights = pytest.mark.skipif(_weights is None, reason="DINOv3 ViT-B weights not found")


def _cosine(a, b):
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


@requires_cuda
@requires_weights
@pytest.mark.ssl_unit
def test_ssl_backbone_loads_into_backbone_v2(tmp_path):
    """Convert an SSL backbone and load it into dinov3_vitb16; features must match."""
    pytest.importorskip("timm")
    from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
    from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
    from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

    # Build our ViT and load the real DINOv3 weights via the forward (timm->TAO) remap.
    model = DinoV3VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        init_values=1e-5, drop_path_schedule="linear", num_classes=0, drop_path_rate=0.0,
        register_tokens=4, use_custom_attention=True,
    )
    timm_sd = DinoV3PlModel._load_pretrained_state_dict(_weights)
    remapped, _ = DinoV3PlModel._remap_dinov3_state_dict(timm_sd, model.state_dict())
    model.load_state_dict(remapped, strict=False)

    # Save a stripped backbone file (TAO naming), then convert to timm layout.
    stripped = {k: v for k, v in model.state_dict().items()
                if k != "mask_token" and not k.endswith("rope.periods")}
    src = str(tmp_path / "teacher_backbone.pth")
    torch.save(stripped, src)
    dst = str(tmp_path / "dinov3_vitb_backbone.safetensors")
    convert_ssl_to_timm(src, dst, source="teacher", validate=True,
                        timm_model_name="vit_base_patch16_dinov3")

    # Load into the existing backbone_v2 registry entry (strict timm load happens inside).
    backbone = BACKBONE_REGISTRY.get("dinov3_vitb16")(
        pretrained_backbone_path=dst, num_classes=0).cuda().half().eval()
    ours = model.cuda().half().eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256).cuda().half()
    with torch.no_grad():
        cls_v2, _feat_v2 = backbone(x, return_features=True, return_logits=False)
        out = ours(x)

    cls_cos = _cosine(out["x_norm_clstoken"], cls_v2)
    assert cls_cos > 0.99, f"CLS cosine after backbone_v2 round-trip too low: {cls_cos}"
