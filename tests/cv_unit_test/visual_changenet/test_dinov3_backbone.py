# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Visual ChangeNet DINOv3 backbone integration."""

import pytest
import torch
from torch import nn

from nvidia_tao_pytorch.cv.backbone_v2 import dino_v3 as backbone_v2_dino_v3
from nvidia_tao_pytorch.cv.backbone_v2.dino_v3 import validate_dinov3_checkpoint
from nvidia_tao_pytorch.cv.visual_changenet.backbone import dino_v3
from nvidia_tao_pytorch.cv.visual_changenet.backbone.vit_adapter import vit_adapter_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.classification.models import (
    changenet as classification_changenet,
)
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models import (
    changenet as segmentation_changenet,
)


DINOV3_BACKBONES = {
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
    "vit_7b_dinov3",
}


def _valid_checkpoint():
    return {
        "cls_token": torch.empty(1),
        "patch_embed.proj.weight": torch.empty(1),
        "blocks.0.gamma_1": torch.empty(1),
    }


class _LogRecorder:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class _PatchEmbed(nn.Module):
    patch_size = (16, 16)

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 12, kernel_size=16, stride=16)


class _TinyDINOv3(nn.Module):
    embed_dim = 12
    num_prefix_tokens = 5
    num_features = 12

    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.blocks = nn.ModuleList([nn.Linear(12, 12)])


def test_all_dinov3_variants_are_registered():
    """Both Visual ChangeNet difference paths expose all TAO DINOv3 variants."""
    assert DINOV3_BACKBONES <= vit_adapter_model_dict.keys()
    assert DINOV3_BACKBONES <= dino_v3.dinov3_model_dict.keys()


def test_7b_factory_uses_40_block_partition(monkeypatch):
    """The 7B factory selects the timm 7B model without constructing it in CI."""
    monkeypatch.setattr(
        dino_v3,
        "_make_dinov3_adapter",
        lambda timm_name, indexes, **kwargs: (timm_name, indexes, kwargs),
    )
    timm_name, indexes, kwargs = dino_v3.vit_7b_dinov3(resolution=224)
    assert timm_name == "vit_7b_patch16_dinov3.lvd1689m"
    assert indexes == [[0, 9], [10, 19], [20, 29], [30, 39]]
    assert kwargs == {"resolution": 224}


def test_dinov3_uses_validated_local_checkpoint_or_hf_fallback(monkeypatch):
    """Local checkpoints are validated; an explicit flag enables the HF fallback."""
    create_calls = []
    load_calls = []
    checkpoint = _valid_checkpoint()

    class Model:
        def load_state_dict(self, state_dict, strict=True):
            load_calls.append((state_dict, strict))

    sentinel = Model()

    def create_model(name, **kwargs):
        create_calls.append((name, kwargs))
        return sentinel

    def load_pretrained(path, **kwargs):
        load_calls.append((path, kwargs))
        return checkpoint

    monkeypatch.setattr(backbone_v2_dino_v3.timm, "create_model", create_model)
    monkeypatch.setattr(
        backbone_v2_dino_v3,
        "load_pretrained_weights",
        load_pretrained,
    )

    assert backbone_v2_dino_v3._load_dino_v3("test_dinov3") is sentinel
    assert create_calls[-1] == ("test_dinov3", {"pretrained": False})

    assert backbone_v2_dino_v3._load_dino_v3(
        "test_dinov3", pretrained=True
    ) is sentinel
    assert create_calls[-1] == ("test_dinov3", {"pretrained": True})

    assert backbone_v2_dino_v3._load_dino_v3(
        "test_dinov3",
        pretrained_backbone_path="converted.safetensors",
    ) is sentinel
    assert create_calls[-1] == ("test_dinov3", {"pretrained": False})
    assert load_calls == [
        ("converted.safetensors", {"weights_only": True}),
        (checkpoint, True),
    ]


def test_hf_failure_has_actionable_guidance(monkeypatch):
    """Gated/offline Hugging Face errors explain both supported recovery paths."""

    def fail_create_model(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(backbone_v2_dino_v3.timm, "create_model", fail_create_model)
    with pytest.raises(RuntimeError) as error:
        backbone_v2_dino_v3._load_dino_v3("test_dinov3", pretrained=True)
    assert "HF_TOKEN" in str(error.value)
    assert "pretrained_backbone_path" in str(error.value)
    assert "dinov3 convert" in str(error.value)


def test_adapter_requests_hf_weights_when_pretrained(monkeypatch):
    """The AOI learnable-difference adapter forwards the HF fallback to timm."""
    calls = []
    sentinel = object()

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(dino_v3, "create_dinov3_model", create_model)
    monkeypatch.setattr(
        dino_v3,
        "DINOV3Adapter",
        lambda model, **kwargs: (model, kwargs),
    )

    model, _ = dino_v3._make_dinov3_adapter(
        "test_dinov3",
        [[0, 0]],
        resolution=224,
        pretrained=True,
    )
    assert model is sentinel
    assert calls == [("test_dinov3", {"pretrained": True, "img_size": 224})]


@pytest.mark.parametrize("pretrained_path", [None, ""])
def test_aoi_callers_freeze_dinov3_and_request_hf_fallback(monkeypatch, pretrained_path):
    """Classification and segmentation forward null DINOv3 paths to timm."""
    classify_logger = _LogRecorder()
    monkeypatch.setattr(classification_changenet, "logger", classify_logger)
    classify_calls = {}
    classify_backbone = nn.Identity()

    def classify_factory(**kwargs):
        classify_calls.update(kwargs)
        return classify_backbone

    monkeypatch.setitem(
        classification_changenet.vit_adapter_model_dict,
        "vit_small_dinov3",
        classify_factory,
    )
    monkeypatch.setattr(
        classification_changenet,
        "ChangeNetClassifyDecoder",
        lambda **kwargs: nn.Identity(),
    )
    classify_model = classification_changenet.ChangeNetClassify(
        model="vit_small_dinov3",
        output_shape=[32, 32],
        pretrained_backbone_path=pretrained_path,
        freeze_backbone=True,
    )
    assert classify_model.backbone is classify_backbone
    assert classify_calls["pretrained"] is True
    assert classify_calls["freeze_at"] == "all"
    assert any("timm/Hugging Face" in message for message in classify_logger.messages)

    segment_calls = {}
    segment_logger = _LogRecorder()
    monkeypatch.setattr(segmentation_changenet, "logger", segment_logger)
    segment_backbone = nn.Identity()

    def segment_factory(**kwargs):
        segment_calls.update(kwargs)
        return segment_backbone

    monkeypatch.setitem(
        segmentation_changenet.vit_adapter_model_dict,
        "vit_small_dinov3",
        segment_factory,
    )
    monkeypatch.setattr(
        segmentation_changenet,
        "DecoderTransformer_v3",
        lambda **kwargs: nn.Identity(),
    )
    segment_model = segmentation_changenet.ChangeNetSegment(
        model="vit_small_dinov3",
        img_size=32,
        pretrained_backbone_path=pretrained_path,
        freeze_backbone=True,
    )
    assert segment_model.backbone is segment_backbone
    assert segment_calls["pretrained"] is True
    assert segment_calls["freeze_at"] == "all"

    assert any("timm/Hugging Face" in message for message in segment_logger.messages)


def test_explicit_euclidean_checkpoint_uses_aoi_state_dict_loader(monkeypatch):
    """The cls-token path keeps AOI parsing and wrapper validation."""
    calls = {}
    checkpoint = _valid_checkpoint()

    class Backbone(nn.Module):
        def load_state_dict(self, state_dict, strict=True):
            calls["state_dict"] = state_dict
            calls["strict"] = strict
            return "loaded"

    def factory(**kwargs):
        calls.update(kwargs)
        return Backbone()

    monkeypatch.setitem(
        classification_changenet.dinov3_model_dict,
        "vit_small_dinov3",
        factory,
    )

    def load_pretrained(path, **kwargs):
        calls["loaded_path"] = path
        return checkpoint

    monkeypatch.setattr(
        classification_changenet, "load_pretrained_weights", load_pretrained
    )
    classification_changenet.ChangeNetClassify(
        model="vit_small_dinov3",
        difference_module="euclidean",
        output_shape=[32, 32],
        pretrained_backbone_path="converted.safetensors",
        freeze_backbone=True,
    )
    assert "pretrained_backbone_path" not in calls
    assert calls["pretrained"] is False
    assert calls["freeze_at"] == "all"
    assert calls["loaded_path"] == "converted.safetensors"
    assert calls["state_dict"] is checkpoint
    assert calls["strict"] is False


def test_existing_backbones_still_require_an_explicit_checkpoint_to_freeze():
    """The null-checkpoint exception is limited to DINOv3 HF profiles."""
    with pytest.raises(ValueError, match="without specifying"):
        classification_changenet.ChangeNetClassify(
            model="c_radio_v2_vit_base_patch16_224",
            freeze_backbone=True,
        )


def test_frozen_dinov3_keeps_only_adapter_trainable():
    """AOI frozen mode freezes the DINOv3 weights while training the task adapter."""
    adapter = dino_v3.DINOV3Adapter(
        _TinyDINOv3(),
        conv_inplane=8,
        interaction_indexes=[],
        add_summary=False,
        freeze_at="all",
    )
    assert all(not parameter.requires_grad for parameter in adapter.vit.parameters())
    assert any(
        parameter.requires_grad
        for name, parameter in adapter.named_parameters()
        if not name.startswith("vit.")
    )


def test_checkpoint_validation_accepts_timm_and_rejects_raw_ssl():
    """Converted/HF timm keys pass while a raw TAO SSL layout gives conversion help."""
    validate_dinov3_checkpoint(
        {
            "cls_token": torch.empty(1),
            "patch_embed.proj.weight": torch.empty(1),
            "blocks.0.gamma_1": torch.empty(1),
        }
    )
    with pytest.raises(ValueError, match="dinov3 convert"):
        validate_dinov3_checkpoint(
            {
                "cls_token": torch.empty(1),
                "register_tokens": torch.empty(1),
                "patch_embed.proj.weight": torch.empty(1),
                "blocks.0.ls1.gamma": torch.empty(1),
            }
        )


def test_wrapper_allows_partial_loads_but_still_rejects_raw_ssl():
    """Intentional strict=False task loads are allowed without admitting raw SSL."""
    wrapper = backbone_v2_dino_v3.DINOV3Wrapper(_TinyDINOv3())
    result = wrapper.load_state_dict(
        {"blocks.0.weight": torch.empty_like(wrapper.inner.blocks[0].weight)},
        strict=False,
    )
    assert "inner.blocks.0.bias" in result.missing_keys
    with pytest.raises(ValueError, match="dinov3 convert"):
        wrapper.load_state_dict(
            {"register_tokens": torch.empty(1)},
            strict=False,
        )


def test_registry_local_path_rejects_raw_ssl_before_model_creation(monkeypatch):
    """The explicit registry path validates weights before handing them to timm."""
    raw_checkpoint = {
        "cls_token": torch.empty(1),
        "register_tokens": torch.empty(1),
        "patch_embed.proj.weight": torch.empty(1),
        "blocks.0.ls1.gamma": torch.empty(1),
    }
    monkeypatch.setattr(
        backbone_v2_dino_v3,
        "load_pretrained_weights",
        lambda *args, **kwargs: raw_checkpoint,
    )

    def unexpected_create(*args, **kwargs):
        raise AssertionError("model construction must follow validation")

    monkeypatch.setattr(backbone_v2_dino_v3.timm, "create_model", unexpected_create)
    with pytest.raises(ValueError, match="dinov3 convert"):
        backbone_v2_dino_v3._load_dino_v3(
            "test_dinov3",
            pretrained_backbone_path="raw_ssl.safetensors",
        )
