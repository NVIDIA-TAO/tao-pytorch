# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 config unit tests (defaults, param map, inheritance from nvdinov2)."""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import (
    DINOv3BackboneConfig,
    DINOv3CuDNNConfig,
    DINOv3ExportExpConfig,
    DINOv3TrainExpConfig,
    DINOv3TransformConfig,
    ExperimentConfig,
    map_params,
    SUPPORTED_BACKBONES,
    SUPPORTED_IMAGE_SIZES,
    validate_img_size,
)

EXPECTED_PRETRAINED_DESCRIPTION = (
    "Path to DINOv3 pretrained weights matching the configured backbone. "
    "Accepts a timm-format directory or file, or a stripped TAO DINOv3 "
    "backbone checkpoint. DINOv2/NVDINOv2 checkpoints are not supported."
)


@pytest.mark.config
@pytest.mark.ssl_unit
def test_default_model_name_is_dinov3():
    """The DINOv3 experiment config defaults its model name to 'dinov3'."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.model_name == "dinov3"


@pytest.mark.config
@pytest.mark.ssl_unit
def test_pretrained_model_path_description_is_dinov3_specific():
    """DINOv3 metadata must not inherit the NVDINOv2 checkpoint contract."""
    field = DINOv3TrainExpConfig.__dataclass_fields__["pretrained_model_path"]
    assert field.metadata["description"] == EXPECTED_PRETRAINED_DESCRIPTION
    assert field.metadata["default_value"] is None


@pytest.mark.config
@pytest.mark.ssl_unit
def test_cudnn_defaults_support_custom_attention():
    """DINOv3 must not inherit deterministic CuDNN from the common train config."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.train.cudnn.benchmark is True
    assert cfg.train.cudnn.deterministic is False

    fields = DINOv3CuDNNConfig.__dataclass_fields__
    assert fields["benchmark"].metadata["description"]
    assert fields["benchmark"].metadata["display_name"] == "CuDNN benchmark"
    assert fields["benchmark"].metadata["popular"] == "no"
    assert fields["deterministic"].metadata["description"]
    assert fields["deterministic"].metadata["display_name"] == "CuDNN deterministic"
    assert fields["deterministic"].metadata["popular"] == "no"


@pytest.mark.config
@pytest.mark.ssl_unit
def test_backbone_patch16_rope_defaults():
    """DINOv3 backbone defaults: patch-16, 4 register tokens, ViT-B, RoPE theta present."""
    cfg = OmegaConf.structured(ExperimentConfig())
    bb = cfg.model.backbone
    assert bb.patch_size == 16
    assert bb.num_register_tokens == 4
    assert bb.teacher_type == "vit_b"
    assert bb.student_type == "vit_b"
    assert bb.img_size == 256
    assert SUPPORTED_IMAGE_SIZES == (256, 512, 768)
    assert bb.rope_theta == 100.0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_validate_img_size_is_public_config_contract():
    """The shared validator accepts mappings/dataclasses and rejects unsupported values."""
    for img_size in SUPPORTED_IMAGE_SIZES:
        validate_img_size({"img_size": img_size})

    backbone_config = DINOv3BackboneConfig(img_size=300)
    with pytest.raises(
        ValueError,
        match=r"model\.backbone\.img_size: 300.*\[256, 512, 768\]",
    ):
        validate_img_size(backbone_config)
    assert ExperimentConfig().model.backbone.img_size == 256


@pytest.mark.config
@pytest.mark.ssl_unit
def test_gram_and_lora_present():
    """Gram and (disabled) LoRA configs are present on the model config."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert hasattr(cfg.model, "gram")
    assert hasattr(cfg.model, "lora")
    assert cfg.model.lora.enable is False


@pytest.mark.config
@pytest.mark.ssl_unit
def test_dinov3_256_transform_defaults():
    """DINOv3 defaults to 256 global crops with patch-16-friendly local crops."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.dataset.transform.global_crops_size == 256
    assert cfg.dataset.transform.local_crops_size % 16 == 0
    field = DINOv3TransformConfig.__dataclass_fields__["global_crops_size"]
    assert field.metadata["description"] == "Size of global crops for DINOv3 training."


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_b():
    """The v3 param map carries the ViT-B (768/12/12, standard MLP) entry."""
    assert map_params["embed_dim"]["vit_b"] == 768
    assert map_params["depth"]["vit_b"] == 12
    assert map_params["num_heads"]["vit_b"] == 12
    assert map_params["mlp_layer"]["vit_b"] == "mlp"
    # ViT-H+ and ViT-7B use SwiGLU.
    assert map_params["mlp_layer"]["vit_h_plus"] == "swiglu"
    assert map_params["mlp_layer"]["vit_7b"] == "swiglu"
    assert set(SUPPORTED_BACKBONES) == {"vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"}


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_l():
    """The v3 param map carries the ViT-L (1024/24/16, standard GELU MLP) entry (Phase 2)."""
    assert map_params["embed_dim"]["vit_l"] == 1024
    assert map_params["depth"]["vit_l"] == 24
    assert map_params["num_heads"]["vit_l"] == 16
    assert map_params["mlp_layer"]["vit_l"] == "mlp"
    # head_dim = 1024 / 16 = 64 (same as ViT-B) -> RoPE (needs head_dim % 4 == 0) works unchanged.
    assert (map_params["embed_dim"]["vit_l"] // map_params["num_heads"]["vit_l"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_s():
    """The v3 param map carries the ViT-S (384/12/6, standard MLP) entry."""
    assert map_params["embed_dim"]["vit_s"] == 384
    assert map_params["depth"]["vit_s"] == 12
    assert map_params["num_heads"]["vit_s"] == 6
    assert map_params["mlp_layer"]["vit_s"] == "mlp"
    assert map_params["mlp_ratio"]["vit_s"] == 4.0
    # head_dim = 384 / 6 = 64 (same as ViT-B/L) -> RoPE (needs head_dim % 4 == 0) works unchanged.
    assert (map_params["embed_dim"]["vit_s"] // map_params["num_heads"]["vit_s"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_s_plus():
    """The v3 param map carries the ViT-S+ (384/12/6, SwiGLU) entry."""
    assert map_params["embed_dim"]["vit_s_plus"] == 384
    assert map_params["depth"]["vit_s_plus"] == 12
    assert map_params["num_heads"]["vit_s_plus"] == 6
    assert map_params["mlp_layer"]["vit_s_plus"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_s_plus"] == 4.0
    assert (map_params["embed_dim"]["vit_s_plus"] // map_params["num_heads"]["vit_s_plus"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_h_plus():
    """The v3 param map carries the ViT-H+ (1280/32/20, SwiGLU) entry."""
    assert map_params["embed_dim"]["vit_h_plus"] == 1280
    assert map_params["depth"]["vit_h_plus"] == 32
    assert map_params["num_heads"]["vit_h_plus"] == 20
    assert map_params["mlp_layer"]["vit_h_plus"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_h_plus"] == 4.0
    assert (map_params["embed_dim"]["vit_h_plus"] // map_params["num_heads"]["vit_h_plus"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_7b():
    """The v3 param map carries the ViT-7B (4096/40/32, SwiGLU) entry; the public 7B
    checkpoint uses the narrow SwiGLU (mlp_ratio 2.0)."""
    assert map_params["embed_dim"]["vit_7b"] == 4096
    assert map_params["depth"]["vit_7b"] == 40
    assert map_params["num_heads"]["vit_7b"] == 32
    assert map_params["mlp_layer"]["vit_7b"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_7b"] == 2.0
    assert (map_params["embed_dim"]["vit_7b"] // map_params["num_heads"]["vit_7b"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_export_trace_shape_matches_backbone():
    """Export ONNX trace defaults match the patch-16 backbone (256, not nvdinov2's 518)."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.export.input_width == 256
    assert cfg.export.input_height == 256
    assert cfg.export.input_width % cfg.model.backbone.patch_size == 0
    assert cfg.export.input_height % cfg.model.backbone.patch_size == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_export_checkpoint_contract_selects_teacher():
    """The generated schema must document deterministic teacher selection for full checkpoints."""
    field = DINOv3ExportExpConfig.__dataclass_fields__["checkpoint"]
    assert "full Lightning training checkpoint" in field.metadata["description"]
    assert "always selects the EMA teacher backbone" in field.metadata["description"]
