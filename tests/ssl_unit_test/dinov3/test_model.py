# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model unit tests.

The full-model tests need a CUDA GPU (the inherited nvdinov2 attention uses xformers'
memory_efficient_attention), so they skip when CUDA is unavailable.
"""
import pytest
from omegaconf import OmegaConf
import torch

from nvidia_tao_pytorch.config.dinov3.default_config import (
    ExperimentConfig,
    SUPPORTED_IMAGE_SIZES,
    validate_img_size,
)
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
from nvidia_tao_pytorch.ssl.dinov3.model.layers.attention import RoPEMemoryEfficientAttention

BATCH_SIZE = 2
IMAGE_CHANNEL = 3
N_GLOBAL_CROPS = 1
GLOBAL_CROPS_SIZE = 256
N_LOCAL_CROPS = 1
LOCAL_CROPS_SIZE = 112
PATCH_SIZE = 16
TEACHER_TEMPERATURE = 0.995

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _assert_actionable_checkpoint_error(error, path):
    """Every invalid-checkpoint failure should identify the field, path, and accepted family."""
    message = str(error.value)
    assert "DINOv3 pretrained_model_path" in message
    assert str(path) in message
    assert "DINOv2/NVDINOv2 checkpoints are not supported" in message


@pytest.fixture
def _test_batch():
    torch.manual_seed(47)
    batch = {}
    batch["global_crops"] = torch.randn(BATCH_SIZE * N_GLOBAL_CROPS, IMAGE_CHANNEL, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE)
    batch["local_crops"] = torch.randn(BATCH_SIZE * N_LOCAL_CROPS, IMAGE_CHANNEL, LOCAL_CROPS_SIZE, LOCAL_CROPS_SIZE)
    n_tok = (GLOBAL_CROPS_SIZE // PATCH_SIZE) * (GLOBAL_CROPS_SIZE // PATCH_SIZE)
    batch["global_masks"] = torch.zeros((BATCH_SIZE * N_GLOBAL_CROPS, n_tok), dtype=torch.bool)
    batch["global_masks_indices"] = batch["global_masks"].flatten().nonzero().flatten()
    batch["global_masks_weight"] = (
        1 / batch["global_masks"].float().sum(-1).clamp(min=1.0)
    ).unsqueeze(-1).expand_as(batch["global_masks"])[batch["global_masks"]]
    yield batch


@requires_cuda
@pytest.mark.ssl_unit
def test_dinov3_vit_builds_and_forwards():
    """The v3 ViT-B builds with no absolute pos-embed and runs the single-tensor RoPE forward."""
    model = DinoV3VisionTransformer(
        img_size=GLOBAL_CROPS_SIZE, patch_size=PATCH_SIZE, embed_dim=768, depth=12,
        num_heads=12, init_values=1e-5, drop_path_schedule="linear", num_classes=0,
        drop_path_rate=0.0, register_tokens=4, use_custom_attention=True,
    ).cuda().half().eval()

    # DINOv3 drops the absolute positional embedding (RoPE replaces it).
    assert model.pos_embed is None
    assert "pos_embed" not in model.state_dict()

    x = torch.randn(BATCH_SIZE, IMAGE_CHANNEL, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE).cuda().half()
    with torch.no_grad():
        out = model(x)

    n_patches = (GLOBAL_CROPS_SIZE // PATCH_SIZE) ** 2
    assert out["x_norm_clstoken"].shape == (BATCH_SIZE, 768)
    assert out["x_norm_patchtokens"].shape == (BATCH_SIZE, n_patches, 768)


@requires_cuda
@pytest.mark.ssl_unit
def test_dinov3_backbone_multicrop_finite():
    """The multi-crop RoPE list path yields finite features in both eval and train modes.

    This is the meaningful check of the new code: RoPE is threaded through the nested
    ``BlockDiagonalMask`` batching and, with drop_path > 0, the stochastic-depth path in
    train mode. (Single-tensor eval is covered by ``test_dinov3_vit_builds_and_forwards``.)
    """
    backbone = DinoV3VisionTransformer(
        img_size=GLOBAL_CROPS_SIZE, patch_size=PATCH_SIZE, embed_dim=768, depth=12,
        num_heads=12, init_values=1e-5, drop_path_schedule="linear", num_classes=0,
        drop_path_rate=0.4, register_tokens=4, use_custom_attention=True,
    ).to(torch.float16).cuda()

    gc = torch.randn(BATCH_SIZE, IMAGE_CHANNEL, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE).half().cuda()
    lc = torch.randn(BATCH_SIZE, IMAGE_CHANNEL, LOCAL_CROPS_SIZE, LOCAL_CROPS_SIZE).half().cuda()
    n_tok = (GLOBAL_CROPS_SIZE // PATCH_SIZE) ** 2
    gm = torch.zeros((BATCH_SIZE, n_tok), dtype=torch.bool).cuda()

    for mode in ("eval", "train"):
        getattr(backbone, mode)()
        with torch.no_grad():
            out = backbone([gc, lc], masks=[gm, None])
        for crop in out:
            assert torch.isfinite(crop["x_norm_patchtokens"]).all(), f"non-finite patches in {mode}"
            assert torch.isfinite(crop["x_norm_clstoken"]).all(), f"non-finite cls in {mode}"


@requires_cuda
@pytest.mark.ssl_unit
def test_dinov3_attention_fallback_matches_xformers():
    """The non-xformers fallback (used e.g. on Blackwell) must match memory_efficient_attention.

    Regression guard: the fallback previously computed scores over the head axis
    (``[B, N, H, H]``) instead of the sequence axis, succeeding dimensionally but returning
    garbage. With ``attn_drop=0`` and identical input the fallback should track the xformers
    path closely (fp16 tolerance).
    """
    torch.manual_seed(0)
    attn = RoPEMemoryEfficientAttention(
        dim=768, num_heads=12, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0,
    ).cuda().half().eval()

    x = torch.randn(BATCH_SIZE, 197, 768).cuda().half()
    with torch.no_grad():
        out_xf = attn(x, use_custom_attention=True)
        out_fb = attn(x, use_custom_attention=False)

    assert out_fb.shape == out_xf.shape == (BATCH_SIZE, 197, 768)
    cos = torch.nn.functional.cosine_similarity(
        out_fb.float().reshape(-1, 768), out_xf.float().reshape(-1, 768), dim=-1
    ).mean().item()
    assert cos > 0.99, f"fallback attention diverges from xformers: cosine {cos}"


@requires_cuda
@pytest.mark.ssl_unit
def test_dinov3_model_forward(_test_batch):
    """Build DinoV3PlModel and run teacher + student forward (nested multi-crop RoPE path).

    Mirrors the nvdinov2 model test's contract: the forward runs end-to-end and returns a
    scalar loss tensor. Finiteness is not asserted -- on random fp16 inputs with an untrained
    head and 131072 prototypes the DINO/iBOT/KoLeo terms can legitimately be inf/nan (the
    nvdinov2 test likewise only checks the forward executes).
    """
    experiment_config = OmegaConf.structured(ExperimentConfig())

    model = DinoV3PlModel(experiment_config)
    model.to(torch.float16).train().cuda()

    # The inherited _extra_losses hook is a no-op for the base/DINOv3 scaffold (Gram added later).
    assert model._extra_losses() == []

    global_crops = _test_batch["global_crops"].to(torch.float16).cuda()
    local_crops = _test_batch["local_crops"].to(torch.float16).cuda()
    global_masks = _test_batch["global_masks"].cuda()
    global_masks_indices = _test_batch["global_masks_indices"].cuda()
    global_masks_weight = _test_batch["global_masks_weight"].to(torch.float16).cuda()

    with torch.no_grad():
        teacher_dino_centered, teacher_ibot_centered, _ = model.teacher_forward(
            global_crops=global_crops,
            global_masks_indices=global_masks_indices,
            teacher_temperature=TEACHER_TEMPERATURE,
        )
        loss = model.student_forward(
            global_crops=global_crops,
            global_masks=global_masks,
            global_masks_indices=global_masks_indices,
            global_masks_weight=global_masks_weight,
            local_crops=local_crops,
            teacher_dino_centered=teacher_dino_centered,
            teacher_ibot_centered=teacher_ibot_centered,
        )
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0


@pytest.mark.ssl_unit
@pytest.mark.parametrize("role", ["teacher_type", "student_type"])
def test_dinov3_unsupported_backbone_type_raises(role):
    """An unsupported backbone name must raise a clear ValueError up front, not a KeyError.

    Regression for bug 6460904: the JSON-schema enum (STR_FIELD valid_options) is doc-only, so
    'vit_h' (not a real TAO DINOv3 arch -- only vit_h_plus exists) passed config validation and
    died with a bare KeyError deep in model init. Validation now happens before any heavy/CUDA
    work, so this raises without building the model.
    """
    cfg = OmegaConf.structured(ExperimentConfig())
    setattr(cfg.model.backbone, role, "vit_h")
    with pytest.raises(ValueError, match="Unsupported model.backbone"):
        DinoV3PlModel(cfg)


@pytest.mark.ssl_unit
def test_dinov3_validate_backbone_types_accepts_supported():
    """Every advertised architecture passes the up-front backbone validation."""
    from nvidia_tao_pytorch.config.dinov3.default_config import map_params, SUPPORTED_BACKBONES
    for name in SUPPORTED_BACKBONES:
        DinoV3PlModel._validate_backbone_types(
            {"teacher_type": name, "student_type": name}, map_params
        )


@pytest.mark.ssl_unit
def test_dinov3_unsupported_img_size_raises_before_model_build():
    """An out-of-enum image size must fail before model or CUDA initialization."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.backbone.img_size = 300

    with pytest.raises(
        ValueError,
        match=r"model\.backbone\.img_size: 300.*\[256, 512, 768\]",
    ):
        DinoV3PlModel(cfg)


@pytest.mark.ssl_unit
@pytest.mark.parametrize("img_size", SUPPORTED_IMAGE_SIZES)
def test_dinov3_validate_img_size_accepts_supported(img_size):
    """Every image size advertised by the schema passes runtime validation."""
    validate_img_size({"img_size": img_size})


@pytest.mark.ssl_unit
@pytest.mark.parametrize("path_kind", ["missing", "empty_directory"])
def test_dinov3_pretrained_path_error_is_actionable(tmp_path, path_kind):
    """Missing paths and empty directories should fail before a low-level loader traceback."""
    path = tmp_path / path_kind
    if path_kind == "empty_directory":
        path.mkdir()

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._load_pretrained_state_dict(path)
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_corrupt_pretrained_file_error_is_actionable(tmp_path):
    """A corrupt checkpoint should report the configured path and accepted DINOv3 formats."""
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"not a safetensors checkpoint")

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._load_pretrained_state_dict(path)
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_payload_must_be_tensor_state_dict(tmp_path):
    """A loadable file with the wrong payload type is still not a valid checkpoint."""
    path = tmp_path / "not-a-state-dict.pth"
    torch.save(["not", "a", "state", "dict"], path)

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._load_pretrained_state_dict(path)
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_loader_accepts_tensor_state_dict(tmp_path):
    """A supported file containing a tensor state dict should load unchanged."""
    path = tmp_path / "model.pth"
    expected = {"cls_token": torch.randn(1, 1, 4)}
    torch.save(expected, path)

    loaded = DinoV3PlModel._load_pretrained_state_dict(path)
    assert set(loaded) == set(expected)
    assert torch.equal(loaded["cls_token"], expected["cls_token"])


@pytest.mark.ssl_unit
def test_dinov3_pretrained_loader_preserves_tlt_support(tmp_path):
    """Inference advertises torch-serialized .tlt checkpoints, so the shared loader accepts them."""
    path = tmp_path / "model.tlt"
    expected = {"cls_token": torch.randn(1, 1, 4)}
    torch.save(expected, path)

    loaded = DinoV3PlModel._load_pretrained_state_dict(path)
    assert torch.equal(loaded["cls_token"], expected["cls_token"])


@pytest.mark.ssl_unit
@pytest.mark.parametrize("checkpoint_kind", ["full", "stripped"])
def test_dinov3_export_restores_teacher_checkpoint(monkeypatch, checkpoint_kind):
    """Export selects full-checkpoint teacher weights and preserves the stripped path."""
    from nvidia_tao_pytorch.ssl.dinov3.scripts.export import _restore_export_checkpoint

    student_cls_token = torch.zeros(1, 1, 4)
    teacher_cls_token = torch.ones(1, 1, 4)
    if checkpoint_kind == "full":
        checkpoint = {
            "student.backbone.cls_token": student_cls_token,
            "teacher.backbone.cls_token": teacher_cls_token,
        }
    else:
        checkpoint = {"cls_token": teacher_cls_token}

    model = object.__new__(DinoV3PlModel)
    torch.nn.Module.__init__(model)
    model.pretrained_weights = "model_epoch_001.pth"
    model.student = torch.nn.ModuleDict({
        "backbone": torch.nn.Module(),
    })
    model.student.backbone.register_parameter(
        "cls_token",
        torch.nn.Parameter(torch.full((1, 1, 4), -1.0)),
    )
    model.teacher = torch.nn.ModuleDict({
        "backbone": torch.nn.Module(),
    })
    model.teacher.backbone.register_parameter(
        "cls_token",
        torch.nn.Parameter(torch.full((1, 1, 4), -2.0)),
    )
    model.model_config = OmegaConf.create({"distill": {"enable": False}})

    loaded_paths = []

    def _load_checkpoint(path):
        loaded_paths.append(path)
        return checkpoint

    monkeypatch.setattr(model, "_load_pretrained_state_dict", _load_checkpoint)
    log_messages = []
    monkeypatch.setattr(
        "nvidia_tao_pytorch.ssl.dinov3.model.pl_model.logging.info",
        log_messages.append,
    )

    _restore_export_checkpoint(model, model.pretrained_weights)

    assert loaded_paths == [model.pretrained_weights]
    assert torch.equal(model.teacher.backbone.cls_token, teacher_cls_token)
    expected_student = (
        torch.full((1, 1, 4), -1.0)
        if checkpoint_kind == "full"
        else teacher_cls_token
    )
    assert torch.equal(model.student.backbone.cls_token, expected_student)
    assert not torch.equal(model.teacher.backbone.cls_token, student_cls_token)
    if checkpoint_kind == "full":
        assert any("selected 'teacher.backbone'" in message for message in log_messages)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_validation_rejects_wrong_family(tmp_path):
    """Absolute positional embeddings identify an unsupported DINOv2-style checkpoint."""
    path = tmp_path / "dinov2.pth"
    state_dict = {
        "cls_token": torch.randn(1, 1, 4),
        "pos_embed": torch.randn(1, 5, 4),
    }
    reference = {"cls_token": torch.zeros(1, 1, 4)}

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._validate_and_remap_pretrained_state_dict(
            state_dict,
            reference,
            path,
        )
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_validation_wraps_remap_errors(tmp_path):
    """Malformed SwiGLU pairs should not expose a low-level torch concatenation error."""
    path = tmp_path / "malformed-swiglu.pth"
    state_dict = {
        "blocks.0.mlp.fc1_g.weight": torch.randn(4, 3),
        "blocks.0.mlp.fc1_x.weight": torch.randn(4, 5),
    }

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._validate_and_remap_pretrained_state_dict(
            state_dict,
            {},
            path,
        )
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_validation_requires_backbone_coverage(tmp_path):
    """Partial or shape-incompatible checkpoints must not silently initialize the rest."""
    path = tmp_path / "partial.pth"
    state_dict = {"cls_token": torch.randn(1, 1, 4)}
    reference = {
        "cls_token": torch.zeros(1, 1, 4),
        "patch_embed.proj.weight": torch.zeros(4, 3, 2, 2),
        "mask_token": torch.zeros(1, 4),
    }

    with pytest.raises(ValueError) as error:
        DinoV3PlModel._validate_and_remap_pretrained_state_dict(
            state_dict,
            reference,
            path,
        )
    _assert_actionable_checkpoint_error(error, path)


@pytest.mark.ssl_unit
def test_dinov3_pretrained_validation_accepts_matching_timm_keys(tmp_path):
    """A matching timm state dict may omit only the locally initialized iBOT mask token."""
    path = tmp_path / "dinov3.pth"
    state_dict = {
        "cls_token": torch.randn(1, 1, 4),
        "reg_token": torch.randn(1, 2, 4),
    }
    reference = {
        "cls_token": torch.zeros(1, 1, 4),
        "register_tokens": torch.zeros(1, 2, 4),
        "mask_token": torch.zeros(1, 4),
    }

    remapped, unmapped = DinoV3PlModel._validate_and_remap_pretrained_state_dict(
        state_dict,
        reference,
        path,
    )
    assert set(remapped) == {"cls_token", "register_tokens"}
    assert not unmapped
