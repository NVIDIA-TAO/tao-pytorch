# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone unit tests."""

import gc
import os
import platform

import pytest
import torch

from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2 import (
    convnext,
    convnext_v2,
    dino_v2,
    dino_v3,
    efficientvit,
    fan,
    fastervit,
    gcvit,
    hiera,
    mit,
    open_clip,
    radio,
    resnet,
    swin,
    vit,
)
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


TEST_BACKBONE_DIR_CI = "/home/scratch.metropolis2/tao_ci/tao_pytorch/models/backbones/"
TEST_BACKBONE_DIR_ORD = "/lustre/fsw/portfolios/edgeai/users/hongyuc/workspace/tao_exp/pretrained_model/backbones/"
if os.path.exists(TEST_BACKBONE_DIR_CI):
    TEST_BACKBONE_DIR = TEST_BACKBONE_DIR_CI
elif os.path.exists(TEST_BACKBONE_DIR_ORD):
    TEST_BACKBONE_DIR = TEST_BACKBONE_DIR_ORD
else:
    TEST_BACKBONE_DIR = None


TEST_TOPOLOGIES = [
    # Format: (backbone_class, filename, output_shape, expected_output)
    # ConvNeXt.
    pytest.param((convnext.convnext_tiny, None, None, None), id="convnext_tiny"),
    pytest.param((convnext_v2.convnextv2_atto, None, None, None), id="convnextv2_atto"),
    # DINOV2.
    pytest.param(
        (
            dino_v2.vit_large_patch14_dinov2_swiglu,
            "vit_large_patch14_dinov2_swiglu.ckpt",
            (1, 1024),
            [0.4775, 0.1357, -0.4266, -0.6113, 1.6212],
        ),
        id="vit_large_patch14_dinov2_swiglu",
    ),
    # DINOV3.
    pytest.param(
        (
            dino_v3.dinov3_vitb16,
            "dinov3_base_lvd1689m.safetensors",  # From TIMM.
            (1, 768),
            [0.4872, 0.5306, 0.9414, -0.0965, -0.2063],
        ),
        id="dinov3_vitb16",
    ),
    pytest.param((dino_v3.dinov3_vits16, None, None, None), id="dinov3_vits16"),
    pytest.param((dino_v3.dinov3_vits16plus, None, None, None), id="dinov3_vits16plus"),
    # EfficientViT.
    pytest.param((efficientvit.efficientvit_b0, None, None, None), id="efficientvit_b0"),
    pytest.param((efficientvit.efficientvit_l0, None, None, None), id="efficientvit_l0"),
    # FAN.
    pytest.param((fan.fan_tiny_12_p16_224, None, None, None), id="fan_tiny_12_p16_224"),
    pytest.param(
        (
            fan.fan_small_12_p4_hybrid,
            "fan_small_hybrid_nvimagenet_noprefix.ckpt",
            (1, 384),
            [0.2888, 0.0854, -0.1338, 0.7055, 0.0854],
        ),
        id="fan_small_12_p4_hybrid",
    ),
    pytest.param((fan.fan_swin_tiny_patch4_window7_224, None, None, None), id="fan_swin_tiny_patch4_window7_224"),
    # FasterViT.
    pytest.param(
        (
            fastervit.faster_vit_1_224,
            "fastervit_1_nvimagenet_noprefix.ckpt",
            (1, 640),
            [0.0181, 0.0112, 0.0044, -0.0293, -0.0020],
        ),
        id="faster_vit_1_224",
    ),
    # GCViT.
    pytest.param(
        (
            gcvit.gc_vit_xxtiny,
            "gcvit_xxtiny_nvimagenet_noprefix.ckpt",
            (1, 512),
            [-0.1061, -0.1616, -0.0865, -0.0460, -0.0773],
        ),
        id="gc_vit_xxtiny",
    ),
    # Hiera.
    pytest.param(
        (
            hiera.hiera_tiny_224,
            "timm_hiera_tiny_224.ckpt",
            (1, 49, 768),
            [0.6876, 0.7165, -0.2894, 0.1502, 1.2678],
        ),
        id="hiera_tiny_224",
    ),
    # MiT.
    pytest.param(
        (
            mit.mit_b0,
            "mit_b0.ckpt",
            (1, 256),
            [-0.8727, -0.0464, -0.5343, -0.2695, -0.1713],
        ),
        id="mit_b0",
    ),
    # OpenCLIP.
    # TODO(@hongyuc): Large backbone weights failed to be loaded in CI.
    pytest.param(
        (
            open_clip.vit_l_14_siglip_clipa_336,
            None,
            (1, 768),
            None,  # [-4.3043, -4.1365, -1.7053,  9.6923, -0.4220]
        ),
        id="vit_l_14_siglip_clipa_336",
    ),
    # RADIO.
    pytest.param(
        (
            radio.c_radio_v2_vit_base_patch16,
            "c_radio_v2_b.ckpt",
            (1, 2304),
            [0.2599, 0.0563, -0.1938, 0.0035, -0.1208],
        ),
        id="c_radio_v2_vit_base_patch16",
    ),
    pytest.param(
        (
            radio.c_radio_v3_vit_large_patch16_reg4_dinov2,
            "c_radio_v3_l.ckpt",
            (1, 3072),
            [-0.7599, -0.0459,  0.6067, -0.0879, -0.1394],
        ),
        id="c_radio_v3_vit_large_patch16_reg4_dinov2",
    ),
    pytest.param(
        (
            radio.c_radio_v3_vit_large_patch16_reg4_dinov2,
            "c_radio_v3_l.safetensors",  # From HF.
            (1, 3072),
            [-0.7599, -0.0459,  0.6067, -0.0879, -0.1394],
        ),
        id="c_radio_v3_vit_large_patch16_reg4_dinov2_safetensors",  # Test loading from safetensors.
    ),
    # ResNet.
    pytest.param(
        (
            resnet.resnet_18,
            "timm_resnet18_a1_in1k.ckpt",
            (1, 512),
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        id="resnet_18",
    ),
    pytest.param((resnet.resnet_18d, None, None, None), id="resnet_18d"),
    # Swin.
    pytest.param((swin.swin_tiny_patch4_window7_224, None, None, None), id="swin_tiny_patch4_window7_224"),
    # ViT.
    pytest.param(
        (
            vit.vit_base_patch16,
            "timm_vit_base_patch16_224.ckpt",
            (1, 768),
            [0.0194, -1.0218, 0.0009, -0.4033, -0.7687],
        ),
        id="vit_base_patch16",
    ),
]
# TODO(@hongyuc): Large backbone weights failed to be loaded in CI.
LARGE_BACKBONES = [
    # DINOV2.
    pytest.param(
        (dino_v2.vit_giant_patch14_reg4_dinov2_swiglu, None, None, None), id="vit_giant_patch14_reg4_dinov2_swiglu"
    ),
    # DINOV3.
    pytest.param(
        (
            dino_v3.dinov3_vith16plus,
            "dinov3_huge_plus_lvd1689m.safetensors",  # From TIMM.
            (1, 1280),
            [-0.1892, 0.0466, 0.1439, 0.2925, 0.0699],
        ),
        id="dinov3_vith16plus",
    ),
    pytest.param(
        (
            dino_v3.dinov3_vit7b16,
            None,
            (1, 4096),
            None,
        ),
        id="dinov3_vit7b16",
    ),
    # RADIO.
    pytest.param(
        (
            radio.c_radio_p3_vit_huge_patch16_mlpnorm,
            None,  # c_radio_p3.ckpt
            (1, 3840),
            None,  # [-0.1248, -0.0078,  0.0229,  0.2046,  0.8349]
        ),
        id="c_radio_p3_vit_huge_patch16_mlpnorm",
    ),
]
if not os.getenv("CI_PROJECT_DIR", None):
    TEST_TOPOLOGIES.extend(LARGE_BACKBONES)


@pytest.mark.cv_unit
@pytest.mark.skipif(platform.machine().lower() in ['aarch64', 'arm64', 'armv7l', 'armv8l'], reason="Skipping on ARM architecture")
@pytest.mark.parametrize("backbone_data", TEST_TOPOLOGIES)
@pytest.mark.parametrize("activation_checkpoint", [False, True], ids=["ac_off", "ac_on"])
@pytest.mark.parametrize("freeze_at", [[1], "all"], ids=["freeze_at_1", "freeze_all"])
def test_basic_usage(backbone_data, activation_checkpoint, freeze_at):
    """Test the basic usage of the backbones."""
    gc.collect()
    backbone_cls, filename, output_shape, expected_output = backbone_data

    # Common parameters.
    # Most of the tasks require `in_chans=3` and `num_classes=0`.
    kwargs = {
        "in_chans": 3,
        "num_classes": 0,
        "activation_checkpoint": activation_checkpoint,
        "freeze_at": freeze_at,
        "freeze_norm": True,
    }

    # Test the instantiation.
    try:
        backbone = backbone_cls(**kwargs)
    except NotImplementedError as e:
        if "Activation checkpointing is not implemented" in str(e):
            pytest.skip(f"{backbone_cls.__name__} doesn't support activation checkpointing.")
        raise e
    assert isinstance(backbone, BackboneBase), f"Expected BackboneBase, got {type(backbone)}"

    # Test the properties.
    assert backbone.in_chans == kwargs["in_chans"], (
        f"Expected in_chans to be {kwargs['in_chans']}, got {backbone.in_chans}"
    )
    assert backbone.num_classes == kwargs["num_classes"], (
        f"Expected num_classes to be {kwargs['num_classes']}, got {backbone.num_classes}"
    )
    assert backbone.activation_checkpoint == kwargs["activation_checkpoint"], (
        f"Expected activation_checkpoint to be {kwargs['activation_checkpoint']}, got {backbone.activation_checkpoint}"
    )
    assert backbone.freeze_at == kwargs["freeze_at"], (
        f"Expected freeze_at to be {kwargs['freeze_at']}, got {backbone.freeze_at}"
    )
    assert backbone.freeze_norm is kwargs["freeze_norm"], (
        f"Expected freeze_norm to be {kwargs['freeze_norm']}, got {backbone.freeze_norm}"
    )

    # Test the freezing.
    if kwargs["freeze_at"] == "all":
        for p in backbone.parameters():
            assert p.requires_grad is False, f"Expected all parameters to be frozen, but {p} is not."
        assert backbone.training is False, "Expected backbone to be in eval mode, but it is not."
    if isinstance(kwargs["freeze_at"], list):
        stage_dict = backbone.get_stage_dict()
        for freeze_key in kwargs["freeze_at"]:
            module = stage_dict[freeze_key]
            for p in module.parameters():
                assert p.requires_grad is False, f"Expected {freeze_key} to be frozen, but it is not."
            assert module.training is False, f"Expected {freeze_key} to be in eval mode, but it is not."

    # Test the loading if backbone weights are available.
    if TEST_BACKBONE_DIR is not None and filename is not None:
        state_dict = load_pretrained_weights(
            os.path.join(TEST_BACKBONE_DIR, filename),
            map_location="cpu",
            weights_only=False,
        )
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(msg)

    # Test the forward.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.ones(1, 3, 224, 224, device=device)
    backbone.to(device).eval()
    with torch.inference_mode():
        y = backbone.forward_pre_logits(x)
        if isinstance(y, tuple):
            y = y[0]

    # Test the numerics. We use `atol=1e-2` because of the float precision issues.
    if output_shape is not None:
        assert y.shape == output_shape, f"Expected output shape to be {output_shape}, got {y.shape}"
    if TEST_BACKBONE_DIR is not None and expected_output is not None:
        if y.dim() == 2:
            output = y[0, :5]
        elif y.dim() == 3:  # Hiera
            output = y[0, 0, :5]
        else:
            raise ValueError(f"Unexpected output shape: {y.shape}.")
        assert torch.allclose(output, torch.tensor(expected_output, device=device), atol=1e-2), (
            f"Expected output to be {expected_output}, got {output}"
        )

    # Teardown.
    del backbone, x, y
