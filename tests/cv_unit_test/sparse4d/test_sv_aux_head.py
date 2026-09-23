# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Sparse4D SV2D model and configuration plumbing."""

from platform import machine

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.config.sparse4d.model import Sparse4DModelConfig
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import Sparse4D
from nvidia_tao_pytorch.cv.sparse4d.model.sv_aux_head import SVAuxClassifier


pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests are skipped on ARM.",
)


@pytest.mark.cv_unit
@pytest.mark.config
def test_sv_and_loose_to_tight_config_defaults_are_disabled():
    """New co-training features must preserve the released default behavior."""
    config = Sparse4DModelConfig()

    assert config.cotrain_param_touch is False
    assert config.sv_scene_keywords == ["SV2D"]
    assert config.sv_aux_head.enable is False
    assert config.sv_aux_head.num_classes == 0
    assert config.sv_aux_head.fpn_strides == [4, 8, 16, 32]
    assert config.head.loose_to_tight.enable is False
    assert config.head.loose_to_tight.num_classes == 0
    assert config.head.loose_to_tight.pseudo_enable is False
    assert config.head.loose_to_tight.mlp_ckpt == ""
    assert config.head.loose_to_tight.sv_size_weight == 0.25
    assert config.head.instance_bank.reset_on_time_gap is False


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_sv_aux_loss_backpropagates_to_fpn_and_classifier():
    """Valid SV boxes train both the selected FPN map and auxiliary classifier."""
    head = SVAuxClassifier(
        in_channels=4,
        num_classes=3,
        roi_size=2,
        hidden_dim=8,
        fpn_strides=(4,),
        use_level=0,
        loss_weight=0.5,
    )
    feature = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
    data = {
        "det_boxes_2d": [
            [
                torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
                torch.tensor([[8.0, 8.0, 28.0, 28.0]]),
            ]
        ],
        "det_classes_2d": [[torch.tensor([1]), torch.tensor([2])]],
    }

    output = head.loss([feature], data)
    loss = output["loss_sv_aux_cls"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) > 0
    assert head.classifier[-1].weight.grad is not None
    assert torch.count_nonzero(head.classifier[-1].weight.grad) > 0


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_sv_aux_missing_labels_returns_feature_connected_zero():
    """An empty SV batch remains safe for autograd and contributes no loss."""
    head = SVAuxClassifier(
        in_channels=2,
        num_classes=2,
        roi_size=2,
        hidden_dim=4,
        fpn_strides=(4,),
        use_level=0,
    )
    feature = torch.randn(1, 1, 2, 4, 4, requires_grad=True)

    loss = head.loss([feature], {})["loss_sv_aux_cls"]
    assert loss.item() == 0.0
    loss.backward()
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) == 0


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_sv_aux_rejects_out_of_range_pseudo_class():
    """SV supervision fails loudly when cache labels exceed detector classes."""
    head = SVAuxClassifier(
        in_channels=2,
        num_classes=2,
        roi_size=2,
        hidden_dim=4,
        fpn_strides=(4,),
        use_level=0,
    )
    feature = torch.randn(1, 1, 2, 4, 4)
    data = {
        "det_boxes_2d": [[torch.tensor([[1.0, 1.0, 8.0, 8.0]])]],
        "det_classes_2d": [[torch.tensor([2])]],
    }

    with pytest.raises(ValueError, match="outside the configured taxonomy"):
        head.loss([feature], data)


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_sv_aux_zero_fallback_survives_nonfinite_parameters():
    """A skipped batch still returns a finite graph zero after parameter poison."""
    head = SVAuxClassifier(
        in_channels=2,
        num_classes=2,
        roi_size=2,
        hidden_dim=4,
        fpn_strides=(4,),
        use_level=0,
    )
    with torch.no_grad():
        head.classifier[0].weight[0, 0] = float("nan")
    feature = torch.randn(1, 1, 2, 4, 4, requires_grad=True)

    loss = head.loss([feature], {})["loss_sv_aux_cls"]
    assert torch.isfinite(loss)
    loss.backward()
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) == 0
    for parameter in head.classifier.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


class _TinyBackbone(nn.Module):
    def forward_feature_pyramid(self, images):
        return [images[:, :2]]


class _RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, feature_maps, metas):
        self.received = feature_maps
        return {"classification": [feature_maps[0].sum()]}


def _tiny_sparse4d(sv_aux_enabled):
    model = Sparse4D.__new__(Sparse4D)
    nn.Module.__init__(model)
    model.img_backbone = _TinyBackbone()
    model.img_neck = None
    model.depth_branch = None
    model.head = _RecordingHead()
    model.use_grid_mask = False
    model.use_deformable_func = True
    model.sv_aux_enabled = sv_aux_enabled
    return model


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_sparse4d_training_returns_raw_fpn_only_when_sv_aux_is_enabled():
    """The third training tuple item is opt-in and retains [B, N, C, H, W]."""
    images = torch.randn(1, 2, 3, 8, 8)

    default_model = _tiny_sparse4d(sv_aux_enabled=False)
    default_output = default_model.forward_train(images, {})
    assert len(default_output) == 2

    sv_model = _tiny_sparse4d(sv_aux_enabled=True)
    sv_output = sv_model.forward_train(images, {})
    assert len(sv_output) == 3
    model_outs, depths, raw_feature_maps = sv_output
    assert "classification" in model_outs
    assert depths is None
    assert len(raw_feature_maps) == 1
    assert raw_feature_maps[0].shape == (1, 2, 2, 8, 8)
    assert sv_model.head.received is not raw_feature_maps
