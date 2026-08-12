# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure unit tests for SegFormer pretrained-weight initialization."""

from unittest import mock

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from nvidia_tao_pytorch.cv.segformer.scripts import train
from nvidia_tao_pytorch.cv.segformer.utils import checkpoint
from nvidia_tao_pytorch.cv.segformer.utils.checkpoint import (
    PRETRAINED_LOAD_REPORT_PREFIX,
    initialize_pretrained_backbone_weights,
    initialize_pretrained_weights,
)


class _TinySegFormer(nn.Module):
    """Small state-dict fixture with SegFormer-compatible namespaces."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.linear_fuse = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
        )
        self.decoder.linear_pred = nn.Linear(2, num_classes)


class _TinyLightningModule:
    """Minimal wrapper matching ``SegFormerPlModel.model``."""

    def __init__(self, num_classes=2):
        self.model = _TinySegFormer(num_classes=num_classes)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_plain_state_dict_loads_compatible_tensors():
    """A raw model state dict is accepted without Lightning metadata."""
    pl_model = _TinyLightningModule()
    weight = torch.full_like(pl_model.model.backbone.weight, 7.0)
    bias = torch.full_like(pl_model.model.backbone.bias, -3.0)

    report = initialize_pretrained_weights(
        pl_model,
        {
            "backbone.weight": weight,
            "backbone.bias": bias,
        },
    )

    assert report.loaded_keys == ("backbone.bias", "backbone.weight")
    assert torch.equal(pl_model.model.backbone.weight, weight)
    assert torch.equal(pl_model.model.backbone.bias, bias)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_wrapped_state_dict_normalizes_lightning_and_mmseg_prefixes():
    """Wrapped PL/DDP keys and official MMSEG decoder names are normalized."""
    pl_model = _TinyLightningModule()
    backbone_weight = torch.full_like(pl_model.model.backbone.weight, 2.0)
    fuse_weight = torch.full_like(pl_model.model.decoder.linear_fuse[0].weight, 5.0)

    report = initialize_pretrained_weights(
        pl_model,
        {
            "state_dict": {
                "module.model.backbone.weight": backbone_weight,
                "model.decode_head.linear_fuse.conv.weight": fuse_weight,
            }
        },
    )

    assert report.loaded_keys == (
        "backbone.weight",
        "decoder.linear_fuse.0.weight",
    )
    assert torch.equal(pl_model.model.backbone.weight, backbone_weight)
    assert torch.equal(pl_model.model.decoder.linear_fuse[0].weight, fuse_weight)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_mismatched_decoder_head_is_skipped():
    """A class-count mismatch leaves the new decoder head initialized."""
    pl_model = _TinyLightningModule(num_classes=2)
    original_head = pl_model.model.decoder.linear_pred.weight.detach().clone()
    backbone_weight = torch.full_like(pl_model.model.backbone.weight, 4.0)
    foreign_head = torch.ones(19, 2)

    report = initialize_pretrained_weights(
        pl_model,
        {
            "backbone.weight": backbone_weight,
            "decode_head.linear_pred.weight": foreign_head,
        },
    )

    assert report.loaded_keys == ("backbone.weight",)
    assert report.shape_mismatched_keys == ("decoder.linear_pred.weight",)
    assert torch.equal(pl_model.model.backbone.weight, backbone_weight)
    assert torch.equal(pl_model.model.decoder.linear_pred.weight, original_head)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_no_compatible_tensor_fails_without_mutating_model():
    """An unrelated checkpoint cannot silently start from random weights."""
    pl_model = _TinyLightningModule()
    original_state = {
        key: value.detach().clone()
        for key, value in pl_model.model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="no tensors compatible"):
        initialize_pretrained_weights(
            pl_model,
            {"unrelated.weight": torch.ones(2, 2)},
        )

    for key, value in pl_model.model.state_dict().items():
        assert torch.equal(value, original_state[key])


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_backbone_loader_strips_only_leading_checkpoint_namespaces():
    """Official FAN wrappers map to a bare backbone without altering internals."""
    backbone = nn.Module()
    backbone.patch_embed = nn.Module()
    backbone.patch_embed.backbone = nn.Linear(2, 2)
    weight = torch.full_like(backbone.patch_embed.backbone.weight, 11.0)
    bias = torch.full_like(backbone.patch_embed.backbone.bias, -5.0)

    report = initialize_pretrained_backbone_weights(
        backbone,
        {
            "model.backbone.patch_embed.backbone.weight": weight,
            "module.model.backbone.patch_embed.backbone.bias": bias,
            "head.weight": torch.ones(3, 2),
        },
    )

    assert report.loaded_keys == (
        "patch_embed.backbone.bias",
        "patch_embed.backbone.weight",
    )
    assert report.unmatched_keys == ("head.weight",)
    assert torch.equal(backbone.patch_embed.backbone.weight, weight)
    assert torch.equal(backbone.patch_embed.backbone.bias, bias)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_backbone_loader_accepts_bare_keys_and_rejects_zero_matches():
    """Bare PTMs remain supported and unrelated PTMs fail closed."""
    backbone = nn.Linear(2, 2)
    weight = torch.full_like(backbone.weight, 13.0)

    report = initialize_pretrained_backbone_weights(
        backbone,
        {"weight": weight},
    )
    assert report.loaded_keys == ("weight",)
    assert torch.equal(backbone.weight, weight)

    with pytest.raises(RuntimeError, match="configured backbone"):
        initialize_pretrained_backbone_weights(
            backbone,
            {"decoder.weight": torch.ones(2, 2)},
        )


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_positive_load_emits_compact_structured_receipt(monkeypatch):
    """Qualification can prove a nonzero exact-component tensor load."""
    backbone = nn.Linear(2, 2)
    info = mock.Mock()
    monkeypatch.setattr(checkpoint.logging, "info", info)

    report = initialize_pretrained_backbone_weights(
        backbone,
        {"backbone.weight": torch.ones_like(backbone.weight)},
    )

    receipt_calls = [
        call
        for call in info.call_args_list
        if call.args and call.args[0] == "%s%s"
    ]
    assert len(receipt_calls) == 1
    prefix, payload = receipt_calls[0].args[1:]
    assert prefix == PRETRAINED_LOAD_REPORT_PREFIX
    assert '"component":"backbone"' in payload
    assert '"loaded_tensor_count":1' in payload
    assert len(report.loaded_keys) == 1


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_train_initializes_pretrained_weights_and_preserves_resume_checkpoint(monkeypatch):
    """Training uses the PTM initializer and reserves ckpt_path for resume."""
    experiment_config = OmegaConf.create(
        {
            "train": {
                "resume_training_checkpoint_path": "/tmp/resume.ckpt",
                "pretrained_model_path": "/tmp/pretrained.pth",
                "num_nodes": 1,
                "tensorboard": {"enabled": False},
                "use_distributed_sampler": False,
                "sync_batchnorm": False,
            },
            "dataset": {"segment": {}},
        }
    )
    pl_model = mock.Mock(name="pl_model")
    trainer = mock.Mock(name="trainer")
    model_class = mock.Mock(return_value=pl_model)
    model_class.load_from_checkpoint.side_effect = AssertionError(
        "pretrained weights must not use Lightning checkpoint restoration"
    )
    initializer = mock.Mock()
    trainer_class = mock.Mock(return_value=trainer)

    monkeypatch.setattr(
        train,
        "initialize_train_experiment",
        lambda *_: (
            "/tmp/resume.ckpt",
            {"devices": [0], "enable_checkpointing": False},
        ),
    )
    monkeypatch.setattr(train, "SFDataModule", mock.Mock())
    monkeypatch.setattr(train, "SegFormerPlModel", model_class)
    monkeypatch.setattr(train, "initialize_pretrained_weights", initializer)
    monkeypatch.setattr(train, "Trainer", trainer_class)

    train.run_experiment(experiment_config, key="")

    model_class.assert_called_once_with(experiment_config)
    model_class.load_from_checkpoint.assert_not_called()
    initializer.assert_called_once_with(pl_model, "/tmp/pretrained.pth")
    trainer.fit.assert_called_once()
    assert trainer.fit.call_args.kwargs["ckpt_path"] == "/tmp/resume.ckpt"
