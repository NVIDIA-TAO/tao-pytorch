# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet classification model contract tests."""

from unittest import mock

from omegaconf import OmegaConf
import pytest
import torch.nn as nn

from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.classification.models import cn_pl_model


def _model(monkeypatch, loss, difference_module="learnable", export=False):
    spec = OmegaConf.structured(ExperimentConfig())
    spec.model.classify.difference_module = difference_module
    spec.train.classify.loss = loss
    monkeypatch.setattr(cn_pl_model, "build_model", mock.Mock(return_value=nn.Linear(2, 2)))
    return cn_pl_model.ChangeNetPlModel(spec, dm=mock.Mock(), export=export)


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    ("difference_module", "training_loss", "irrelevant_loss"),
    (
        ("learnable", "ce", "contrastive"),
        ("euclidean", "contrastive", "ce"),
    ),
)
def test_non_training_load_ignores_irrelevant_loss_pairing(
    monkeypatch, difference_module, training_loss, irrelevant_loss
):
    """Non-training specs can load checkpoints regardless of configured train loss."""
    trained_model = _model(
        monkeypatch,
        loss=training_loss,
        difference_module=difference_module,
    )
    checkpoint_state = trained_model.state_dict()

    inference_model = _model(
        monkeypatch,
        loss=irrelevant_loss,
        difference_module=difference_module,
    )
    inference_model.load_state_dict(checkpoint_state, strict=True)
    inference_model.setup("test")
    inference_model.setup("predict")

    export_model = _model(
        monkeypatch,
        loss=irrelevant_loss,
        difference_module=difference_module,
        export=True,
    )
    export_model.load_state_dict(checkpoint_state, strict=True)


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    ("difference_module", "loss", "expected_loss"),
    (
        ("learnable", "contrastive", "ce"),
        ("euclidean", "ce", "contrastive"),
    ),
)
def test_training_rejects_loss_that_does_not_match_architecture(
    monkeypatch, difference_module, loss, expected_loss
):
    """The loss/architecture contract remains enforced for training."""
    model = _model(monkeypatch, loss=loss, difference_module=difference_module)

    with pytest.raises(
        ValueError,
        match=f"requires train.classify.loss='{expected_loss}'",
    ):
        model.setup("fit")
