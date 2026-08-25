# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet classification model contract tests."""

from unittest import mock

from omegaconf import OmegaConf
import pytest
import torch
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


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    ("difference_module", "loss", "eval_margin", "expected_margin"),
    (
        ("learnable", "ce", 0.3, 0.3),
        ("euclidean", "contrastive", 2.0, 2.0),
    ),
)
def test_training_metrics_use_eval_margin(
    monkeypatch, difference_module, loss, eval_margin, expected_margin
):
    """In-training train/val metrics threshold at eval_margin, like evaluate."""
    spec = OmegaConf.structured(ExperimentConfig())
    spec.model.classify.difference_module = difference_module
    spec.model.classify.eval_margin = eval_margin
    spec.train.classify.loss = loss
    monkeypatch.setattr(cn_pl_model, "build_model", mock.Mock(return_value=nn.Linear(2, 2)))
    model = cn_pl_model.ChangeNetPlModel(spec, dm=mock.Mock(), export=False)

    assert float(model.train_metrics.margin) == expected_margin
    assert float(model.val_metrics.margin) == expected_margin


@pytest.mark.cv_unit
def test_learnable_degenerate_margin_falls_back(monkeypatch):
    """learnable + eval_margin > 1.0 warns and falls back to 0.5 (NVBug 6662965)."""
    spec = OmegaConf.structured(ExperimentConfig())
    spec.model.classify.difference_module = "learnable"
    spec.model.classify.eval_margin = 2.0  # dataclass default: the degenerate case
    spec.train.classify.loss = "ce"
    monkeypatch.setattr(cn_pl_model, "build_model", mock.Mock(return_value=nn.Linear(2, 2)))

    with pytest.warns(UserWarning, match="eval_margin"):
        model = cn_pl_model.ChangeNetPlModel(spec, dm=mock.Mock(), export=False)

    assert float(model.val_metrics.margin) == 0.5


@pytest.mark.cv_unit
def test_val_metrics_discriminate_on_probability_scores(monkeypatch):
    """Regression: probability scores must actually move val_acc/val_fpr."""
    spec = OmegaConf.structured(ExperimentConfig())
    spec.model.classify.difference_module = "learnable"
    spec.model.classify.eval_margin = 0.3
    spec.train.classify.loss = "ce"
    monkeypatch.setattr(cn_pl_model, "build_model", mock.Mock(return_value=nn.Linear(2, 2)))
    model = cn_pl_model.ChangeNetPlModel(spec, dm=mock.Mock(), export=False)

    # 3 PASS (one above threshold -> false alarm), 2 FAIL both correctly caught.
    # With the old margin=2.0 every score is below threshold, so accuracy
    # collapses to the split PASS ratio (60%) and false_alarm to 0.
    scores = torch.tensor([0.1, 0.2, 0.6, 0.85, 0.9])
    labels = torch.tensor([0, 0, 0, 1, 1])
    model.val_metrics.update(scores, labels)
    result = model.val_metrics.compute()

    pass_ratio = 60.0
    assert result["total_accuracy"].item() == pytest.approx(80.0)
    assert result["total_accuracy"].item() != pass_ratio  # the old degenerate value
    assert result["false_alarm"].item() > 0.0
