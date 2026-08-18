# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Visual ChangeNet epoch-end status logging."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nvidia_tao_pytorch.cv.visual_changenet.classification.models import cn_pl_model as classify_module
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models import cn_pl_model as segment_module


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    ("model_module", "epoch_end_hook"),
    (
        (
            classify_module,
            classify_module.ChangeNetPlModel.on_train_epoch_end,
        ),
        (
            segment_module,
            segment_module.ChangeNetPlModel.on_train_epoch_end,
        ),
    ),
)
def test_train_epoch_end_skips_replayed_hook_without_train_loss(
    monkeypatch, model_module, epoch_end_hook
):
    """A validation-checkpoint resume may replay the hook before new batches."""
    get_status_logger = MagicMock()
    monkeypatch.setattr(
        model_module.status_logging,
        "get_status_logger",
        get_status_logger,
    )
    train_metrics = MagicMock()
    model = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"val_loss": torch.tensor(0.5)}),
        train_metrics=train_metrics,
        status_logging_dict={"existing": "value"},
        visualize_histogram=MagicMock(),
        visualize_metrics=MagicMock(),
    )

    epoch_end_hook(model)

    assert model.status_logging_dict == {"existing": "value"}
    train_metrics.compute.assert_not_called()
    train_metrics.reset.assert_not_called()
    model.visualize_histogram.assert_not_called()
    model.visualize_metrics.assert_not_called()
    get_status_logger.assert_not_called()


@pytest.mark.cv_unit
def test_classification_train_epoch_end_preserves_normal_logging(monkeypatch):
    """Classification still aggregates and publishes a completed epoch."""
    status_logger = MagicMock()
    monkeypatch.setattr(
        classify_module.status_logging,
        "get_status_logger",
        MagicMock(return_value=status_logger),
    )
    train_metrics = MagicMock()
    train_metrics.compute.return_value = {
        "total_accuracy": torch.tensor(80.0),
        "false_alarm": torch.tensor(5.0),
    }
    model = SimpleNamespace(
        trainer=SimpleNamespace(
            logged_metrics={"train_loss_epoch": torch.tensor(0.25)}
        ),
        train_metrics=train_metrics,
        status_logging_dict={},
        tensorboard=SimpleNamespace(infrequent_logging_frequency=2),
        visualize_histogram=MagicMock(),
        visualize_metrics=MagicMock(),
    )

    classify_module.ChangeNetPlModel.on_train_epoch_end(model)

    assert model.status_logging_dict == {
        "train_loss": pytest.approx(0.25),
        "train_acc": pytest.approx(80.0),
        "train_fpr": pytest.approx(5.0),
    }
    model.visualize_histogram.assert_called_once_with(logging_frequency=2)
    model.visualize_metrics.assert_called_once_with({
        "train_acc": pytest.approx(80.0),
        "train_fpr": pytest.approx(5.0),
    })
    train_metrics.reset.assert_called_once_with()
    assert status_logger.kpi == model.status_logging_dict
    status_logger.write.assert_called_once_with(
        message="Train metrics generated.",
        status_level=classify_module.status_logging.Status.RUNNING,
    )


@pytest.mark.cv_unit
def test_segmentation_train_epoch_end_preserves_normal_logging(monkeypatch):
    """Segmentation still publishes loss for a completed training epoch."""
    status_logger = MagicMock()
    monkeypatch.setattr(
        segment_module.status_logging,
        "get_status_logger",
        MagicMock(return_value=status_logger),
    )
    model = SimpleNamespace(
        trainer=SimpleNamespace(
            logged_metrics={"train_loss_epoch": torch.tensor(0.0)}
        ),
        status_logging_dict={},
    )

    segment_module.ChangeNetPlModel.on_train_epoch_end(model)

    assert model.status_logging_dict == {"train_loss": pytest.approx(0.0)}
    assert status_logger.kpi == model.status_logging_dict
    status_logger.write.assert_called_once_with(
        message="Train metrics generated.",
        status_level=segment_module.status_logging.Status.RUNNING,
    )
