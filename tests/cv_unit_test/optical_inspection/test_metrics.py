# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optical inspection metric regression tests."""

import pytest
import torch

from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import AOIMetrics


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
def test_false_rates_use_their_class_denominators():
    """FPR is over pass samples and FNR is over defect samples."""
    metrics = AOIMetrics(margin=2.0)
    predictions = torch.tensor([0, 0, 0, 0, 0, 0, 3, 3, 3, 0])
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    metrics.update(predictions, targets)
    result = metrics.compute()

    assert result["total_accuracy"].item() == pytest.approx(70.0)
    assert result["defect_accuracy"].item() == pytest.approx(50.0)
    assert result["false_alarm"].item() == pytest.approx(25.0)
    assert result["false_negative"].item() == pytest.approx(50.0)


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
@pytest.mark.parametrize(
    ("predictions", "targets"),
    (
        (torch.tensor([3, 3]), torch.tensor([1, 1])),
        (torch.tensor([0, 0]), torch.tensor([0, 0])),
    ),
)
def test_false_rates_handle_single_class_batches(predictions, targets):
    """An absent negative or positive class yields a defined zero rate."""
    metrics = AOIMetrics(margin=2.0)
    metrics.update(predictions, targets)

    result = metrics.compute()

    assert torch.isfinite(result["total_accuracy"])
    assert result["false_alarm"].item() == 0.0
    assert result["false_negative"].item() == 0.0


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
def test_metrics_handle_empty_state():
    """Computing before any update returns finite zero metrics."""
    result = AOIMetrics().compute()

    assert all(metric.item() == 0.0 for metric in result.values())
