# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure mocked-distributed tests for SegFormer metric finalization."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.cv.segformer.model import segformer_pl_model
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel
from nvidia_tao_pytorch.cv.segformer.utils import iou_metric
from nvidia_tao_pytorch.cv.segformer.utils.iou_metric import MeanIoUMeter


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_scores_sum_sufficient_statistics_before_metric_calculation(monkeypatch):
    """DDP metrics are calculated from globally summed pixel counts."""
    meter = MeanIoUMeter(n_class=2)
    local_statistics = (
        np.array([2.0, 1.0]),
        np.array([2.0, 2.0]),
        np.array([2.0, 1.0]),
        np.array([2.0, 2.0]),
    )
    remote_statistics = np.array(
        (
            np.array([0.0, 3.0]),
            np.array([2.0, 3.0]),
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0]),
        )
    )
    meter.initialize(*local_statistics)
    all_reduce = mock.Mock(
        side_effect=lambda tensor, **_: tensor.add_(
            torch.as_tensor(remote_statistics, dtype=tensor.dtype)
        )
    )
    monkeypatch.setattr(
        iou_metric,
        "is_dist_avail_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(iou_metric.dist, "all_reduce", all_reduce)

    scores, means = meter.get_scores(device=torch.device("cpu"))

    expected_scores, expected_means = MeanIoUMeter.total_area_to_metrics(
        *(np.array(local_statistics) + remote_statistics),
        n_class=2,
    )
    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["op"] == iou_metric.dist.ReduceOp.SUM
    assert scores == pytest.approx(expected_scores)
    assert means == pytest.approx(expected_means)
    assert scores["miou"] != pytest.approx(
        MeanIoUMeter.total_area_to_metrics(*local_statistics, n_class=2)[0]["miou"]
    )


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_scores_do_not_reduce_without_distributed_initialization(monkeypatch):
    """Single-process metric calculation retains the local fast path."""
    meter = MeanIoUMeter(n_class=2)
    statistics = (
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
        np.array([1.0, 3.0]),
        np.array([2.0, 2.0]),
    )
    meter.initialize(*statistics)
    monkeypatch.setattr(
        iou_metric,
        "is_dist_avail_and_initialized",
        lambda: False,
    )
    all_reduce = mock.Mock()
    monkeypatch.setattr(iou_metric.dist, "all_reduce", all_reduce)

    scores, means = meter.get_scores()

    expected_scores, expected_means = MeanIoUMeter.total_area_to_metrics(
        *statistics,
        n_class=2,
    )
    all_reduce.assert_not_called()
    assert scores == pytest.approx(expected_scores)
    assert means == pytest.approx(expected_means)


@pytest.mark.cv_unit
@pytest.mark.segformer
def test_status_kpis_are_written_only_by_global_rank_zero(monkeypatch):
    """Nonzero ranks cannot race the global writer for status.json."""
    status_logger = mock.Mock()
    monkeypatch.setattr(
        segformer_pl_model.status_logging,
        "get_status_logger",
        lambda: status_logger,
    )
    module = SimpleNamespace(status_logging_dict={"previous": 1})

    monkeypatch.setattr(segformer_pl_model, "get_global_rank", lambda: 1)
    SegFormerPlModel._write_status_kpis(
        module,
        {"val_miou": 0.5},
        "Eval metrics generated.",
    )

    assert module.status_logging_dict == {"previous": 1}
    status_logger.write.assert_not_called()

    monkeypatch.setattr(segformer_pl_model, "get_global_rank", lambda: 0)
    SegFormerPlModel._write_status_kpis(
        module,
        {"val_miou": 0.75},
        "Eval metrics generated.",
    )

    assert module.status_logging_dict == {"val_miou": 0.75}
    assert status_logger.kpi == {"val_miou": 0.75}
    status_logger.write.assert_called_once_with(
        message="Eval metrics generated.",
        status_level=segformer_pl_model.status_logging.Status.RUNNING,
    )
