# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SegFormer confusion-matrix metric aggregation."""

import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.cv.segformer.utils.iou_metric import MeanIoUMeter


def test_get_scores_all_reduces_counts_before_miou(monkeypatch):
    """Distributed mIoU must be computed from global additive totals."""
    meter = MeanIoUMeter(n_class=2)
    local = (
        np.array([8.0, 1.0]),
        np.array([10.0, 4.0]),
        np.array([9.0, 2.0]),
        np.array([9.0, 3.0]),
    )
    remote = np.array(
        [
            [1.0, 6.0],
            [4.0, 8.0],
            [2.0, 7.0],
            [3.0, 7.0],
        ]
    )
    meter.initialize(*local)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    def fake_all_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.SUM
        tensor.add_(torch.as_tensor(remote, dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    scores, _ = meter.get_scores(sync_dist=True)
    expected, _ = MeanIoUMeter.total_area_to_metrics(
        *(np.stack(local) + remote), n_class=2
    )
    per_rank_mean = np.mean(
        [
            MeanIoUMeter.total_area_to_metrics(*local, n_class=2)[0]["miou"],
            MeanIoUMeter.total_area_to_metrics(*remote, n_class=2)[0]["miou"],
        ]
    )

    assert scores["miou"] == pytest.approx(expected["miou"])
    assert scores["miou"] != pytest.approx(per_rank_mean)
