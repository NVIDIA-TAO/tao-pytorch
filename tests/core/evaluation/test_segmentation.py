# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for segmentation probe pieces: BNHead, LR schedule, IoU accumulators."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.evaluation.segmentation import (
    BNHead,
    IoUAccumulator,
    SmallObjectIoUAccumulator,
    _get_lr,
)


@pytest.mark.unit
def test_bnhead_shape():
    """BNHead maps [B,D,h,w] → [B,num_classes,h,w]."""
    head = BNHead(in_channels=8, num_classes=5)
    out = head(torch.randn(2, 8, 4, 4))
    assert out.shape == (2, 5, 4, 4)


@pytest.mark.unit
def test_bnhead_one_step_reduces_loss():
    """A single AdamW step on a fixed batch lowers the CE loss (head trains)."""
    torch.manual_seed(0)
    head = BNHead(8, 3)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss(ignore_index=255)
    feat = torch.randn(2, 8, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    before = crit(head(feat), target)
    opt.zero_grad()
    before.backward()
    opt.step()
    after = crit(head(feat), target)
    assert after.item() < before.item()


@pytest.mark.unit
def test_lr_schedule_warmup_then_decay():
    """LR warms up linearly then decays poly(power=1) to ~0 at total_steps."""
    base = 1e-3
    assert _get_lr(0, base, warmup_steps=1500, total_steps=80_000) == pytest.approx(base * 1e-6)
    assert _get_lr(1500, base, 1500, 80_000) == pytest.approx(base)        # peak at end of warmup
    assert _get_lr(80_000, base, 1500, 80_000) == pytest.approx(0.0)       # decays to 0
    mid = _get_lr(40_000, base, 1500, 80_000)
    assert 0.0 < mid < base


@pytest.mark.unit
def test_iou_accumulator_known_case():
    """mIoU on a hand-checked 2x2 / 2-class example."""
    preds = torch.tensor([[0, 0], [1, 1]])
    targets = torch.tensor([[0, 1], [1, 1]])
    acc = IoUAccumulator(num_classes=2, ignore_index=255)
    acc.update(preds, targets)
    # class0 IoU = 1/2 ; class1 IoU = 2/3 -> mean*100
    expected = (0.5 + 2.0 / 3.0) / 2.0 * 100.0
    assert acc.miou() == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
def test_iou_accumulator_respects_ignore():
    """Ignore-index pixels are excluded from intersection/union."""
    preds = torch.tensor([[0, 1]])
    targets = torch.tensor([[0, 255]])     # second pixel ignored
    acc = IoUAccumulator(num_classes=2, ignore_index=255)
    acc.update(preds, targets)
    # only pixel (0,0): class0 perfect, class1 absent -> mIoU over present classes = 100
    assert acc.miou() == pytest.approx(100.0, abs=1e-4)


@pytest.mark.unit
def test_small_object_accumulator_empty_when_no_small():
    """With no small GT components, small-object mIoU is NaN (nothing accumulated)."""
    acc = SmallObjectIoUAccumulator(num_classes=2, ignore_index=255, area_thresh=1)
    acc.update(torch.zeros(8, 8, dtype=torch.long), torch.zeros(8, 8, dtype=torch.long))
    assert acc.miou() != acc.miou()        # NaN
