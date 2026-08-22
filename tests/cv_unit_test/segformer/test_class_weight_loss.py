# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SegFormer class-weighted cross entropy."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("pytorch_lightning")

from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def make_loss_only_model(weights, num_classes=4, loss="ce", iou_weight=0.5):
    """Construct only the state needed by _build_criterion/_compute_loss."""
    model = SegFormerPlModel.__new__(SegFormerPlModel)
    torch.nn.Module.__init__(model)
    model.train_config = SimpleNamespace(segment={"loss": loss, "iou_weight": iou_weight})
    model.n_class = num_classes
    model.weights = tuple(weights)
    model._build_criterion()
    return model


def test_configured_class_weights_are_applied_to_cross_entropy():
    model = make_loss_only_model((1.0, 2.0, 3.0, 4.0))
    prediction = torch.tensor(
        [[[[3.0, 0.0]], [[0.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1]]], dtype=torch.long)

    actual = model._compute_loss(prediction, target)
    expected = F.cross_entropy(
        prediction,
        target,
        weight=prediction.new_tensor((1.0, 2.0, 3.0, 4.0)),
        reduction="none",
    ).mean()
    unweighted = F.cross_entropy(prediction, target, reduction="none").mean()

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, unweighted)


def test_nonlegacy_weight_count_mismatch_fails_fast():
    with pytest.raises(ValueError, match="exactly 4"):
        make_loss_only_model((1.0, 2.0), num_classes=4)


def test_empty_weights_select_unweighted_cross_entropy():
    model = make_loss_only_model((), num_classes=4)
    prediction = torch.randn(1, 4, 3, 3)
    target = torch.tensor([[[0, 1, 2], [3, 2, 1], [0, 0, 3]]], dtype=torch.long)

    assert model.class_weights is None
    assert torch.allclose(
        model._compute_loss(prediction, target),
        F.cross_entropy(prediction, target, reduction="none").mean(),
    )


def test_legacy_five_value_default_remains_unweighted_when_count_mismatches():
    model = make_loss_only_model((0.5, 0.5, 0.5, 0.8, 1.0), num_classes=4)
    assert model.class_weights is None


def test_ce_mmiou_combines_weighted_ce_and_soft_minimax_iou():
    model = make_loss_only_model((1.0, 2.0, 3.0, 4.0), loss="ce_mmiou", iou_weight=0.25)
    prediction = torch.randn(1, 4, 3, 3)
    target = torch.tensor([[[0, 1, 2], [3, 2, 1], [0, 0, 3]]], dtype=torch.long)

    weighted_ce = F.cross_entropy(
        prediction,
        target,
        weight=prediction.new_tensor((1.0, 2.0, 3.0, 4.0)),
        reduction="none",
    ).mean()
    expected = 0.75 * weighted_ce + 0.25 * model._auxiliary_loss(prediction, target)

    assert torch.allclose(model._compute_loss(prediction, target), expected)


def test_ce_mmiou_rejects_invalid_iou_weight():
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_loss_only_model((1.0, 1.0, 1.0, 1.0), loss="ce_mmiou", iou_weight=1.5)


@pytest.mark.parametrize("loss_name", ["ce_lovasz", "ce_boundary"])
def test_additional_composite_losses_are_finite_and_differentiable(loss_name):
    model = make_loss_only_model((1.0, 2.0, 3.0, 4.0), loss=loss_name, iou_weight=0.25)
    prediction = torch.randn(1, 4, 4, 4, requires_grad=True)
    target = torch.tensor(
        [[[0, 0, 1, 1], [0, 2, 2, 1], [3, 2, 2, 1], [3, 3, 3, 1]]],
        dtype=torch.long,
    )

    loss = model._compute_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
