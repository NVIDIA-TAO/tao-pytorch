# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the framework-independent Loose-to-Tight utilities."""

import pytest
import torch

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_loss import (
    loose_and_features,
    loose_to_tight_2d_loss,
)
from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_match import (
    aggregate_consistency,
    match_camera,
    pairwise_giou,
)
from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    NUM_GEOM_FEATURES,
    LooseToTightMLP,
    assemble_features,
)


pytestmark = [pytest.mark.cv_unit, pytest.mark.sparse4d]


def test_feature_shape_and_mlp_output_range():
    """Feature encoding and predicted side scales follow their public contract."""
    num_classes = 7
    class_ids = torch.tensor([0, 6])
    extents = torch.tensor([[1.0, 2.0, 3.0], [0.5, 4.0, 1.5]])
    relative_yaw = torch.tensor([0.0, 0.5])
    elevation = torch.tensor([0.1, -0.2])
    bearing = torch.tensor([-0.3, 0.4])
    loose_boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [100.0, 120.0, 180.0, 200.0]])
    distances = torch.tensor([10.0, 20.0])

    features = assemble_features(
        class_ids,
        extents,
        relative_yaw,
        elevation,
        bearing,
        loose_boxes,
        (200.0, 240.0),
        distances,
        num_classes=num_classes,
    )

    assert features.shape == (2, num_classes + NUM_GEOM_FEATURES)
    torch.testing.assert_close(
        features[:, :num_classes].sum(dim=-1),
        torch.ones(2),
    )
    assert features[0, 0] == 1.0
    assert features[1, 6] == 1.0
    assert torch.isfinite(features).all()

    model = LooseToTightMLP(
        num_classes=num_classes,
        hidden_dim=16,
        shrink_min=0.6,
        shrink_max=0.95,
    )
    side_scales = model(features)
    assert side_scales.shape == (2, 4)
    assert torch.all(side_scales >= 0.6)
    assert torch.all(side_scales <= 0.95)

    tight_boxes = model.apply_to_loose(loose_boxes, side_scales)
    assert torch.all(tight_boxes[..., :2] >= loose_boxes[..., :2])
    assert torch.all(tight_boxes[..., 2:] <= loose_boxes[..., 2:])
    recovered_scales = model.shrinkage_target(loose_boxes, tight_boxes)
    torch.testing.assert_close(recovered_scales, side_scales)


def test_projection_and_loss_are_differentiable():
    """The 2D loss propagates finite, non-zero gradients into decoded 3D boxes."""
    boxes7 = torch.tensor(
        [[0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.2]],
        requires_grad=True,
    )
    projection = torch.tensor(
        [
            [100.0, 0.0, 50.0, 0.0],
            [0.0, 100.0, 40.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ego_to_camera = torch.eye(4)
    image_size = torch.tensor([100.0, 80.0])

    loose_boxes, features, valid = loose_and_features(
        boxes7,
        torch.tensor([2]),
        projection,
        ego_to_camera,
        image_size,
        num_classes=7,
    )
    assert valid.tolist() == [True]
    assert loose_boxes.shape == (1, 4)
    assert features.shape == (1, 7 + NUM_GEOM_FEATURES)
    assert torch.all(loose_boxes >= 0.0)
    assert torch.all(loose_boxes <= torch.tensor([100.0, 80.0, 100.0, 80.0]))

    prediction = LooseToTightMLP.apply_to_loose(
        loose_boxes,
        loose_boxes.new_full((1, 4), 0.8),
    )
    visible_target = prediction.detach() + prediction.new_tensor(
        [[1.5, -1.0, 2.0, -0.5]]
    )
    loss = loose_to_tight_2d_loss(
        prediction,
        visible_target,
        occ_weight=torch.tensor([0.75]),
        img_diag=torch.tensor([128.0]),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert boxes7.grad is not None
    assert torch.isfinite(boxes7.grad).all()
    assert boxes7.grad.abs().sum().item() > 0.0


def _camera_matches(query_ids, scores):
    """Build a compact synthetic matcher result for aggregation tests."""
    count = len(query_ids)
    return {
        "q": torch.tensor(query_ids, dtype=torch.long),
        "box": torch.tensor([[10.0, 10.0, 50.0, 50.0]]).repeat(count, 1),
        "cls": torch.full((count,), 2, dtype=torch.long),
        "score": torch.tensor(scores),
        "giou": torch.ones(count),
    }


def test_camera_match_and_cross_camera_aggregation():
    """Matching is one-to-one and aggregation gates and de-duplicates queries."""
    amodal = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [10.0, 10.0, 50.0, 50.0],
            [100.0, 100.0, 140.0, 140.0],
        ]
    )
    query_probabilities = torch.full((3, 4), 0.05)
    query_probabilities[0, 0] = 0.9
    query_probabilities[1, 0] = 0.8
    query_probabilities[2, 1] = 0.95
    matches = match_camera(
        amodal,
        torch.ones(3, dtype=torch.bool),
        query_probabilities,
        torch.tensor([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 140.0, 140.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0.9, 0.85]),
        img_diag=200.0,
        giou_thr=0.5,
    )

    assert matches["q"].shape[0] == 2
    assert len(set(matches["q"].tolist()) & {0, 1}) == 1
    assert 2 in matches["q"].tolist()
    torch.testing.assert_close(
        pairwise_giou(matches["box"], matches["box"]).diagonal(),
        torch.ones(2),
    )

    wrong_class = match_camera(
        amodal[:1],
        torch.ones(1, dtype=torch.bool),
        query_probabilities[:1],
        amodal[:1],
        torch.tensor([3]),
        torch.tensor([0.9]),
        img_diag=200.0,
        giou_thr=0.5,
        class_gate=True,
    )
    assert wrong_class["q"].numel() == 0

    camera_zero = _camera_matches([0, 1, 2], [0.95, 0.7, 0.8])
    camera_one = _camera_matches([0, 1], [0.9, 0.65])
    boxes7 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.2, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    keep_pairs, class_target, class_weight = aggregate_consistency(
        [camera_zero, camera_one],
        num_queries=3,
        boxes7=boxes7,
        min_cams=2,
        dedup_dist=1.0,
    )

    assert class_target.tolist() == [2, -1, -1]
    assert class_weight[0] == pytest.approx(0.95)
    assert class_weight[1:].tolist() == [0.0, 0.0]
    assert len(keep_pairs) == 2
    assert all(pair[0] == 0 for pair in keep_pairs)
