# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for InternVideo2 VTC loss semantics."""

import pytest
import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.video_clip.model.losses import (
    InternVideo2VTCLoss,
    internvideo2_similarity,
)


def _reference_similarity(vision_features, text_features, temp, agg_method="mean"):
    """Reference equivalent of OpenGVLab InternVideo2 get_sim."""
    vision_features = F.normalize(vision_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    if vision_features.ndim == 3:
        sim_v2t = torch.einsum(
            "mld,nd->mln", vision_features, text_features
        ) / temp
        sim_t2v = torch.einsum(
            "nd,mld->nlm", text_features, vision_features
        ) / temp
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(dim=1)
            sim_t2v = sim_t2v.mean(dim=1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(dim=1)[0]
            sim_t2v = sim_t2v.max(dim=1)[0]
    elif text_features.ndim == 3:
        sim_v2t = torch.einsum(
            "nd,mld->nlm", vision_features, text_features
        ) / temp
        sim_t2v = torch.einsum(
            "nld,md->nlm", text_features, vision_features
        ) / temp
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(dim=1)
            sim_t2v = sim_t2v.mean(dim=1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(dim=1)[0]
            sim_t2v = sim_t2v.max(dim=1)[0]
    else:
        sim_v2t = vision_features @ text_features.T / temp
        sim_t2v = sim_v2t.T
    return sim_v2t, sim_t2v


def _reference_targets(similarity, idx=None):
    """Reference equivalent of OpenGVLab InternVideo2 target mask."""
    if idx is None:
        targets = torch.zeros_like(similarity)
        targets.fill_diagonal_(1)
        return targets
    idx = idx.view(-1, 1)
    targets = torch.eq(idx, idx.T).to(similarity.dtype)
    return targets / targets.sum(dim=1, keepdim=True)


def _reference_vtc_loss(vision_features, text_features, idx, temp=0.01):
    """Reference equivalent of OpenGVLab InternVideo2 VTC loss."""
    sim_v2t, sim_t2v = _reference_similarity(
        vision_features, text_features, temp
    )
    targets = _reference_targets(sim_v2t, idx=idx)
    loss_i2t = -torch.sum(
        F.log_softmax(sim_v2t, dim=1) * targets,
        dim=1,
    ).mean()
    loss_t2i = -torch.sum(
        F.log_softmax(sim_t2v, dim=1) * targets,
        dim=1,
    ).mean()
    return (loss_i2t + loss_t2i) / 2


@pytest.mark.multimodal_unit
class TestInternVideo2VTCLoss:
    """Test InternVideo2 VTC loss against source-equivalent logic."""

    def test_unique_idx_matches_reference_loss(self):
        """Unique sample ids should match the original diagonal case."""
        torch.manual_seed(7)
        vision = torch.randn(4, 8, requires_grad=True)
        text = torch.randn(4, 8, requires_grad=True)
        idx = torch.arange(4)
        logit_scale = torch.tensor(100.0)

        actual = InternVideo2VTCLoss()(vision, text, logit_scale, idx=idx)
        expected = _reference_vtc_loss(vision, text, idx, temp=0.01)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_duplicate_idx_multi_positive_matches_reference_loss(self):
        """Duplicate ids should produce multi-positive normalized targets."""
        torch.manual_seed(11)
        vision = torch.randn(5, 8, requires_grad=True)
        text = torch.randn(5, 8, requires_grad=True)
        idx = torch.tensor([0, 1, 1, 2, 2])
        logit_scale = torch.tensor(100.0)

        actual = InternVideo2VTCLoss()(vision, text, logit_scale, idx=idx)
        expected = _reference_vtc_loss(vision, text, idx, temp=0.01)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_three_dimensional_vision_features_match_reference_loss(self):
        """Video-token features should use InternVideo2 aggregation."""
        torch.manual_seed(13)
        vision = torch.randn(3, 2, 8, requires_grad=True)
        text = torch.randn(3, 8, requires_grad=True)
        idx = torch.arange(3)
        logit_scale = torch.tensor(100.0)

        actual = InternVideo2VTCLoss()(vision, text, logit_scale, idx=idx)
        expected = _reference_vtc_loss(vision, text, idx, temp=0.01)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_similarity_matches_reference_temperature_convention(self):
        """TAO logit_scale should be equivalent to InternVideo2 1 / temp."""
        torch.manual_seed(17)
        vision = torch.randn(4, 8)
        text = torch.randn(4, 8)

        actual = internvideo2_similarity(
            vision, text, logit_scale=torch.tensor(100.0)
        )
        expected = _reference_similarity(vision, text, temp=0.01)

        assert torch.allclose(actual[0], expected[0], atol=1e-6)
        assert torch.allclose(actual[1], expected[1], atol=1e-6)

    def test_loss_backpropagates_to_features_and_scale(self):
        """Loss should produce gradients for features and temperature scale."""
        torch.manual_seed(19)
        vision = torch.randn(4, 8, requires_grad=True)
        text = torch.randn(4, 8, requires_grad=True)
        logit_scale = torch.tensor(100.0, requires_grad=True)
        idx = torch.tensor([0, 1, 1, 2])

        loss = InternVideo2VTCLoss()(vision, text, logit_scale, idx=idx)
        loss.backward()

        assert vision.grad is not None
        assert text.grad is not None
        assert logit_scale.grad is not None
        assert torch.isfinite(vision.grad).all()
        assert torch.isfinite(text.grad).all()
        assert torch.isfinite(logit_scale.grad)
