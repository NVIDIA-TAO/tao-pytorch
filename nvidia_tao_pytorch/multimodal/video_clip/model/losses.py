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

"""Losses for CLIP-compatible multimodal models."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _distributed_is_ready(world_size):
    """Return True when distributed all-gather can be used."""
    return (
        world_size > 1 and
        dist.is_available() and
        dist.is_initialized()
    )


def _all_gather_with_grad(tensor):
    """All-gather tensors while preserving gradients."""
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor.contiguous()), dim=0)


def _all_gather_no_grad(tensor, world_size):
    """All-gather tensors that do not require gradients."""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _normalize_targets(similarity, idx=None):
    """Build InternVideo2 VTC targets from sample ids."""
    if idx is None:
        return torch.eye(
            similarity.shape[0],
            dtype=similarity.dtype,
            device=similarity.device,
        )

    idx = idx.to(device=similarity.device).view(-1, 1)
    targets = torch.eq(idx, idx.T).to(dtype=similarity.dtype)
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)


def internvideo2_similarity(
    vision_features,
    text_features,
    logit_scale,
    agg_method="mean",
):
    """Compute InternVideo2 video-text similarity.

    This mirrors OpenGVLab InternVideo2's ``get_sim`` semantics while using
    TAO's CLIP convention that ``logit_scale`` is the reciprocal temperature.
    """
    vision_features = F.normalize(vision_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logit_scale = logit_scale.to(
        device=vision_features.device,
        dtype=vision_features.dtype,
    )

    if vision_features.ndim == 3:
        sim_v2t = torch.einsum(
            "mld,nd->mln", vision_features, text_features
        ) * logit_scale
        sim_t2v = torch.einsum(
            "nd,mld->nlm", text_features, vision_features
        ) * logit_scale
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(dim=1)
            sim_t2v = sim_t2v.mean(dim=1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(dim=1)[0]
            sim_t2v = sim_t2v.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")
    elif text_features.ndim == 3:
        sim_v2t = torch.einsum(
            "nd,mld->nlm", vision_features, text_features
        ) * logit_scale
        sim_t2v = torch.einsum(
            "nld,md->nlm", text_features, vision_features
        ) * logit_scale
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(dim=1)
            sim_t2v = sim_t2v.mean(dim=1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(dim=1)[0]
            sim_t2v = sim_t2v.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")
    else:
        sim_v2t = vision_features @ text_features.T * logit_scale
        sim_t2v = sim_v2t.T

    return sim_v2t, sim_t2v


class InternVideo2VTCLoss(nn.Module):
    """InternVideo2 video-text contrastive loss.

    This is the TAO equivalent of OpenGVLab InternVideo2's
    ``VTC_VTM_Loss.vtc_loss``. It keeps the ``idx`` positive-pair semantics
    needed for retrieval fine-tuning, including duplicate ids for
    multi-positive batches.
    """

    def __init__(self, rank=0, world_size=1, agg_method="mean"):
        """Initialize the loss."""
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.agg_method = agg_method

    def forward(self, vision_features, text_features, logit_scale, idx=None):
        """Compute bidirectional video-text contrastive loss."""
        if self.world_size > 1 and not _distributed_is_ready(self.world_size):
            raise RuntimeError(
                "Distributed InternVideo2 VTC loss requested with "
                f"world_size={self.world_size}, but torch.distributed is not "
                "initialized."
            )

        if _distributed_is_ready(self.world_size):
            vision_features = _all_gather_with_grad(vision_features)
            text_features = _all_gather_with_grad(text_features)
            if idx is not None:
                idx = _all_gather_no_grad(idx, self.world_size)

        sim_v2t, sim_t2v = internvideo2_similarity(
            vision_features,
            text_features,
            logit_scale,
            agg_method=self.agg_method,
        )
        targets = _normalize_targets(sim_v2t, idx=idx)

        loss_i2t = -torch.sum(
            F.log_softmax(sim_v2t, dim=1) * targets,
            dim=1,
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2v, dim=1) * targets,
            dim=1,
        ).mean()
        return (loss_i2t + loss_t2i) / 2
