# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Metrics for evaluator."""

import torch


def ap_per_mask(pstats: torch.Tensor) -> torch.Tensor:
    """Compute AP@[0.5:0.95] treating all masks as a single class.

    Args:
        pstats (torch.Tensor): Shape (N, niou + 2), where:
            - [:niou]: TP at different IoUs
            - [-2]: confidence
            - [-1]: predicted class (ignored)

    Returns:
        torch.Tensor: Shape (niou,), average precision per IoU threshold.
    """
    device = pstats.device
    niou = pstats.shape[1] - 2
    conf = pstats[:, -2]

    # Sort by descending confidence
    pstats = pstats[torch.argsort(-conf)]

    # Ground truth count estimate (max TP across IoUs)
    n_gt = pstats[:, :niou].sum(dim=0).max().item()
    if n_gt == 0:
        return torch.zeros(niou, device=device)

    tps = pstats[:, :niou]
    fps = 1.0 - tps

    tps_cum = torch.cumsum(tps, dim=0)
    fps_cum = torch.cumsum(fps, dim=0)

    recall_curve = tps_cum / (n_gt + 1e-16)
    precision_curve = tps_cum / (tps_cum + fps_cum + 1e-16)

    ap = torch.zeros(niou, device=device)

    for j in range(niou):
        r = recall_curve[:, j]
        p = precision_curve[:, j]

        # Precision envelope
        r = torch.cat(
            [torch.tensor([0.0], device=device), r, torch.tensor([1.0], device=device)]
        )
        p = torch.cat(
            [torch.tensor([0.0], device=device), p, torch.tensor([0.0], device=device)]
        )
        p[1:] = torch.maximum(p[1:], p[:-1])
        idx = (r[1:] != r[:-1]).nonzero(as_tuple=False).flatten()
        ap[j] = torch.sum((r[idx + 1] - r[idx]) * p[idx + 1])

    return ap
