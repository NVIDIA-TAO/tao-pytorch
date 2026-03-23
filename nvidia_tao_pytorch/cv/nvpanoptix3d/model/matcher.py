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

"""Modules to compute the matching cost and solve the corresponding LSAP."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import autocast

from nvidia_tao_pytorch.cv.mask2former.utils.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute pairwise Dice loss between predicted masks and target masks.

    Args:
        inputs: Logits tensor of shape (N_pred, P) where P is the number
            of pixels/points (already flattened).
        targets: Binary targets tensor of shape (N_tgt, P).

    Returns:
        Tensor of shape (N_pred, N_tgt) containing Dice loss values.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute pairwise sigmoid cross-entropy cost between predictions and targets.

    This returns a cost matrix of shape (N_pred, N_tgt) by aggregating
    per-location BCE losses over the spatial dimension.

    Args:
        inputs: Logits tensor of shape (N_pred, P).
        targets: Binary targets tensor of shape (N_tgt, P).

    Returns:
        Tensor of shape (N_pred, N_tgt) containing CE costs, normalized by P.
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: Logits tensor of shape (N_pred, P).
        targets: Binary targets tensor of shape (N_tgt, P).
        alpha: Weighting factor in range (0, 1) to balance positives vs negatives.
        gamma: Focusing parameter for down-weighting easy examples.

    Returns:
        Tensor of shape (N_pred, N_tgt) containing focal costs, normalized by P.
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) \
        + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))

    return loss / hw


class HungarianMatcher(nn.Module):
    """Hungarian matcher for bipartite assignment of predictions to targets.

    This module builds a cost matrix per image using classification, mask BCE,
    and Dice costs, and solves a linear sum assignment (LSAP) to produce a
    1-to-1 matching between predictions (queries) and ground-truth instances.
    Unmatched queries are treated as "no-object".
    """

    def __init__(
        self, cost_class: float = 1, cost_mask: float = 1,
        cost_dice: float = 1, num_points: int = 0,
        use_point_sample: bool = False
    ):
        """Create a Hungarian matcher.

        Args:
            cost_class: Relative weight for the classification term.
            cost_mask: Relative weight for the mask cross-entropy term.
            cost_dice: Relative weight for the Dice term.
            num_points: Number of points to sample per mask when using point sampling.
            use_point_sample: If True, compute mask costs on randomly sampled points
                (with shared points per image) for efficiency.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.use_point_sample = use_point_sample
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """Compute matching indices with a memory-efficient per-image loop.

        Args:
            outputs: Dict containing at least pred_logits ((B, Q, C+1))
                and pred_masks ((B, Q, H, W)).
            targets: List of dicts, each containing:
                - labels: (N_i,) target class ids
                - masks: (N_i, H_i, W_i) target masks

        Returns:
            List of (src_idx, tgt_idx) tuples (one per batch element), where
            indices are 1D int64 tensors.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes+1]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]  # [1,2,3, ……]
            tgt_mask = targets[b]["masks"].to(out_mask)  # [c, h, w] c = len(tgt_ids)

            # Compute the classification cost.
            cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_total_targets]

            # Mask2former
            if self.use_point_sample:
                out_mask = out_mask[:, None]  # [num_queries, 1, H_pred, W_pred]
                tgt_mask = tgt_mask[:, None]  # [c, 1, h, w]

                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,  # [c, 1, h, w]
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),  # [c, self.num_points, 2]
                    align_corners=False,
                ).squeeze(1)  # [c, self.num_points]

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)  # [num_queries, self.num_points]
            else:
                tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")
                # Flatten spatial dimension
                out_mask = out_mask.flatten(1)  # [num_queries, H*W]
                tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            with autocast(enabled=False, device_type="cuda"):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()  # [num_queries, num_total_targets]

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Perform Hungarian matching between predictions and targets.

        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes]
                with the classification logits
                "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred]
                with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size),
            where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes
                is the number of ground-truth objects in the target) containing the class labels
                "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        """String representation."""
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
