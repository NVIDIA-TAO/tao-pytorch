# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Modified from Mask2Former (https://github.com/facebookresearch/Mask2Former) for multi-view outputs

"""Hungarian matching between GT and predicted masks for supervision."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from nvidia_tao_pytorch.cv.mask2former.utils.point_features import point_sample


def batch_dice_loss_masked(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
):
    """Pairwise Dice cost over annotated, non-ignored points only.

    ``valid`` is shared by every predicted/target mask in one sample because
    panoptic annotation validity is a property of the source pixels, not of
    an individual instance.  Returning zero when no sampled point is valid
    leaves assignment to the classification cost instead of treating void as
    background.
    """
    inputs = inputs.sigmoid()
    valid = valid.to(dtype=inputs.dtype).flatten()
    targets = targets * valid.unsqueeze(0)
    inputs = inputs * valid.unsqueeze(0)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss * (valid.sum() > 0).to(dtype=loss.dtype)


batch_dice_loss_masked_jit = torch.jit.script(batch_dice_loss_masked)


def batch_sigmoid_ce_loss_masked(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
):
    """Pairwise BCE cost over annotated, non-ignored points only."""
    valid = valid.to(dtype=inputs.dtype).flatten()
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    ) * valid.unsqueeze(0)
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    ) * valid.unsqueeze(0)
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    return loss / valid.sum().clamp_min(1.0)


batch_sigmoid_ce_loss_masked_jit = torch.jit.script(
    batch_sigmoid_ce_loss_masked
)


class HungarianMatcher(nn.Module):
    """Assignment between targets and predictions via the Hungarian algorithm.

    There are typically more predictions than targets; unmatched predictions
    are treated as no-object.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        label_mode: str = "softmax",
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        self.label_mode = label_mode
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("At least one matching cost must be nonzero")

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """Solve the assignment one batch element at a time to bound peak memory."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]

            if len(tgt_ids) == 0:
                indices.append(([], []))
                continue

            # Classification matching cost. This has to score class agreement
            # with the same objective that is back-propagated, or the bipartite
            # assignment optimises a different target than the criterion.
            # ``sigmoid`` mode uses the Deformable-DETR / MaskDINO focal cost
            # (positive minus negative, fp32 for log stability); ``softmax``
            # mode uses Mask2Former's negative-probability cost.
            cls_logits = outputs["pred_logits"][b].float()
            if self.label_mode == "sigmoid":
                alpha, gamma = 0.25, 2.0
                prob = cls_logits.sigmoid()
                neg_cost_class = (
                    (1 - alpha) * (prob ** gamma) * (-(1 - prob).clamp_min(1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - prob) ** gamma) * (-prob.clamp_min(1e-8).log())
                )
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                out_prob = cls_logits.softmax(-1)
                cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]
            tgt_mask = targets[b]["masks"].to(out_mask)
            valid_mask = targets[b]["valid_panoptic_mask"].to(
                device=out_mask.device,
            )

            out_mask = out_mask.transpose(0, 1)
            if out_mask.shape[1] != tgt_mask.shape[1]:
                raise ValueError("Outputs and targets must have the same number of views")
            if valid_mask.shape != tgt_mask.shape[1:]:
                raise ValueError("valid_panoptic_mask must have shape [views, H, W]")
            n_out, num_views = out_mask.shape[:2]
            n_tgt = tgt_mask.shape[0]

            tgt_mask = tgt_mask.flatten(0, 1).unsqueeze(1)
            out_mask = out_mask.flatten(0, 1).unsqueeze(1)

            point_coords = torch.rand(
                num_views, self.num_points, 2, device=out_mask.device
            )

            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(n_tgt, 1, 1),
                align_corners=False,
            ).squeeze(1)
            tgt_mask = tgt_mask.unflatten(0, (n_tgt, num_views)).flatten(-2)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(n_out, 1, 1),
                align_corners=False,
            ).squeeze(1)
            out_mask = out_mask.unflatten(0, (n_out, num_views)).flatten(-2)

            valid_points = point_sample(
                valid_mask.float().unsqueeze(1),
                point_coords,
                mode="nearest",
                align_corners=False,
            ).squeeze(1).flatten()

            with torch.amp.autocast("cuda", enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                valid_points = valid_points.float()
                cost_mask = batch_sigmoid_ce_loss_masked_jit(
                    out_mask, tgt_mask, valid_points,
                )
                cost_dice = batch_dice_loss_masked_jit(
                    out_mask, tgt_mask, valid_points,
                )

            C = (
                self.cost_mask * cost_mask +
                self.cost_class * cost_class +
                self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            C[~torch.isfinite(C)] = 1e4
            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64, device=outputs["pred_logits"].device),
                torch.as_tensor(j, dtype=torch.int64, device=outputs["pred_logits"].device),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Match predicted queries to targets and return the per-sample index pairs."""
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        """Render the configured matching costs."""
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
