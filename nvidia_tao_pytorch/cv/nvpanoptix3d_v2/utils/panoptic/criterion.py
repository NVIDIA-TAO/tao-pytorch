# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Modified from Mask2Former (https://github.com/facebookresearch/Mask2Former) for multi-view outputs

"""Mask-transformer set criterion for panoptic segmentation."""

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_pytorch.core.distributed.comm import (
    get_world_size,
    is_dist_avail_and_initialized,
)
from nvidia_tao_pytorch.cv.mask2former.utils.criterion import calculate_uncertainty
from nvidia_tao_pytorch.cv.mask2former.utils.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


def masked_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    num_masks: float,
):
    """Per-view Dice loss with void/ignored sample points at zero weight."""
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(1)
        valid = valid.unsqueeze(1)
    inputs = inputs.sigmoid()
    valid = valid.to(dtype=inputs.dtype)
    inputs = inputs * valid
    targets = targets * valid
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    valid_views = (valid.sum(-1) > 0).to(dtype=loss.dtype)
    per_mask = (loss * valid_views).sum(-1) / valid_views.sum(-1).clamp_min(1.0)
    return per_mask.sum() / num_masks


masked_dice_loss_jit = torch.jit.script(masked_dice_loss)


def masked_sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    num_masks: float,
):
    """Per-view BCE normalised only over valid panoptic sample points."""
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(1)
        valid = valid.unsqueeze(1)
    valid = valid.to(dtype=inputs.dtype)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    valid_counts = valid.sum(-1)
    per_view = (loss * valid).sum(-1) / valid_counts.clamp_min(1.0)
    valid_views = (valid_counts > 0).to(dtype=per_view.dtype)
    per_mask = (
        (per_view * valid_views).sum(-1) /
        valid_views.sum(-1).clamp_min(1.0)
    )
    return per_mask.sum() / num_masks


masked_sigmoid_ce_loss_jit = torch.jit.script(masked_sigmoid_ce_loss)


def sigmoid_focal_loss(
    inputs, targets, num_masks, output_mask, alpha: float = 0.25, gamma: float = 2
):
    """Focal loss for the open-vocabulary classification head.

    ``output_mask`` zeroes the vocabulary columns a batch's source taxonomy
    does not cover, so union classes absent from that root never become
    false negatives.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss * output_mask
    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """DETR-style set prediction loss with Hungarian matching.

    Steps:
      1. Hungarian assignment between GT and predictions
      2. Supervise each matched pair (classification + mask)
    """

    def __init__(
        self,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        label_mode="softmax",
        objectness_no_object_weight=0.1,
        objectness_ignore_overlap_threshold=0.5,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        aux_losses=None,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = list(losses)
        # Rank and objectness are final-query decisions, so callers can limit
        # auxiliary supervision to labels and masks.
        self.aux_losses = (
            list(aux_losses) if aux_losses is not None else list(losses)
        )
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.label_mode = label_mode
        self.objectness_no_object_weight = float(objectness_no_object_weight)
        self.objectness_ignore_overlap_threshold = float(
            objectness_ignore_overlap_threshold
        )
        if not 0.0 <= self.objectness_ignore_overlap_threshold <= 1.0:
            raise ValueError(
                "objectness_ignore_overlap_threshold must be in [0, 1]"
            )

    # classification losses

    def loss_labels_sigmoid(self, outputs, targets, indices, num_masks):
        """Classification loss for the multi-label sigmoid head."""
        if "pred_logits" not in outputs:
            raise ValueError("Classification loss requires pred_logits")
        src_logits = outputs["pred_logits"].float()
        num_classes = src_logits.shape[-1]
        idx = self.get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        output_mask = torch.stack([t["output_mask"] for t in targets]).unsqueeze(1)

        # Vocabulary masking operates per class. Extend it per query so
        # unmatched predictions dominated by ignored pixels receive no
        # all-negative semantic target. Matched queries must always retain
        # their positive and negative class supervision, even if their current
        # predicted mask also overlaps an ignored region.
        class_supervision_mask = output_mask.expand(
            -1, src_logits.shape[1], -1,
        )
        ignored_queries = self.queries_on_ignored_regions(outputs, targets)
        if ignored_queries is not None:
            matched_queries = torch.zeros_like(ignored_queries)
            matched_queries[idx] = True
            ignored_unmatched = ignored_queries & ~matched_queries
            class_supervision_mask = (
                class_supervision_mask & ~ignored_unmatched.unsqueeze(-1)
            )

        target_classes_oh = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
        )
        target_classes_oh.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_oh = target_classes_oh[:, :, :-1]

        # sigmoid_focal_loss sums over eligible ScanNet++ class columns.
        # Normalize by the number of active columns.
        vocab_ref = 100.0
        active_classes = output_mask.sum(dim=-1).float().mean().clamp_min(1.0)
        loss_focal = (
            sigmoid_focal_loss(
                src_logits, target_classes_oh, num_masks,
                class_supervision_mask,
            ) *
            src_logits.shape[1] *
            (vocab_ref / active_classes)
        )
        return {"loss_ce": loss_focal}

    def loss_labels_softmax(self, outputs, targets, indices, num_masks):
        """Classification loss for the single-label softmax head."""
        del num_masks
        if "pred_logits" not in outputs:
            raise ValueError("Classification loss requires pred_logits")
        src_logits = outputs["pred_logits"].float()
        num_classes = src_logits.shape[-1] - 1
        idx = self.get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        output_mask = torch.stack([t["output_mask"] for t in targets]).unsqueeze(1)
        output_mask = F.pad(output_mask, (0, 1), value=True)
        src_logits_masked = torch.where(output_mask, src_logits, float("-inf"))

        empty_weight = torch.ones(num_classes + 1, device=src_logits.device)
        empty_weight[-1] = self.eos_coef

        loss_ce = F.cross_entropy(
            src_logits_masked.transpose(1, 2), target_classes, empty_weight
        )
        return {"loss_ce": loss_ce}

    def loss_labels_rank(self, outputs, targets, indices, num_masks):
        """Cross-entropy over classes for Hungarian-matched queries only.

        Sigmoid focal loss teaches independent class compatibility, while
        inference emits one winning class per query. This complementary loss
        directly trains that top-1 decision and sharpens close semantic pairs
        such as cushion/pillow and cabinet/kitchen-cabinet.
        """
        del num_masks
        src_logits = outputs["pred_logits"].float()
        src_idx = self.get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            return {"loss_rank": src_logits.sum() * 0.0}

        matched_logits = src_logits[src_idx]
        matched_targets = torch.cat([
            target["labels"][tgt_idx]
            for target, (_, tgt_idx) in zip(targets, indices)
        ])
        matched_column_masks = torch.cat([
            target["output_mask"].unsqueeze(0).expand(src_idx_i.numel(), -1)
            for target, (src_idx_i, _) in zip(targets, indices)
        ], dim=0)
        matched_logits = matched_logits.masked_fill(
            ~matched_column_masks.to(matched_logits.device), -1.0e4,
        )
        return {
            "loss_rank": F.cross_entropy(matched_logits, matched_targets)
        }

    def loss_objectness(self, outputs, targets, indices, _num_masks):
        """Binary query supervision, excluding ignored-region detections.

        Matched queries are always positive. An unmatched query whose
        predicted foreground support lies primarily on ignored pixels
        gets zero weight instead of a false no-object target.
        """
        logits = outputs.get("pred_objectness")
        if logits is None:
            raise ValueError(
                "objectness loss requested but model returned no "
                "pred_objectness logits"
            )
        logits = logits.float()
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        if logits.ndim != 2:
            raise ValueError(
                f"pred_objectness must have shape [B,Q], got {logits.shape}"
            )

        target = torch.zeros_like(logits)
        src_idx = self.get_src_permutation_idx(indices)
        target[src_idx] = 1.0
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        neg_weight = self.objectness_no_object_weight
        weights = target + (1.0 - target) * neg_weight

        ignored_queries = self.queries_on_ignored_regions(outputs, targets)
        if ignored_queries is not None:
            weights = weights.masked_fill(
                ignored_queries & (target == 0), 0.0,
            )
        loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
        return {"loss_objectness": loss}

    @torch.no_grad()
    def queries_on_ignored_regions(self, outputs, targets):
        """Return ``[B,Q]`` for masks dominated by ignored pixels."""
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None or not targets:
            return None
        if pred_masks.ndim != 5:
            raise ValueError(
                "pred_masks must have shape [B,views,Q,H,W] for "
                "ignore-aware objectness"
            )

        batch_size, num_views, num_queries, height, width = pred_masks.shape
        if len(targets) != batch_size:
            raise ValueError("targets and pred_masks batch size must match")

        ignore_queries = torch.zeros(
            batch_size, num_queries, dtype=torch.bool,
            device=pred_masks.device,
        )
        for batch_index, target_dict in enumerate(targets):
            ignored = target_dict["ignored_panoptic_mask"]
            if not bool(ignored.any()):
                continue
            ignored = ignored.to(device=pred_masks.device, dtype=torch.float32)
            if ignored.shape[0] != num_views:
                raise ValueError(
                    "ignored_panoptic_mask and pred_masks view counts must match"
                )
            ignored = F.interpolate(
                ignored.unsqueeze(1), size=(height, width), mode="nearest",
            ).squeeze(1).bool()

            predicted_foreground = (
                pred_masks[batch_index].transpose(0, 1).detach() > 0
            )
            support = predicted_foreground.sum(dim=(1, 2, 3))
            overlap = (
                predicted_foreground & ignored.unsqueeze(0)
            ).sum(dim=(1, 2, 3))
            fraction = overlap.float() / support.clamp_min(1).float()
            ignore_queries[batch_index] = (
                (support > 0) &
                (fraction >= self.objectness_ignore_overlap_threshold)
            )
        return ignore_queries

    # mask losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Point-sampled mask and Dice losses for the Hungarian-matched queries."""
        if "pred_masks" not in outputs:
            raise ValueError("Mask loss requires pred_masks")

        src_idx = self.get_src_permutation_idx(indices)
        tgt_idx = self.get_tgt_permutation_idx(indices)

        if src_idx[0].numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero.clone()}

        src_masks = outputs["pred_masks"]
        src_masks = src_masks.transpose(1, 2)[src_idx]

        masks = [t["masks"] for t in targets]
        max_n = max(m.shape[0] for m in masks)
        padded = []
        for m in masks:
            if m.shape[0] < max_n:
                pad = m.new_zeros(max_n - m.shape[0], *m.shape[1:])
                m = torch.cat([m, pad], dim=0)
            padded.append(m)
        target_masks = torch.stack(padded).to(src_masks)
        target_masks = target_masks[tgt_idx]

        valid_masks = torch.stack([
            target["valid_panoptic_mask"] for target in targets
        ]).to(
            device=src_masks.device
        )
        valid_masks = valid_masks[src_idx[0]]

        n, v = src_masks.shape[:2]
        src_masks = src_masks.flatten(0, 1).unsqueeze(1)
        target_masks = target_masks.flatten(0, 1).unsqueeze(1)
        valid_masks = valid_masks.flatten(0, 1).unsqueeze(1)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False
            ).squeeze(1)
            point_valid = point_sample(
                valid_masks.float(), point_coords, mode="nearest",
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks, point_coords, align_corners=False
        ).squeeze(1)

        point_logits = point_logits.unflatten(0, (n, v))
        point_labels = point_labels.unflatten(0, (n, v))
        point_valid = point_valid.unflatten(0, (n, v))

        return {
            "loss_mask": masked_sigmoid_ce_loss_jit(
                point_logits, point_labels, point_valid, num_masks,
            ),
            "loss_dice": masked_dice_loss_jit(
                point_logits, point_labels, point_valid, num_masks,
            ),
        }

    # index helpers

    def get_src_permutation_idx(self, indices):
        """Return batch and source-query indices for matched predictions."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_tgt_permutation_idx(self, indices):
        """Return batch and target indices for matched annotations."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        """Dispatch to the loss function registered under ``loss``."""
        loss_map = {
            "labels": self.loss_labels_softmax if self.label_mode == "softmax" else self.loss_labels_sigmoid,
            "labels_rank": self.loss_labels_rank,
            "objectness": self.loss_objectness,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Unsupported loss: {loss}")
        return loss_map[loss](outputs, targets, indices, num_masks)

    # main forward

    def forward(self, outputs, targets):
        """Match predictions to targets and accumulate every configured loss term."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        final_indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, final_indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.aux_losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, final_indices

    def __repr__(self):
        """Render the configured loss terms and their weights."""
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "aux_losses: {}".format(self.aux_losses),
            "weight_dict: {}".format(self.weight_dict),
            "eos_coef: {}".format(self.eos_coef),
            "objectness_no_object_weight: {}".format(
                self.objectness_no_object_weight
            ),
            "objectness_ignore_overlap_threshold: {}".format(
                self.objectness_ignore_overlap_threshold
            ),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
