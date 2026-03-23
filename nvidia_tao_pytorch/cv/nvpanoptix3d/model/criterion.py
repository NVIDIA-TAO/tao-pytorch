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

"""Criterion functions for NVPanoptix3D model."""

import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.mask2former.utils.misc import nested_tensor_from_tensor_list
from nvidia_tao_pytorch.core.distributed.comm import (
    get_world_size,
    is_dist_avail_and_initialized,
)
from nvidia_tao_pytorch.cv.mask2former.utils.point_features import (
    point_sample, get_uncertain_point_coords_with_randomness
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: Logits tensor of shape (N, ...) where N is the number of masks/elements.
        targets: Binary tensor with the same shape as inputs (0/1).
        num_masks: Normalization factor, typically the number of target masks (possibly
            averaged across distributed workers).

    Returns:
        A scalar tensor representing the normalized Dice loss.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """Sigmoid focal loss (RetinaNet) for binary targets.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        inputs: Logits tensor of shape (N, ...) where N is the number of elements.
        targets: Binary tensor with the same shape as inputs (0/1).
        num_masks: Normalization factor.
        alpha: Class balancing weight in (0, 1). If set < 0, alpha weighting is disabled.
        gamma: Focusing parameter controlling down-weighting of easy examples.

    Returns:
        A scalar tensor representing the normalized focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """Binary cross-entropy (with logits) loss normalized by num_masks.

    Args:
        inputs: Logits tensor of shape (N, ...) where N is the number of elements.
        targets: Binary tensor with the same shape as inputs (0/1).
        num_masks: Normalization factor.

    Returns:
        A scalar tensor representing the normalized BCE loss.
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)


def depth_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    room_mask: torch.Tensor = None,
):
    """Compute L1 depth regression loss on valid pixels.

    Valid pixels are those with targets > 0, optionally intersected with room_mask.
    If no valid pixels exist, returns a zero scalar tensor on the correct device/dtype.

    Args:
        inputs: Predicted depth tensor, shape (B, H, W) or (B, 1, H, W).
        targets: Ground-truth depth tensor, same shape as inputs.
        room_mask: Optional boolean/binary mask, shape (B, H, W) or (B, 1, H, W).

    Returns:
        A scalar tensor representing mean absolute error on valid pixels.
    """
    # Accept both (B, H, W) and (B, 1, H, W) shapes.
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if room_mask is not None and room_mask.ndim == 3:
        room_mask = room_mask.unsqueeze(1)

    if inputs.shape != targets.shape:
        raise ValueError(f"depth_loss expects same shapes, got {inputs.shape} vs {targets.shape}")

    valid = targets > 0
    if room_mask is not None:
        valid = valid & room_mask.bool()

    if valid.any():
        loss_reg = (targets[valid] - inputs[valid]).abs().mean()
    else:
        # No valid pixels: return a zero loss that won't break backward.
        loss_reg = inputs.new_zeros(())
    return loss_reg


def calculate_uncertainty(logits):
    """Compute an uncertainty score for point sampling from mask logits.

    Args:
        logits: Tensor of shape (R, 1, ...) where R is the number of masks. Must have
            channel dimension == 1.

    Returns:
        Tensor of shape (R, 1, ...) where larger values mean higher uncertainty.
    """
    assert logits.shape[1] == 1, "calculate_uncertainty expects logits with channel dim == 1."
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """Loss container for NVPanoptix3D training.

    The criterion performs Hungarian matching between predictions and ground-truth
    instances, then computes a configurable set of losses (classification, 2D
    mask BCE/Dice, depth, 3D geometry, 3D occupancy, and 3D panoptic masks).

    Weighting behavior:
    - Each target dict may provide loss_weight to down-weight specific samples.
    - Reductions are performed as weighted means either per image or per matched
      instance, depending on the loss.

    This class expects model outputs to include the keys required by the enabled
    losses and targets to provide the corresponding supervision tensors.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 use_point_sample=False):
        """Initialize criterion.

        Args:
            num_classes: Number of semantic classes (excluding the "no-object" class).
            matcher: Module that computes matching between predictions and GT per image.
            weight_dict: Mapping from loss name to its scalar weight (applied outside here).
            eos_coef: Relative classification weight applied to the "no-object" class.
            losses: List of loss names to compute (keys for get_loss).
            num_points: Number of points for point sampling in mask loss (if enabled).
            oversample_ratio: Oversampling ratio for point sampling.
            importance_sample_ratio: Fraction of points sampled by uncertainty.
            use_point_sample: If True, compute mask losses on sampled points (faster).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.use_point_sample = use_point_sample
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.loss_occ_2d = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _to_float(v) -> float:
        """Robust float conversion for config/sample weights."""
        if torch.is_tensor(v):
            return float(v.item())
        return float(v)

    def _get_sample_weights(self, targets, device: torch.device) -> torch.Tensor:
        """Extract per-image loss weights.

        Args:
            targets: List of per-image target dicts (len == batch size).
            device: Device to place the returned tensor on.

        Returns:
            Float tensor of shape (B,) containing per-image weights. Defaults to 1.0 if
            loss_weight is missing.
        """
        weights = [self._to_float(t.get("loss_weight", 1.0)) for t in targets]
        return torch.as_tensor(weights, device=device, dtype=torch.float32)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted mean over the leading dimension.

        Args:
            values: Tensor of shape (N, ...) that is already reduced to a per-item scalar
                along non-leading dimensions (i.e., shape (N,) recommended).
            weights: Tensor broadcastable to values along the leading dimension.

        Returns:
            A scalar tensor: sum(values * weights) / sum(weights).
        """
        weights = weights.to(device=values.device, dtype=values.dtype)
        denom = weights.sum().clamp_min(1e-12)
        return (values * weights).sum() / denom

    def loss_mp_occ(self, pred, target, sample_weights=None):
        """Multi-plane occupancy loss (2D stage).

        Computes BCEWithLogits per pixel, masked by valid depth pixels. Optionally reduces the
        final scalar with a per-image weighted mean.

        Args:
            pred: Predicted multi-plane occupancy, shape (B, S, H, W).
            target: Dict with:
                - occupancy: shape (B, S, H, W)
                - depth_map: shape (B, H, W) (non-zero indicates valid)
            sample_weights: Optional per-image weights, shape (B,).

        Returns:
            A scalar tensor representing the occupancy loss.
        """
        eps = 1e-7
        loss_occ = self.loss_occ_2d(pred, target["occupancy"])

        valid_masks = (target["depth_map"] != 0.0).bool().to(pred.device)

        # Resize valid_masks to match loss_occ spatial dimensions if needed
        if valid_masks.shape[-2:] != loss_occ.shape[-2:]:
            valid_masks = F.interpolate(
                valid_masks.float(),
                size=loss_occ.shape[-2:],
                mode="nearest",
            ).bool()

        loss_occ = loss_occ * valid_masks

        # Reduce per-image, then (optionally) apply per-image weights.
        if sample_weights is None:
            mp_occ_loss = loss_occ.sum() / (valid_masks.sum() + eps)
        else:
            if valid_masks.ndim == 3:
                valid_masks = valid_masks.unsqueeze(1)
            per_image_num = loss_occ.sum(dim=(1, 2, 3))
            per_image_den = valid_masks.sum(dim=(1, 2, 3)).to(loss_occ).clamp_min(eps)
            per_image = per_image_num / per_image_den
            mp_occ_loss = self._weighted_mean(per_image, sample_weights)

        return mp_occ_loss

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss for matched queries.

        Uses cross-entropy over num_classes + 1 (including "no-object"). Reduced to a per-image
        loss and then optionally weighted by targets[*]["loss_weight"].

        Args:
            outputs: Model outputs dict containing pred_logits of shape (B, Q, C+1).
            targets: List of target dicts with labels.
            indices: Matcher output mapping predictions↔targets per image.
            num_masks: Unused here (kept for API symmetry with other losses).

        Returns:
            Dict with loss_ce scalar.
        """
        assert "pred_logits" in outputs, "loss_labels expects outputs to include pred_logits."
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # Compute per-image CE and apply per-image weighting (e.g. down-weight aug samples).
        loss_ce_per = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="none",
        )  # (B, Q)
        loss_ce_per_image = loss_ce_per.mean(dim=1)  # (B,)
        sample_weights = self._get_sample_weights(targets, src_logits.device)
        loss_ce = self._weighted_mean(loss_ce_per_image, sample_weights)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_depths(self, outputs, targets, indices, num_masks, room_mask=None):
        """Depth regression loss (per-image, optionally weighted by augmentation).

        This loss supervises outputs["pred_depths"] against targets[*]["depths"] using the
        top-level depth_loss() helper (mean absolute error on valid pixels only).

        Args:
            outputs: Dict containing pred_depths with shape (B, H, W) or (B, 1, H, W).
            targets: List of dicts with key depths. Each depths is expected to be a tensor
                of shape (1, Ht, Wt) (or compatible) with invalid pixels set to 0.
            indices: Matcher output (unused for depth supervision).
            num_masks: Normalization factor (unused; depth uses weighted mean reduction).
            room_mask: Optional boolean/binary tensor of shape (B, Hr, Wr) or (B, 1, Hr, Wr).

        Returns:
            Dict with a single scalar entry: {"loss_depth": <tensor>}.
        """
        assert "pred_depths" in outputs, "loss_depths expects outputs to include pred_depths."
        src_depths = outputs["pred_depths"]
        # Standardize prediction to (B, 1, H, W)
        if src_depths.ndim == 3:
            src_depths = src_depths.unsqueeze(1)

        depths = []
        for t in targets:
            d = t["depths"][0:1, :, :].unsqueeze(0).to(src_depths)  # (1, 1, H, W)
            if d.shape[-2:] != src_depths.shape[-2:]:
                d = F.interpolate(
                    d,
                    size=src_depths.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            depths.append(d)
        target_depths = torch.cat(depths, dim=0)

        # convert room_mask to float
        if room_mask is not None:
            room_mask = room_mask.float()

            room_mask = F.interpolate(
                room_mask,
                size=src_depths.shape[-2:],
                mode="nearest-exact",
            )
            # convert room_mask to back to bool
            room_mask = room_mask.bool()

        losses = {
            # Compute per-image depth loss then apply per-image weights (e.g. down-weight aug samples).
            "loss_depth": self._weighted_mean(
                torch.stack([
                    depth_loss(
                        src_depths[b:b + 1],
                        target_depths[b:b + 1],
                        None if room_mask is None else room_mask[b:b + 1],
                    )
                    for b in range(src_depths.shape[0])
                ]),
                self._get_sample_weights(targets, src_depths.device),
            ),
        }

        del src_depths
        del target_depths
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask losses (BCE + Dice) for matched masks.

        Computes losses either on sampled points (if enabled) or on full flattened masks.
        Per-mask losses are reduced using the corresponding source image loss_weight.

        Args:
            outputs: Dict containing pred_masks (B, Q, H, W) or equivalent.
            targets: List of dicts each containing masks (Ni, H, W) and optional loss_weight.
            indices: Matcher output mapping predictions↔targets per image.
            num_masks: Unused here (kept for API symmetry; normalization is handled via weighted mean).

        Returns:
            Dict with loss_mask and loss_dice scalars.
        """
        assert "pred_masks" in outputs, "loss_masks expects outputs to include pred_masks."

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        if self.use_point_sample:
            src_masks = src_masks[:, None]
            target_masks = target_masks[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
        else:
            point_logits = src_masks.flatten(1)
            point_labels = target_masks.flatten(1)

        sample_weights = self._get_sample_weights(targets, point_logits.device)
        # One weight per matched mask, derived from its source image index.
        mask_weights = sample_weights[src_idx[0]]

        # Weighted sigmoid CE loss over matched masks.
        ce_per_mask = F.binary_cross_entropy_with_logits(
            point_logits, point_labels, reduction="none"
        ).mean(dim=1)
        loss_mask = self._weighted_mean(ce_per_mask, mask_weights)

        # Weighted dice loss over matched masks.
        inputs = point_logits.sigmoid().flatten(1)
        targets_f = point_labels.flatten(1)
        numerator = 2 * (inputs * targets_f).sum(-1)
        denominator = inputs.sum(-1) + targets_f.sum(-1)
        dice_per_mask = 1 - (numerator + 1) / (denominator + 1)
        loss_dice = self._weighted_mean(dice_per_mask, mask_weights)

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice
        }

        del src_masks
        del target_masks
        return losses

    def loss_geometry(self, outputs, targets, indices, num_masks):
        """3D geometry (TSDF) regression loss at sparse query points.

        Args:
            outputs: Dict containing pred_geometry (a sparse tensor-like object) with:
                - batch_indexed_coordinates: long tensor (N, 4) where the first column is batch
                  index and remaining columns are spatial indices.
                - feature_tensor: tensor (N, 1) containing predicted TSDF values.
            targets: List of dicts with:
                - geometry: tensor (B, D, H, W) (or compatible) containing GT TSDF.
                - weighting3d_256: tensor with same shape as geometry containing per-voxel weights.
                - optional loss_weight: float.
            indices: Matcher output (unused for geometry supervision).
            num_masks: Normalization factor (unused; geometry uses weighted mean reduction).

        Returns:
            Dict with a single scalar entry: {"loss_geometry": <tensor>}.
        """
        assert "pred_geometry" in outputs, "loss_geometry expects outputs to include pred_geometry."
        pred = outputs["pred_geometry"]
        coords = pred.batch_indexed_coordinates.long()

        target_geometry = torch.stack([t["geometry"] for t in targets])
        target_weighting = torch.stack([t["weighting3d_256"] for t in targets])
        target_geometry = target_geometry[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
        target_weighting = target_weighting[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]

        loss = F.l1_loss(pred.feature_tensor.squeeze(-1), target_geometry, reduction="none")
        loss = (loss * target_weighting)
        sample_weights = self._get_sample_weights(targets, loss.device)
        point_weights = sample_weights[coords[:, 0]]
        loss = self._weighted_mean(loss, point_weights)
        return {
            "loss_geometry": loss,
        }

    def loss_occupancy(self, outputs, targets, indices, num_masks):
        """3D occupancy loss at multiple pyramid resolutions (64/128/256).

        This computes binary occupancy supervision for 3D grids at multiple resolutions.

        Supervision inputs:
        - Targets provide dense occupancy grids: occupancy_64, occupancy_128, occupancy_256
          and corresponding per-voxel weights: weighting3d_64/128/256.
        - Predictions provide:
          - a dense tensor for level 64 (where we can compute full-grid BCE),
          - sparse tensors for levels 128/256 (where we compute BCE only at predicted coordinates).

        Args:
            outputs: Dict containing pred_occupancies (list/tuple of levels).
            targets: List of dicts containing occupancy and weighting tensors for each level,
                plus optional loss_weight.
            indices: Matcher output (unused for occupancy supervision).
            num_masks: Normalization factor (unused; occupancy uses weighted mean reduction).

        Returns:
            Dict with keys loss_occupancy_64, loss_occupancy_128, loss_occupancy_256.
        """
        assert "pred_occupancies" in outputs, "loss_occupancy expects outputs to include pred_occupancies."
        preds = outputs["pred_occupancies"]
        sample_weights = self._get_sample_weights(targets, preds[0].device)

        target_occupancies = [
            torch.stack([t["occupancy_64"] for t in targets]),
            torch.stack([t["occupancy_128"] for t in targets]),
            torch.stack([t["occupancy_256"] for t in targets]),
        ]

        target_weightings = [
            torch.stack([t["weighting3d_64"] for t in targets]),
            torch.stack([t["weighting3d_128"] for t in targets]),
            torch.stack([t["weighting3d_256"] for t in targets])
        ]

        # calculate occupancy loss for lvl 64, dense grid
        pred0 = preds[0]
        pred0 = pred0.to(target_occupancies[0].device)
        is_valid = torch.isfinite(pred0)
        loss = F.binary_cross_entropy_with_logits(pred0, target_occupancies[0], reduction="none")
        loss.masked_fill_(~is_valid, 0)
        loss = loss * target_weightings[0]
        # Reduce per-image, then apply per-image weights.
        per_image_num = loss.view(loss.shape[0], -1).sum(dim=1)
        per_image_den = is_valid.view(is_valid.shape[0], -1).sum(dim=1).to(loss).clamp_min(1)
        loss = self._weighted_mean(per_image_num / per_image_den, sample_weights)

        level_losses = [loss]

        for lvl in range(1, len(preds)):
            pred = preds[lvl]
            coords = pred.batch_indexed_coordinates.long()
            coords = coords.clone()
            # do not scale coordinates by stride since warpconvnet already on stride coordinates
            coords_spatial = coords[:, 1:]
            batch_index = coords[:, 0]
            target_occupancy = target_occupancies[lvl][
                batch_index,
                coords_spatial[:, 0], coords_spatial[:, 1], coords_spatial[:, 2],
            ]
            target_weighting = target_weightings[lvl][
                batch_index,
                coords_spatial[:, 0], coords_spatial[:, 1], coords_spatial[:, 2],
            ]
            logits = pred.feature_tensor.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, target_occupancy, reduction="none",
            )
            loss = loss * target_weighting
            point_weights = sample_weights[batch_index]
            loss = self._weighted_mean(loss, point_weights)
            level_losses.append(loss)

        losses = {f"loss_occupancy_{lvl}": loss for loss, lvl in zip(level_losses, [64, 128, 256])}
        return losses

    def loss_panoptic(self, outputs, targets, indices, num_masks):
        """3D panoptic mask loss at multiple resolutions (64/128/256).

        This supervises predicted 3D instance masks (outputs["pred_segms"]) against GT 3D masks
        (targets[*]["masks_3d_*"]) for the matched instances given by Hungarian matching.

        Args:
            outputs: Dict containing pred_segms (list of pyramid levels).
            targets: List of per-image target dicts with keys described above.
            indices: List of matcher pairs (src_idx, tgt_idx) per image selecting matched instances.
            num_masks: Normalization factor (typically weighted count of GT instances).

        Returns:
            Dict with keys loss_panoptic_64, loss_panoptic_128, loss_panoptic_256.
        """
        assert "pred_segms" in outputs, "loss_panoptic expects outputs to include pred_segms."
        preds = outputs["pred_segms"]

        level_losses = []

        dtype = preds[0].dtype
        sample_weights = self._get_sample_weights(targets, preds[0].device)

        for b in range(len(indices)):
            # lvl 64, dense grid
            src_idx, tgt_idx = indices[b][0], indices[b][1]
            w = sample_weights[b]

            target_weightings = [
                targets[b]["weighting3d_64"],
                targets[b]["weighting3d_128"],
                targets[b]["weighting3d_256"],
            ]

            target_masks = [
                targets[b]["masks_3d_64"][tgt_idx].to(dtype),
                targets[b]["masks_3d_128"][tgt_idx].to(dtype),
                targets[b]["masks_3d_256"][tgt_idx].to(dtype),
            ]

            pred = preds[0][b][src_idx]
            is_valid = torch.isfinite(pred)
            loss = F.binary_cross_entropy_with_logits(pred, target_masks[0], reduction="none")
            loss.masked_fill_(~is_valid, 0)
            loss = loss * target_weightings[0]
            loss = loss.sum() / is_valid.sum().clip(min=1) * len(src_idx)
            if len(level_losses) == 0:
                level_losses.append(loss * w)
            else:
                level_losses[0] += loss * w

            # lvl 128, 256, sparse grid:
            for lvl in range(len(preds)):
                if lvl == 0:
                    continue
                pred_lvl = preds[lvl]
                batch_mask = pred_lvl.batch_indexed_coordinates[:, 0] == b
                features = pred_lvl.feature_tensor[batch_mask]
                coords = pred_lvl.coordinate_tensor[batch_mask].long()
                coords = coords.clone()
                if coords.numel() == 0 or features.numel() == 0:
                    # Empty sparse predictions must remain finite so training
                    # can continue and matched GT masks still contribute loss.
                    loss = target_masks[lvl].flatten(1).any(dim=1).to(dtype).sum()
                else:
                    pred_vals = features.T[src_idx]
                    target_mask = target_masks[lvl][:, coords[:, 0], coords[:, 1], coords[:, 2]]
                    target_weighting = target_weightings[lvl][coords[:, 0], coords[:, 1], coords[:, 2]]
                    loss = F.binary_cross_entropy_with_logits(pred_vals, target_mask, reduction="none")
                    loss = (loss * target_weighting).mean(-1).sum()

                if len(level_losses) <= lvl:
                    level_losses.append(loss * w)
                else:
                    level_losses[lvl] += loss * w

        losses = {
            f"loss_panoptic_{lvl}": loss / num_masks for loss, lvl in zip(level_losses, [64, 128, 256])
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        """Build indices to gather matched predictions from batched outputs.

        Args:
            indices: List of (src_idx, tgt_idx) tuples from the matcher.

        Returns:
            A tuple (batch_idx, src_idx) such that outputs[..., batch_idx, src_idx] (or
            tensor[(batch_idx, src_idx)] for (B, Q, ...) tensors) selects the matched queries
            across the whole batch.
        """
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Build indices to gather matched targets from batched target tensors.

        Args:
            indices: List of (src_idx, tgt_idx) tuples from the matcher.

        Returns:
            A tuple (batch_idx, tgt_idx) to select matched target entries across the batch.
        """
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_binary_mask(self, target):
        """Convert a per-pixel class map into a one-hot tensor.

        Args:
            target: Integer tensor of shape (H, W) containing class ids per pixel.

        Returns:
            One-hot tensor of shape (num_classes + 1, H, W) on CUDA.
        """
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x).cuda()
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def get_loss(self, loss, outputs, targets, indices, num_masks, room_mask=None):
        """Compute a single loss by name.

        Args:
            loss: Loss name (must be a key in the internal loss_map).
            outputs: Model outputs dict for the current decoder layer.
            targets: List of per-image target dicts.
            indices: Matcher indices for this layer.
            num_masks: Normalization factor (shared across losses).
            room_mask: Optional room mask for depth loss.

        Returns:
            Dict mapping a loss key to its scalar tensor value.
        """
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "depths": self.loss_depths,
            "geometry": self.loss_geometry,
            "occupancy": self.loss_occupancy,
            "panoptic": self.loss_panoptic,
        }
        assert loss in loss_map, f"This loss is not supported: {loss}"
        if loss == "depths":
            return loss_map[loss](outputs, targets, indices, num_masks, room_mask)
        else:
            return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, occupancy_preds=None, occupancy_targets=None, room_mask=None):
        """Compute and return all requested losses.

        Args:
            outputs: Model outputs dict. May include aux_outputs for intermediate decoder layers.
            targets: List of per-image dicts (len == batch size). Expected keys depend on configured
                losses (see the corresponding loss_* docstrings). May include optional loss_weight.
            occupancy_preds: Optional multi-plane occupancy prediction (B, S, H, W).
            occupancy_targets: Optional dict of occupancy targets (see loss_mp_occ).
            room_mask: Optional room mask for depth loss.

        Returns:
            Dict mapping loss name to scalar tensors.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Per-image loss weights (e.g. to down-weight augmented samples).
        device = next(iter(outputs.values())).device
        sample_weights = self._get_sample_weights(targets, device)

        # compute the (weighted) average number of target boxes accross all nodes, for normalization purposes
        per_image_num = torch.as_tensor([len(t["labels"]) for t in targets], device=device, dtype=torch.float32)
        num_masks = (per_image_num * sample_weights).sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "depths":
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, room_mask))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # compute the occupancy loss
        if occupancy_preds is not None:
            losses["loss_mp_occ"] = self.loss_mp_occ(occupancy_preds, occupancy_targets, sample_weights)

        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # skip depth loss for auxiliary outputs
                    if loss == "depths":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_targets(self, gt_masks):
        """Create a minimal targets list from GT semantic maps.

        Args:
            gt_masks: Iterable of integer tensors, each of shape (H, W), with class ids per pixel.

        Returns:
            List of dicts with:
            - labels: 1D tensor of class ids present in the image (excluding background/0).
            - masks: Tensor of shape (N, H, W) with one binary mask per label.
        """
        targets = []
        for mask in gt_masks:
            binary_masks = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[1:]
            binary_masks = binary_masks[labels]
            targets.append({"masks": binary_masks, "labels": labels})
        return targets

    def __repr__(self):
        """String representation."""
        head = "Criterion " + self.__class__.__name__
        body = [
            f"matcher: {self.matcher.__repr__(_repr_indent=8)}",
            f"losses: {self.losses}",
            f"weight_dict: {self.weight_dict}",
            f"num_classes: {self.num_classes}",
            f"eos_coef: {self.eos_coef}",
            f"num_points: {self.num_points}",
            f"oversample_ratio: {self.oversample_ratio}",
            f"importance_sample_ratio: {self.importance_sample_ratio}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
