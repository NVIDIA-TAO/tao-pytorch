# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Losses for NVPanoptix3Dv2 panoptic training.

Combined loss:
  L = L_panoptic + λ_metric · L_metric_depth

L_panoptic:
  - Focal/sigmoid cross-entropy for classification
  - Binary CE + Dice for mask prediction
  - Deep supervision across all decoder layers

L_metric_depth (MetricScaleHead supervision):
  - SILog (scale-invariant log) loss between corrected metric depth and GT
  - Optional AbsRel auxiliary term for direct metric accuracy
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.metric_depth import MetricDepthLoss
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.criterion import SetCriterion
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.matcher import HungarianMatcher

# Broad catch-all names are excluded if a ScanNet++ taxonomy contains them.
# Their text embeddings sit near many specific object names, so using them as
# positives would inject conflicting gradients into the language projection.
IGNORE_CLASS_NAMES = frozenset({
    "other structure",
    "other furniture",
    "other prop",
    "trinkets",
})


class NVPanoptix3Dv2PanopticLoss(nn.Module):
    """
    Combined loss for NVPanoptix3Dv2 training.

    L = L_panoptic + λ_metric · L_metric_depth
    """

    def __init__(
        self,
        dec_layers: int = 6,
        deep_supervision: bool = True,
        class_weight: float = 1.0,
        rank_weight: float = 0.0,
        objectness_weight: float = 0.0,
        objectness_no_object_weight: float = 0.1,
        objectness_ignore_overlap_threshold: float = 0.5,
        mask_weight: float = 20.0,
        dice_weight: float = 1.0,
        num_points: int = 2048,
        label_mode: str = "sigmoid",
        metric_depth_weight: float = 0.0,
        metric_silog_weight: float = 1.0,
        metric_absrel_weight: float = 0.1,
        metric_silog_lambda: float = 0.85,
        metric_min_depth: float = 0.1,
        metric_max_depth: float = 10.0,
    ):
        super().__init__()
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
            label_mode=label_mode,
        )

        panoptic_weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        weight_dict = dict(panoptic_weight_dict)
        if rank_weight > 0:
            weight_dict["loss_rank"] = float(rank_weight)
        if objectness_weight > 0:
            weight_dict["loss_objectness"] = float(objectness_weight)
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers):
                # Only the standard class/mask objectives receive deep
                # supervision. Rank and objectness are final-query decisions;
                # supervising them on independently re-matched intermediate
                # layers destabilizes query identity and multiplies their
                # effective weight by the number of decoder predictions.
                aux_weight_dict.update({
                    f"{k}_{i}": v for k, v in panoptic_weight_dict.items()
                })
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        aux_losses = ["labels", "masks"]
        if rank_weight > 0:
            losses.append("labels_rank")
        if objectness_weight > 0:
            losses.append("objectness")

        self.criterion = SetCriterion(
            matcher=matcher,
            weight_dict=weight_dict,
            # NVPanoptix3Dv2 uses sigmoid labels, where eos_coef is inactive.
            eos_coef=0.1,
            losses=losses,
            aux_losses=aux_losses,
            num_points=num_points,
            label_mode=label_mode,
            objectness_no_object_weight=objectness_no_object_weight,
            objectness_ignore_overlap_threshold=(
                objectness_ignore_overlap_threshold
            ),
        )

        self.metric_depth_weight = metric_depth_weight
        self.metric_depth_loss = MetricDepthLoss(
            silog_weight=metric_silog_weight,
            absrel_weight=metric_absrel_weight,
            silog_lambda=metric_silog_lambda,
            min_depth=metric_min_depth,
            max_depth=metric_max_depth,
        ) if metric_depth_weight > 0 else None

    def prepare_targets(
        self,
        gts: List[Dict],
        classes: List[str],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert collated multi-view GT into SetCriterion format.

        Args:
            gts: list of S view dicts, each with batched tensors:
                 ``pan_inst_id`` (B, H, W) and ``pan_cls_id`` (B, H, W).
            classes: class name list of length C.
            device: target device.

        Returns:
            list of B target dicts, each containing:
              - ``labels``:      (N_inst,)       int64 class indices
              - ``masks``:       (N_inst, S, H, W) float32 binary masks
              - ``output_mask``: (C,) bool — enabled ScanNet++ classes
              - ``valid_panoptic_mask``: (S, H, W) bool — pixels eligible
                for Hungarian mask costs and BCE/Dice supervision
              - ``ignored_panoptic_mask``: (S, H, W) bool — void or disabled
                pixels that must not create negative objectness targets
        """
        num_classes = len(classes)
        S = len(gts)
        B = gts[0]["pan_inst_id"].shape[0]
        H, W = gts[0]["pan_inst_id"].shape[1:]
        output_mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        # Exclude broad catch-all classes from the native ScanNet++ taxonomy.
        ignore_idx = {
            i for i, name in enumerate(classes) if name in IGNORE_CLASS_NAMES
        }
        for i in ignore_idx:
            output_mask[i] = False

        targets = []
        for b in range(B):
            inst_maps = torch.stack([
                gts[v]["pan_inst_id"][b].to(device=device)
                for v in range(S)
            ])
            cls_maps = torch.stack([
                gts[v]["pan_cls_id"][b].to(device=device)
                for v in range(S)
            ])

            annotated = inst_maps != 0
            class_in_range = (cls_maps >= 0) & (cls_maps < num_classes)
            class_enabled = torch.zeros_like(class_in_range)
            if class_in_range.any():
                class_enabled[class_in_range] = output_mask[
                    cls_maps[class_in_range].long()
                ]

            valid_panoptic_mask = annotated & class_enabled
            # ScanNet++ panoptic id 0 is the void/unlabelled sentinel. Treat
            # the full complement as ignored rather than semantic background.
            ignored_panoptic_mask = ~valid_panoptic_mask

            all_uids = torch.unique(inst_maps[valid_panoptic_mask]).sort().values

            labels = []
            masks = []
            valid_uids = []
            for uid_tensor in all_uids:
                uid = int(uid_tensor.item())
                mv_mask_bool = (
                    (inst_maps == uid_tensor) & valid_panoptic_mask
                )
                cls_idx = int(cls_maps[mv_mask_bool][0].item())
                mv_mask = mv_mask_bool.to(dtype=torch.float32)

                labels.append(cls_idx)
                masks.append(mv_mask)
                valid_uids.append(uid)

            if not labels:
                targets.append({
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                    "masks": torch.zeros(0, S, H, W, device=device),
                    "output_mask": output_mask,
                    "valid_panoptic_mask": valid_panoptic_mask,
                    "ignored_panoptic_mask": ignored_panoptic_mask,
                    "instance_uids": [],
                })
            else:
                targets.append({
                    "labels": torch.tensor(labels, dtype=torch.long, device=device),
                    "masks": torch.stack(masks).to(device),
                    "output_mask": output_mask,
                    "valid_panoptic_mask": valid_panoptic_mask,
                    "ignored_panoptic_mask": ignored_panoptic_mask,
                    "instance_uids": valid_uids,
                })

        return targets

    def forward(
        self,
        gts: List[Dict],
        panoptic_output: Dict[str, torch.Tensor],
        classes: List[str],
        geometry_output: Optional[Dict[str, torch.Tensor]] = None,
        gt_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined panoptic and optional metric-depth loss.

        Args:
            gts:              list of GT dicts per batch element
            panoptic_output:  dict with pred_logits, pred_masks, aux_outputs
            classes:          list of class names
            geometry_output:  optional dict with depth, metric_depth, metric_points, etc.
            gt_depth:         optional [B, S, H, W] metric GT depth in metres

        Returns:
            (total_loss, loss_details)
        """
        device = panoptic_output["pred_logits"].device
        targets = self.prepare_targets(gts, classes, device)

        # Panoptic loss (classification + mask)
        losses, _ = self.criterion(panoptic_output, targets)

        total_loss = sum(
            losses[k] * self.criterion.weight_dict.get(k, 1.0)
            for k in losses
            if k in self.criterion.weight_dict
        )

        loss_details = {"panoptic_loss": total_loss.detach()}
        loss_details.update({k: v.detach() for k, v in losses.items()})

        # Metric depth loss (MetricScaleHead supervision)
        if (
            self.metric_depth_loss is not None and
            geometry_output is not None and
            gt_depth is not None
        ):
            metric_depth = geometry_output.get("metric_depth")
            if metric_depth is not None:
                metric_losses = self.metric_depth_loss(
                    metric_depth, gt_depth,
                )
                total_loss = (
                    total_loss +
                    self.metric_depth_weight * metric_losses["loss_metric_total"]
                )
                loss_details.update({k: v.detach() for k, v in metric_losses.items()})

        return total_loss, loss_details
