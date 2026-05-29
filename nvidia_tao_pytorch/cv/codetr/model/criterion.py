# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CoDETR loss criterion — DINO Hungarian matching + ATSS collaborative losses."""

import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.dino.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.codetr.model.collaborative_head import ATSSMatcher, ATSSCollaborativeHead
from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops
from nvidia_tao_pytorch.core.distributed.comm import get_world_size, is_dist_avail_and_initialized


def _sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for multi-class classification."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.sum() / num_boxes


def _delta_xywh_decode(anchors, deltas, stds=(0.1, 0.1, 0.2, 0.2)):
    """Decode DeltaXYWH predictions to xyxy boxes.

    Matches mmdet DeltaXYWHBBoxCoder with the given target_stds.

    Args:
        anchors (Tensor): [N, 4] anchor boxes in xyxy format.
        deltas (Tensor): [N, 4] predicted (dx, dy, dw, dh) deltas.
        stds (tuple): target_stds for denormalization.

    Returns:
        decoded (Tensor): [N, 4] predicted boxes in xyxy format.
    """
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0] * stds[0]
    dy = deltas[:, 1] * stds[1]
    dw = deltas[:, 2] * stds[2]
    dh = deltas[:, 3] * stds[3]

    # Clamp to prevent overflow in exp
    dw = dw.clamp(max=4.0)
    dh = dh.clamp(max=4.0)

    cx = ax + dx * aw
    cy = ay + dy * ah
    w = aw * torch.exp(dw)
    h = ah * torch.exp(dh)

    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


class CoDETRCriterion(SetCriterion):
    """CoDETR loss criterion.

    Extends DINO's SetCriterion with ATSS auxiliary head losses.
    Collaborative losses are returned unweighted and scaled by the
    Lightning module's weight_dict.

    Args:
        num_classes (int): number of foreground classes.
        matcher (HungarianMatcher): matcher used by the DETR head.
        focal_alpha (float): alpha value for focal classification loss.
        losses (list[str]): DETR loss names to compute.
        strides (list[int]): feature-map strides used by the ATSS collaborative
            head anchor generator.
        co_head_loss_weight (float): accepted for API compatibility. The
            scalar is applied by the Lightning module's weight_dict.
        atss_topk (int): number of ATSS candidate anchors per ground-truth box
            and feature level.
    """

    def __init__(self, num_classes, matcher, focal_alpha, losses,
                 strides, co_head_loss_weight=1.0, atss_topk=9):
        """Initialize CoDETRCriterion.

        Args:
            num_classes (int): number of foreground classes.
            matcher: HungarianMatcher for the DETR head.
            focal_alpha (float): focal loss alpha.
            losses (list[str]): DETR losses to apply.
            strides (list[int]): feature map strides for ATSS head.
            co_head_loss_weight (float): accepted for API compatibility. The
                scalar is applied by the Lightning module's weight_dict.
            atss_topk (int): ATSS top-k candidates per level per GT.
        """
        super().__init__(num_classes=num_classes, matcher=matcher,
                         focal_alpha=focal_alpha, losses=losses)
        self.strides = strides
        self.co_head_loss_weight = co_head_loss_weight
        self.atss_matcher = ATSSMatcher(topk=atss_topk)

    def compute_collab_loss(self, collab_output, targets, img_hw):
        """Compute ATSS loss for one collaborative head.

        Args:
            collab_output (tuple): (cls_scores, bbox_preds, centernesses)
                each a list of Tensors, one per feature level.
            targets (list[dict]): per-image GT dicts with keys 'boxes' (cxcywh
                normalized) and 'labels'.
            img_hw (tuple[int,int]): (H, W) of the input image.

        Returns:
            losses (dict): keys 'collab_loss_cls', 'collab_loss_bbox', 'collab_loss_centerness'.
        """
        cls_scores, bbox_preds, centernesses = collab_output
        H, W = img_hw

        # Generate anchors for this batch (same for all images)
        feature_maps_fake = cls_scores   # shapes [B, *, H_i, W_i]
        anchors_per_level = ATSSCollaborativeHead.generate_anchors(feature_maps_fake, self.strides)

        all_anchors = torch.cat(anchors_per_level, dim=0)   # [N_total, 4]

        # Flatten per-level predictions across spatial dims
        # cls_flat: [B, N_total, num_classes]
        # reg_flat: [B, N_total, 4]
        # cen_flat: [B, N_total]
        device = cls_scores[0].device
        batch_cls, batch_reg, batch_cen = [], [], []
        for cls_s, reg_s, cen_s in zip(cls_scores, bbox_preds, centernesses):
            B, C = cls_s.shape[:2]
            batch_cls.append(cls_s.permute(0, 2, 3, 1).reshape(B, -1, C))
            batch_reg.append(reg_s.permute(0, 2, 3, 1).reshape(B, -1, 4))
            batch_cen.append(cen_s.permute(0, 2, 3, 1).reshape(B, -1))
        cls_flat = torch.cat(batch_cls, dim=1)   # [B, N_total, num_classes]
        reg_flat = torch.cat(batch_reg, dim=1)   # [B, N_total, 4]
        cen_flat = torch.cat(batch_cen, dim=1)   # [B, N_total]

        all_cls_targets, all_reg_targets, all_cen_targets = [], [], []
        all_pos_masks = []
        num_pos = 0

        for tgt in targets:
            # Convert normalized cxcywh → absolute xyxy
            boxes_n = tgt['boxes']   # [G, 4] normalized cxcywh
            labels = tgt['labels']   # [G]

            if boxes_n.numel() == 0:
                n_total = all_anchors.shape[0]
                all_cls_targets.append(torch.zeros(n_total, self.num_classes, device=device))
                all_reg_targets.append(torch.zeros(n_total, 4, device=device))
                all_cen_targets.append(torch.zeros(n_total, device=device))
                all_pos_masks.append(torch.zeros(n_total, dtype=torch.bool, device=device))
                continue

            # cxcywh → xyxy, scale to absolute pixels
            cx, cy, w_, h_ = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
            x1 = (cx - w_ * 0.5) * W
            y1 = (cy - h_ * 0.5) * H
            x2 = (cx + w_ * 0.5) * W
            y2 = (cy + h_ * 0.5) * H
            gt_boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

            assigned_labels, assigned_boxes, assigned_centerness = self.atss_matcher(
                anchors_per_level, gt_boxes_abs, labels, self.strides
            )

            pos_mask = assigned_labels > 0   # fg is 1..C
            n_total = all_anchors.shape[0]

            # One-hot cls targets (0 for bg)
            cls_target = torch.zeros(n_total, self.num_classes, device=device)
            if pos_mask.any():
                fg_labels = (assigned_labels[pos_mask] - 1).clamp(0, self.num_classes - 1)
                cls_target[pos_mask] = F.one_hot(fg_labels, self.num_classes).float()

            all_cls_targets.append(cls_target)
            all_reg_targets.append(assigned_boxes)
            all_cen_targets.append(assigned_centerness)
            all_pos_masks.append(pos_mask)
            num_pos += int(pos_mask.sum())

        # Normalize by number of positive anchors across the batch
        num_pos_total = torch.tensor([num_pos], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos_total)
        num_pos_total = torch.clamp(num_pos_total / get_world_size(), min=1).item()

        # Stack across batch
        cls_targets = torch.stack(all_cls_targets)     # [B, N, C]
        cen_targets = torch.stack(all_cen_targets)     # [B, N]
        pos_masks = torch.stack(all_pos_masks)         # [B, N]
        reg_targets = torch.stack(all_reg_targets)     # [B, N, 4]

        # Classification loss (focal)
        loss_cls = _sigmoid_focal_loss(cls_flat, cls_targets, num_pos_total,
                                       alpha=self.focal_alpha)

        # Regression and centerness losses on positive anchors only
        if pos_masks.any():
            pos_reg_pred = reg_flat[pos_masks]    # [P, 4] DeltaXYWH deltas
            pos_reg_tgt = reg_targets[pos_masks]  # [P, 4] xyxy absolute

            # Expand anchors to match positive mask
            B = cls_flat.shape[0]
            all_anchors_b = all_anchors.unsqueeze(0).expand(B, -1, -1)[pos_masks]  # [P, 4]

            # Decode DeltaXYWH predictions to xyxy
            decoded_pred = _delta_xywh_decode(all_anchors_b, pos_reg_pred)

            # Centerness-weighted GIoU loss (matching reference Co-DETR)
            pos_cen_targets = cen_targets[pos_masks]
            giou = box_ops.generalized_box_iou(decoded_pred, pos_reg_tgt)
            giou_loss = 1 - torch.diag(giou)
            # Normalize by sum of centerness targets (matching reference)
            cen_sum = pos_cen_targets.sum().clamp(min=1.0)
            loss_bbox = (giou_loss * pos_cen_targets).sum() / cen_sum

            loss_centerness = F.binary_cross_entropy_with_logits(
                cen_flat[pos_masks], pos_cen_targets, reduction='sum'
            ) / num_pos_total
        else:
            loss_bbox = cls_flat.sum() * 0.0
            loss_centerness = cls_flat.sum() * 0.0

        return {
            'collab_loss_cls': loss_cls,
            'collab_loss_bbox': loss_bbox,
            'collab_loss_centerness': loss_centerness,
        }

    def forward(self, outputs, targets, return_indices=False):
        """Compute full CoDETR loss.

        Extends DINO SetCriterion.forward() with collaborative head losses.
        """
        # Standard DINO loss
        losses = super().forward(outputs, targets, return_indices=return_indices)

        # Collaborative head losses (training only, collab_outputs present)
        if 'collab_outputs' in outputs and outputs['collab_outputs']:
            # Infer image H, W from the first level feature map shape
            # cls_scores[0] has shape [B, num_classes, H_0, W_0]
            first_cls = outputs['collab_outputs'][0][0][0]
            H_feat, W_feat = first_cls.shape[-2:]
            stride0 = self.strides[0] if self.strides else 8
            img_H = H_feat * stride0
            img_W = W_feat * stride0

            for head_idx, collab_output in enumerate(outputs['collab_outputs']):
                collab_losses = self.compute_collab_loss(collab_output, targets, (img_H, img_W))
                suffix = '' if head_idx == 0 else f'_{head_idx}'
                for k, v in collab_losses.items():
                    losses[k + suffix] = v

        if return_indices:
            return losses

        return losses
