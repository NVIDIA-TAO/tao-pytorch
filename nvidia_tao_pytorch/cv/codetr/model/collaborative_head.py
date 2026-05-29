# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATSS collaborative auxiliary head for Co-DETR training."""

import math
import torch
import torch.nn as nn


class Scale(nn.Module):
    """Learnable per-level scale factor for regression outputs.

    Args:
        init_value (float): initial scalar value for the learnable scale
            parameter.
    """

    def __init__(self, init_value=1.0):
        """Initialize scale.

        Args:
            init_value (float): initial scalar value for the learnable scale
                parameter.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        """Apply scale."""
        return x * self.scale


class ATSSCollaborativeHead(nn.Module):
    """ATSS-based collaborative auxiliary head for Co-DETR.

    Applied to encoder-projected backbone features during training only.
    Provides additional one-to-many supervision signal to strengthen the encoder.

    Args:
        num_classes (int): number of foreground classes, excluding background.
        in_channels (int): number of channels in each input feature map.
        feat_channels (int): number of channels used inside the classification
            and regression towers.
        num_convs (int): number of convolution layers in each tower.
        strides (tuple[int]): feature-map stride for each input level. The
            values are used for anchor generation and must match the feature
            levels passed to ``forward``.

    Reference: "DETRs with Collaborative Hybrid Assignments Training" (NeurIPS 2023)
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 feat_channels=256,
                 num_convs=4,
                 strides=(8, 16, 32, 64)):
        """Initialize ATSSCollaborativeHead.

        Args:
            num_classes (int): number of foreground classes (no background).
            in_channels (int): input channel count (= hidden_dim from encoder projection).
            feat_channels (int): feature channels inside the head towers.
            num_convs (int): number of conv layers in each tower.
            strides (tuple): per-level stride values. Used to generate anchors.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.strides = strides

        self._build_head()
        self._init_weights()

    def _build_head(self):
        """Build classification and regression tower conv layers."""
        cls_convs = []
        reg_convs = []
        for i in range(self.num_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, self.feat_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, self.feat_channels),
                nn.ReLU(inplace=True),
            ))
            reg_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, self.feat_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, self.feat_channels),
                nn.ReLU(inplace=True),
            ))
        self.cls_convs = nn.ModuleList(cls_convs)
        self.reg_convs = nn.ModuleList(reg_convs)

        # Output projection layers (3x3 to match reference Co-DETR)
        self.cls_pred = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.centerness_pred = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        # Per-level learnable scale for regression
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_weights(self):
        """Initialize classification bias for focal loss stability."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
        nn.init.normal_(self.centerness_pred.weight, std=0.01)
        nn.init.constant_(self.centerness_pred.bias, 0)

    def forward(self, features):
        """Compute per-level class scores, box predictions and centerness.

        Args:
            features (list[Tensor]): projected multi-scale features,
                each of shape [B, in_channels, H_i, W_i].

        Returns:
            cls_scores (list[Tensor]): [B, num_classes, H_i, W_i] per level.
            bbox_preds (list[Tensor]): [B, 4, H_i, W_i] per level (ltrb).
            centernesses (list[Tensor]): [B, 1, H_i, W_i] per level.
        """
        assert len(features) == len(self.strides), (
            f"Expected {len(self.strides)} feature levels, got {len(features)}"
        )
        cls_scores, bbox_preds, centernesses = [], [], []
        for feat, scale in zip(features, self.scales):
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            cls_scores.append(self.cls_pred(cls_feat))
            bbox_preds.append(scale(self.reg_pred(reg_feat)).float())
            centernesses.append(self.centerness_pred(reg_feat))

        return cls_scores, bbox_preds, centernesses

    @staticmethod
    def generate_anchors(feature_maps, strides, octave_base_scale=8):
        """Generate square anchors for each spatial location.

        Matches the reference Co-DETR anchor generator:
        octave_base_scale=8, scales_per_octave=1, ratios=[1.0].
        Each anchor is a square of size ``octave_base_scale * stride``.

        Args:
            feature_maps (list[Tensor]): feature tensors, shape [B, C, H_i, W_i].
            strides (list[int]): stride for each feature level.
            octave_base_scale (int): anchor size multiplier.

        Returns:
            anchors_per_level (list[Tensor]): each [H_i*W_i, 4] in xyxy format.
        """
        anchors_per_level = []
        for feat, stride in zip(feature_maps, strides):
            _, _, h, w = feat.shape
            device = feat.device
            # Grid centers in image coordinates
            shift_x = (torch.arange(w, device=device) + 0.5) * stride
            shift_y = (torch.arange(h, device=device) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            cx = grid_x.reshape(-1)
            cy = grid_y.reshape(-1)
            half = octave_base_scale * stride * 0.5
            anchors = torch.stack([cx - half, cy - half, cx + half, cy + half], dim=-1)
            anchors_per_level.append(anchors)
        return anchors_per_level


class ATSSMatcher:
    """Adaptive Training Sample Selection matcher.

    For each GT box, selects topk anchor candidates per level by center
    distance, then applies an adaptive IoU threshold to determine positives.

    Args:
        topk (int): number of nearest anchor candidates to consider per
            ground-truth box and feature level.

    Reference: "Bridging the Gap Between Anchor-based and Anchor-free
    Detection via Adaptive Training Sample Selection" (CVPR 2020)
    """

    def __init__(self, topk=9):
        """Initialize matcher.

        Args:
            topk (int): number of candidate anchors per GT per level.
        """
        self.topk = topk

    def __call__(self, anchors_per_level, gt_boxes, gt_labels, strides):
        """Match anchors to ground-truth boxes.

        Args:
            anchors_per_level (list[Tensor]): per-level anchor tensors [N_i, 4] xyxy.
            gt_boxes (Tensor): [G, 4] in xyxy (image coords).
            gt_labels (Tensor): [G] class indices (0-based).
            strides (list[int]): stride per level.

        Returns:
            assigned_labels (Tensor): [N_total] assigned class index (-1=ignore, 0=bg, 1..C=fg).
            assigned_boxes (Tensor): [N_total, 4] assigned GT boxes in ltrb (pixel) format.
            assigned_centerness (Tensor): [N_total] centerness targets (0 for bg).
        """
        all_anchors = torch.cat(anchors_per_level, dim=0)   # [N_total, 4]
        n_total = all_anchors.shape[0]
        num_gts = gt_boxes.shape[0]
        device = all_anchors.device

        if num_gts == 0:
            # No GT: all anchors are background
            assigned_labels = torch.zeros(n_total, dtype=torch.long, device=device)
            assigned_boxes = torch.zeros(n_total, 4, device=device)
            assigned_centerness = torch.zeros(n_total, device=device)
            return assigned_labels, assigned_boxes, assigned_centerness

        # Anchor center coordinates
        anchor_cx = (all_anchors[:, 0] + all_anchors[:, 2]) * 0.5
        anchor_cy = (all_anchors[:, 1] + all_anchors[:, 3]) * 0.5

        # GT center coordinates
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5

        # Pairwise L2 distance: [num_gts, n_total]
        dist = ((anchor_cx[None] - gt_cx[:, None]) ** 2 +
                (anchor_cy[None] - gt_cy[:, None]) ** 2).sqrt()

        # Per-level anchor counts
        level_sizes = [a.shape[0] for a in anchors_per_level]
        level_starts = [0] + list(torch.cumsum(torch.tensor(level_sizes), 0)[:-1].tolist())

        candidate_mask = torch.zeros(num_gts, n_total, dtype=torch.bool, device=device)
        for gt_idx in range(num_gts):
            for start, size in zip(level_starts, level_sizes):
                level_dist = dist[gt_idx, start:start + size]
                topk_k = min(self.topk, size)
                _, topk_idx = level_dist.topk(topk_k, largest=False)
                candidate_mask[gt_idx, start + topk_idx] = True

        # Compute IoU for candidate pairs
        iou_matrix = self._box_iou(gt_boxes, all_anchors)   # [num_gts, n_total]

        # Adaptive IoU threshold = mean + std over topk candidates per GT
        assigned_gt = torch.full((n_total,), -1, dtype=torch.long, device=device)
        for gt_idx in range(num_gts):
            cand = candidate_mask[gt_idx]
            cand_iou = iou_matrix[gt_idx][cand]
            if cand_iou.numel() == 0:
                continue
            threshold = cand_iou.mean() + cand_iou.std()
            # Positive: candidate AND IoU >= threshold AND anchor inside GT box
            pos = cand.clone()
            pos_iou = iou_matrix[gt_idx]
            pos &= pos_iou >= threshold
            # Anchor center must be inside GT box
            inside = ((anchor_cx >= gt_boxes[gt_idx, 0]) &
                      (anchor_cy >= gt_boxes[gt_idx, 1]) &
                      (anchor_cx <= gt_boxes[gt_idx, 2]) &
                      (anchor_cy <= gt_boxes[gt_idx, 3]))
            pos &= inside
            # Assign: resolve conflicts later (highest IoU wins)
            pos_idx = pos.nonzero(as_tuple=False).squeeze(1)
            for idx in pos_idx:
                idx = idx.item()
                if assigned_gt[idx] == -1:
                    assigned_gt[idx] = gt_idx
                else:
                    # Keep the GT with higher IoU
                    old_gt = assigned_gt[idx]
                    if iou_matrix[gt_idx, idx] > iou_matrix[old_gt, idx]:
                        assigned_gt[idx] = gt_idx

        # Build assignment tensors
        assigned_labels = torch.zeros(n_total, dtype=torch.long, device=device)   # 0 = background
        assigned_boxes = torch.zeros(n_total, 4, device=device)
        assigned_centerness = torch.zeros(n_total, device=device)

        pos_mask = assigned_gt >= 0
        if pos_mask.any():
            pos_gt = assigned_gt[pos_mask]
            assigned_labels[pos_mask] = gt_labels[pos_gt] + 1     # shift by 1: 0 reserved for bg
            assigned_boxes[pos_mask] = gt_boxes[pos_gt]

            # Centerness targets based on ltrb distances to assigned GT
            pos_anchors = all_anchors[pos_mask]
            pos_gt_boxes = gt_boxes[pos_gt]
            anchor_cx_pos = (pos_anchors[:, 0] + pos_anchors[:, 2]) * 0.5
            anchor_cy_pos = (pos_anchors[:, 1] + pos_anchors[:, 3]) * 0.5
            left = anchor_cx_pos - pos_gt_boxes[:, 0]
            top = anchor_cy_pos - pos_gt_boxes[:, 1]
            right = pos_gt_boxes[:, 2] - anchor_cx_pos
            bottom = pos_gt_boxes[:, 3] - anchor_cy_pos
            ltrb = torch.stack([left, top, right, bottom], dim=-1).clamp(min=0)
            centerness = (ltrb[:, :2].min(dim=1)[0] / ltrb[:, :2].max(dim=1)[0].clamp(min=1e-6) *
                          ltrb[:, 2:].min(dim=1)[0] / ltrb[:, 2:].max(dim=1)[0].clamp(min=1e-6)).sqrt()
            assigned_centerness[pos_mask] = centerness

        return assigned_labels, assigned_boxes, assigned_centerness

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """Compute pairwise IoU. boxes in xyxy format.

        Args:
            boxes1 (Tensor): [M, 4].
            boxes2 (Tensor): [N, 4].

        Returns:
            iou (Tensor): [M, N].
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-6)
