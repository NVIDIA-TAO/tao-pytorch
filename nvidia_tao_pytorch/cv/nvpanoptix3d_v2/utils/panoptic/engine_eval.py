# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Panoptic post-processing for NVPanoptix3Dv2 model outputs."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def panoptic_inference(
    mask_cls: torch.Tensor,
    mask_pred: torch.Tensor,
    target_hw: Tuple[int, int],
    label_mode: str = "sigmoid",
    cls_threshold: float = 0.1,
    mask_threshold: float = 0.25,
    overlap_threshold: float = 0.5,
    presence_logits: Optional[torch.Tensor] = None,
    objectness_logits: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """Convert raw model logits into panoptic segmentation maps.

    Args:
        mask_cls:   [B, Q, C] class logits
        mask_pred:  [B, S, Q, H_feat, W_feat] mask logits (pre-sigmoid)
        target_hw:  (H, W) to resize masks to
        label_mode: "sigmoid" or "softmax"
        cls_threshold: minimum confidence to keep a query
        mask_threshold: binarisation threshold for masks
        overlap_threshold: minimum ratio of original mask area to keep
        presence_logits: optional [B, C] from presence head — gates per-class scores
        objectness_logits: optional [B, Q] matched/unmatched query logits

    Returns:
        list of B dicts, each containing:
          - ``pan``:           [S, H, W] int32 panoptic segment IDs
          - ``segments_info``: list of {id, category_id, score, area}
    """
    B, S, Q, _, _ = mask_pred.shape
    H, W = target_hw

    mask_pred = mask_pred.sigmoid()
    mask_pred = mask_pred.reshape(B * S, Q, mask_pred.shape[-2], mask_pred.shape[-1])
    mask_pred = F.interpolate(mask_pred, size=(H, W), mode="bilinear", align_corners=False)
    mask_pred = mask_pred.reshape(B, S, Q, H, W)

    presence_probs = None
    if presence_logits is not None:
        presence_probs = presence_logits.sigmoid()  # [B, C]
    objectness_probs = None
    if objectness_logits is not None:
        objectness_probs = objectness_logits.sigmoid()  # [B, Q]

    results = []
    for bi in range(B):
        cls_i = mask_cls[bi]
        masks_i = mask_pred[bi]
        masks_i = masks_i.transpose(0, 1)

        if label_mode == "sigmoid":
            query_probs = cls_i.sigmoid()             # [Q, C]
            if presence_probs is not None:
                query_probs = query_probs * presence_probs[bi].unsqueeze(0)
            scores, labels = query_probs.max(-1)
            if objectness_probs is not None:
                scores = scores * objectness_probs[bi]
            keep = scores > cls_threshold
        else:
            scores, labels = F.softmax(cls_i, dim=-1).max(-1)
            if objectness_probs is not None:
                scores = scores * objectness_probs[bi]
            num_classes = cls_i.shape[-1] - 1
            keep = labels.ne(num_classes) & (scores > cls_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = masks_i[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

        panoptic_seg = torch.zeros((S, H, W), dtype=torch.int32, device=masks_i.device)
        segments_info = []

        if cur_masks.shape[0] == 0:
            results.append({"pan": panoptic_seg, "segments_info": segments_info})
            continue

        cur_mask_ids = cur_prob_masks.argmax(0)
        current_segment_id = 0

        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= mask_threshold)
            mask_area = mask.sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id
                segments_info.append({
                    "id": current_segment_id,
                    "category_id": pred_class,
                    "score": float(cur_scores[k].item()),
                    "area": mask_area,
                })

        results.append({"pan": panoptic_seg, "segments_info": segments_info})

    return results
