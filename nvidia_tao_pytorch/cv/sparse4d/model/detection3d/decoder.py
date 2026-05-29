# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder for Sparse4D."""
import torch
from typing import Optional

from nvidia_tao_pytorch.cv.sparse4d.model.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, CNS


def decode_box(box):
    """Decode the box.

    Args:
        box (torch.Tensor): Box to decode.

    Returns:
        torch.Tensor: Decoded box.
    """
    yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
    box = torch.cat(
        [
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
            box[:, VX:],
        ],
        dim=-1,
    )
    return box


class SparseBox3DDecoder:
    """Sparse box 3D decoder."""

    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sort_results: bool = True,
    ):
        """Initialize SparseBox3DDecoder.

        Args:
            num_output (int): Number of output boxes.
            score_threshold (float): Score threshold for filtering boxes.
            sort_results (bool): Whether to sort boxes by score.
        """
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sort_results = sort_results

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        instance_feature=None,
        qulity=None,
        reid_feature=None,
        visibility_scores=None,
        output_idx=-1,
    ):
        """Decode the boxes.

        Args:
            cls_scores (torch.Tensor): Class scores.
            box_preds (torch.Tensor): Box predictions.
            instance_id (torch.Tensor): Instance IDs.
            instance_feature (torch.Tensor): Instance features.
            qulity (torch.Tensor): Quality.
            reid_feature (torch.Tensor): Re-ID features.
            visibility_scores (torch.Tensor): Visibility scores.
            output_idx (int): Output index.
        """
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, _, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sort_results
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)
            cls_scores_origin = torch.gather(cls_scores_origin, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids.cpu()
            if instance_feature is not None:
                feats = instance_feature[i, indices[i]]
                if self.score_threshold is not None:
                    feats = feats[mask[i]]
                output[-1]["instance_feats"] = feats.cpu()
            if reid_feature is not None:
                if isinstance(reid_feature, list):
                    reid_feature = reid_feature[bs]
                reid_feats = reid_feature[i, indices[i]]
                if self.score_threshold is not None:
                    reid_feats = reid_feats[mask[i]]
                output[-1]["reid_feats"] = reid_feats.cpu()
        return output
