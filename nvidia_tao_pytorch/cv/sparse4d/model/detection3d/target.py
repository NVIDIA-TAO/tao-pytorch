# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target for Sparse4D."""

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..base_target import BaseTargetWithDenoising
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import X, Y, Z, W, L, H, YAW


class SparseBox3DTarget(BaseTargetWithDenoising):
    """Sparse box 3D target."""

    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        use_temporal_align=False,
        matching_cost_threshold=1e6,
        gt_assign_threshold=0.5,
    ):
        """Initialize the SparseBox3DTarget.

        Args:
            cls_weight (float): Class weight
            box_weight (float): Box weight
            alpha (float): Alpha
            gamma (float): Gamma
            eps (float): Epsilon
            reg_weights (list): Regression weights
            cls_wise_reg_weights (dict): Class-wise regression weights
            num_dn_groups (int): Number of DN groups
            dn_noise_scale (float): DN noise scale
            max_dn_gt (int): Maximum number of DN GTs
            add_neg_dn (bool): Whether to add negative DN
            num_temp_dn_groups (int): Number of temporary DN groups
            use_temporal_align (bool): Whether to use temporal alignment
            matching_cost_threshold (float): Matching cost threshold
            gt_assign_threshold (float): GT assign threshold
        """
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.use_temporal_align = use_temporal_align
        self.matching_cost_threshold = matching_cost_threshold
        self.gt_assign_threshold = gt_assign_threshold

    def encode_reg_target(self, box_target, device=None):
        """Encode the regression target."""
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1:],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
        instance_id_target=None,
        asset_id_target=None,
        visibility_score_target=None,
        gt_mapping_prev=None,
        use_hungarian_only=True,
        update_gt_indices=False,
        update_gt_index_mapping=False,
    ):
        """Sample the target.

        Args:
            cls_pred (torch.Tensor): Class predictions
            box_pred (torch.Tensor): Box predictions
            cls_target (torch.Tensor): Class targets
            box_target (torch.Tensor): Box targets
            instance_id_target (torch.Tensor): Instance ID targets
            asset_id_target (torch.Tensor): Asset ID targets
            visibility_score_target (torch.Tensor): Visibility score targets
            gt_mapping_prev (dict): GT mapping from previous frame
            use_hungarian_only (bool): Whether to use hungarian only
            update_gt_indices (bool): Whether to update GT indices
        """
        bs, num_pred, num_cls = cls_pred.shape

        if self.use_temporal_align and gt_mapping_prev is not None:
            # gather pred_idx, gt_idx, instance_id from previous frame
            query_indices_curr, gt_indices_curr = self._get_mapped_query_gt_indices(
                bs, gt_mapping_prev, instance_id_target,
                update_gt_indices=update_gt_indices,
            )

        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)

        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        gt_mapping_curr = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()  # [num_pred, num_gt]
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                if self.use_temporal_align and gt_mapping_prev is not None and not use_hungarian_only:
                    # set the cost of pred & gt indices from the previous frame to infinity
                    query_indices_selected, gt_indices_selected = self._update_cost_matrix(cost, query_indices_curr[i], gt_indices_curr[i])
                if np.min(cost) > self.matching_cost_threshold:
                    assign = (
                        np.array([], dtype=np.int64),
                        np.array([], dtype=np.int64)
                    )
                else:
                    assign = linear_sum_assignment(cost)
                    assigned_cost = cost[assign[0], assign[1]]
                    valid_mask = assigned_cost <= self.matching_cost_threshold
                    assign = (assign[0][valid_mask], assign[1][valid_mask])  # remove invalid assignments: previously assigned pairs
                assert len(assign[0]) == len(assign[1]), f"assign[0]: {len(assign[0])} assign[1]: {len(assign[1])} invalid assignments from temporal alignment during hungarian matching"

                if self.use_temporal_align and gt_mapping_prev is not None and not use_hungarian_only:
                    # append the previous frame assignments to the current frame assignments
                    assign = (
                        np.append(
                            np.array(query_indices_selected, dtype=np.int64),
                            assign[0]
                        ),
                        np.append(
                            np.array(gt_indices_selected, dtype=np.int64),
                            assign[1]
                        )
                    )
                    assert len(assign[0]) == len(assign[1]), f"assign[0]: {len(assign[0])} assign[1]: {len(assign[1])} invalid assignments from temporal alignment"

                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
                if self.use_temporal_align and update_gt_index_mapping:
                    # map from gt index to instance id
                    instance_ids = instance_id_target[i][assign[1]]  # length of list == n_gt
                    gt_mapping_curr.append(
                        {
                            pred_idx: [gt_idx, int(instance_id)]
                            for pred_idx, gt_idx, instance_id in zip(assign[0], assign[1], instance_ids)
                        }
                    )
                else:
                    gt_mapping_curr.append(None)
            else:
                indices.append([None, None])
                gt_mapping_curr.append(None)

        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )

        if instance_id_target is not None:
            output_instance_id_target = (
                instance_id_target[0].new_full([bs, num_pred], fill_value=-1, dtype=torch.int64)
            )
        else:
            output_instance_id_target = None

        if asset_id_target is not None:
            output_asset_id_target = (
                asset_id_target[0].new_full([bs, num_pred], fill_value=-1, dtype=torch.int64)
            )
        else:
            output_asset_id_target = None

        if visibility_score_target is not None:
            num_cams = visibility_score_target[0].shape[-1]
            output_visibility_score_target = (
                visibility_score_target[0].new_zeros([bs, num_pred, num_cams], dtype=torch.float32)
            )
        else:
            output_visibility_score_target = None

        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)

        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
            if instance_id_target is not None:
                output_instance_id_target[i, pred_idx] = instance_id_target[i][target_idx]
            if asset_id_target is not None:
                output_asset_id_target[i, pred_idx] = asset_id_target[i][target_idx]
            if visibility_score_target is not None:
                output_visibility_score_target[i, pred_idx] = visibility_score_target[i][target_idx]

        if not update_gt_index_mapping:
            gt_mapping_curr = gt_mapping_prev

        return output_cls_target, output_box_target, output_reg_weights, output_instance_id_target, output_asset_id_target, output_visibility_score_target, gt_mapping_curr

    def _cls_cost(self, cls_pred, cls_target):
        """Calculate the class cost."""
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (-(1 - cls_pred[i] + self.eps).log() * (1 - self.alpha) * cls_pred[i].pow(self.gamma))
                pos_cost = (-(cls_pred[i] + self.eps).log() * self.alpha * (1 - cls_pred[i]).pow(self.gamma))
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]]) * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        """Calculate the box cost."""
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None]) * instance_reg_weights[i][None] * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    ) * self.box_weight
                )
            else:
                cost.append(None)
        return cost

    def _get_mapped_query_gt_indices(
        self,
        bs,
        index_mapping_prev,
        instance_id_target,
        update_gt_indices=False,
    ):
        """Get the mapped query and gt indices."""
        query_indices_curr = []
        gt_indices_prev = []
        gt_indices_curr = []
        for i in range(bs):
            query_indices_batch = list(index_mapping_prev[i].keys())
            if len(query_indices_batch) == 0:
                # no pred_idx from previous frame: usually after reset
                query_indices_curr.append(None)
                gt_indices_curr.append(None)
                continue

            # prepare the mapping from pred_idx to gt_idx and instance_id
            mapped_indices = np.array(list(index_mapping_prev[i].values()))
            gt_indices_prev_batch = list(mapped_indices[:, 0])
            instance_ids_prev = mapped_indices[:, 1]

            gt_indices_curr_batch = []
            query_indices_to_remove = []
            for query_idx, instance_id_prev in zip(query_indices_batch, instance_ids_prev):
                if instance_id_prev in instance_id_target[i]:
                    if update_gt_indices:
                        # find the gt_idx in the current frame
                        gt_indices_curr_batch.append(
                            int((instance_id_target[i] == instance_id_prev).nonzero().squeeze())
                        )
                else:
                    # gt_idx not found in the current frame
                    query_indices_to_remove.append(query_idx)

            if len(query_indices_to_remove) > 0:
                for query_idx_prev in query_indices_to_remove:
                    del index_mapping_prev[i][query_idx_prev]
                query_indices_batch = list(index_mapping_prev[i].keys())
                mapped_indices = np.array(list(index_mapping_prev[i].values()))
                gt_indices_prev_batch = list(mapped_indices[:, 0])

            query_indices_curr.append(query_indices_batch)
            gt_indices_prev.append(gt_indices_prev_batch)

            if update_gt_indices:
                gt_indices_curr.append(gt_indices_curr_batch)

        if not update_gt_indices:
            gt_indices_curr = gt_indices_prev

        return query_indices_curr, gt_indices_curr

    def _update_cost_matrix(self, cost, query_indices_curr, gt_indices_curr):
        """Update the cost matrix."""
        query_indices_selected = []
        gt_indices_selected = []
        if query_indices_curr is not None and gt_indices_curr is not None:
            for qidx, gidx in zip(query_indices_curr, gt_indices_curr):
                if cost[qidx, gidx] < self.gt_assign_threshold:
                    query_indices_selected.append(qidx)
                    gt_indices_selected.append(gidx)

        if query_indices_curr is not None:
            cost[query_indices_selected] = 1e8
        if gt_indices_curr is not None:
            cost[:, gt_indices_selected] = 1e8
        return query_indices_selected, gt_indices_selected

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        """Get the DN anchors."""
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        """Update the DN."""
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for _, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3, f"temp_meta.dim(): {temp_meta.dim()} != 3, temp_meta: {temp_meta}"
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        """Cache the DN."""
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
