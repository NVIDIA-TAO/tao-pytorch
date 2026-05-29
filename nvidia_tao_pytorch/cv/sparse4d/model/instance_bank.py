# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instance Bank for Sparse4D."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.sparse4d.model.anchor_selector import AnchorSelector


def topk(confidence, k, *inputs):
    """Topk function."""
    bs, N = confidence.shape[:2]
    confidence, indices_raw = torch.topk(confidence, k, dim=1)
    indices = (
        indices_raw + torch.arange(bs, device=indices_raw.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for single_input in inputs:
        outputs.append(single_input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs, indices_raw


def topk_gt_index_mapping(indices, gt_index_mapping):
    """Topk gt index mapping function."""
    bs = len(indices)
    output = []
    for i in range(bs):
        batch_indices = indices[i]  # shape: [k]
        batch_gt_mapping = gt_index_mapping[i]  # list of tensors
        batch_dict = {}
        n_unmatched = 0
        # For each index in batch_indices, find matching elements in batch_gt_mapping
        for idx in batch_indices:
            # Find elements where batch_gt_mapping[j][0] equals idx
            if int(idx) in batch_gt_mapping:
                batch_dict[int(idx)] = batch_gt_mapping[int(idx)]
            else:
                # idx is not matched to any gt
                n_unmatched += 1
        output.append(batch_dict)
    return output


class InstanceBank(nn.Module):
    """Instance bank."""

    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        reid_dims=None,
        use_temporal_align=False,
        grid_filter=None
    ):
        """Initialize InstanceBank.

        Args:
            num_anchor (int): Number of anchor.
            embed_dims (int): Embedding dimensions.
            anchor (str): Anchor path.
            anchor_handler (nn.Module): Anchor handler.
            num_temp_instances (int): Number of temporary instances.
            default_time_interval (float): Default time interval.
            confidence_decay (float): Confidence decay.
            anchor_grad (bool): Anchor gradient.
            feat_grad (bool): Feature gradient.
            max_time_interval (float): Maximum time interval.
            reid_dims (int): Re-ID dimensions.
            use_temporal_align (bool): Whether to use temporal alignment.
        """
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.reid_dims = reid_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.use_temporal_align = use_temporal_align
        self.time_interval = self.default_time_interval

        # filter out new anchors that are nearby cached anchors
        self.grid_filter = AnchorSelector(grid_size=grid_filter) if grid_filter is not None else None

        # Setup anchor handler if provided
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            if anchor == "":
                logging.info("Initializing anchor with zeros. Please provide a valid anchor path.")
                anchor = np.zeros([num_anchor, 3])
            else:
                anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        if self.reid_dims is not None:
            self.reid_feature = nn.Parameter(
                torch.zeros([self.anchor.shape[0], self.reid_dims]),
                requires_grad=feat_grad,
            )
        else:
            self.reid_feature = None
        self.reset()

    def init_weight(self):
        """Initialize the weight."""
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)
        if self.reid_feature is not None and self.reid_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.reid_feature.data, gain=1)

    def reset(self):
        """Reset the InstanceBank."""
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0
        self.group_indices = None
        self.scene_indices = None
        self.reset_gt_index_mapping()

    def reset_gt_index_mapping(self):
        """Reset the gt index mapping."""
        # gt_index_mapping: {query_idx: [gt_idx, instance_id]}
        self.gt_index_mapping = None  # direct output from assignment
        self.cached_gt_index_mapping = None  # cached assignment pairs
        self.cached_query_indices = None

    def reset_gt_index_mapping_by_data_indices(self, reset_flags):
        """Reset the gt index mapping by data indices."""
        if any(reset_flags):
            self.gt_index_mapping = None
            self.cached_gt_index_mapping = None
            self.cached_query_indices = None

    def get(self, batch_size, metas=None, dn_metas=None):
        """Get the instance feature, anchor, and re-ID feature."""
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        if self.reid_feature is not None:
            reid_feature = torch.tile(
                self.reid_feature[None], (batch_size, 1, 1)
            )
        else:
            reid_feature = None

        if (
            self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                if "img_metas" in metas and "T_global" in metas["img_metas"][0]:
                    # check if T_global is defined (ego-motion)
                    T_temp2cur = self.cached_anchor.new_tensor(
                        np.stack(
                            [
                                x["T_global_inv"]
                                @ self.metas["img_metas"][i]["T_global"]
                                for i, x in enumerate(metas["img_metas"])
                            ]
                        )
                    )
                else:
                    T_temp2cur = None
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None and dn_metas is not None and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
            # Ensure time_interval depends on timestamp
            timestamp = torch.tensor([0])  # used for onnx export
            time_interval = time_interval + (timestamp * 0).sum()
        self.time_interval = time_interval
        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
            reid_feature,
        )

    def update(self, instance_feature, anchor, confidence):
        """Update the instance feature, anchor, and confidence."""
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        if self.grid_filter is not None:
            # filter out new anchors that are nearby cached anchors
            selected_anchor, selected_feature, _ = self.grid_filter.select_top_anchors(
                anchor, self.cached_anchor, instance_feature, confidence, N)
        else:
            _, (selected_feature, selected_anchor), _ = topk(
                confidence, N, instance_feature, anchor
            )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        """Cache the instance feature, anchor, and confidence."""
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        if self.use_temporal_align:
            (
                self.confidence,
                (self.cached_feature, self.cached_anchor),
                topk_indices,
            ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
            self.set_cached_query_indices(topk_indices)
        else:
            (
                self.confidence,
                (self.cached_feature, self.cached_anchor),
                _,
            ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        """Get the instance ID."""
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        """Update the instance ID."""
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )

    def get_gt_index_mapping(self):
        """Get the gt index mapping."""
        return self.cached_gt_index_mapping

    def set_gt_index_mapping(self, gt_index_mapping):
        """Set the gt index mapping."""
        self.gt_index_mapping = gt_index_mapping

    def cache_gt_index_mapping(self, gt_index_mapping):
        """Cache the gt index mapping."""
        self.gt_index_mapping = gt_index_mapping
        self.cached_gt_index_mapping = topk_gt_index_mapping(self.cached_query_indices, self.gt_index_mapping)

    def update_query_indices_in_cached_gt_index_mapping(self, query_indices):
        """Update the query indices in cached gt index mapping."""
        gt_index_mapping_updated = []
        for i in range(len(query_indices)):
            gt_index_mapping_updated_batch = {}
            query_index_mapping_c2n = {}
            for query_idx_next, query_idx_curr in enumerate(query_indices[i]):
                query_index_mapping_c2n[int(query_idx_curr)] = query_idx_next
            for query_idx_curr, (gt_idx, instance_id) in self.cached_gt_index_mapping[i].items():
                gt_index_mapping_updated_batch[query_index_mapping_c2n[query_idx_curr]] = (gt_idx, instance_id)
            gt_index_mapping_updated.append(gt_index_mapping_updated_batch)
        self.cached_gt_index_mapping = gt_index_mapping_updated

    def set_cached_query_indices(self, cached_query_indices):
        """Set the cached query indices."""
        self.cached_query_indices = []
        for i in range(len(cached_query_indices)):
            self.cached_query_indices.append(cached_query_indices[i])

    def get_cached_query_indices(self):
        """Get the cached query indices."""
        return self.cached_query_indices

    def set_data_indices(self, group_indices, scene_indices):
        """Set the data indices."""
        self.group_indices = group_indices
        self.scene_indices = scene_indices

    def get_data_indices(self):
        """Get the data indices."""
        return self.group_indices, self.scene_indices
