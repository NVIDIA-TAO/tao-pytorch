# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Criterion Loss functions."""

import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.target import SparseBox3DTarget
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import X, Y, Z, SIN_YAW, COS_YAW, CNS, YNS


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce the mean of a tensor across all GPUs/nodes.

    Args:
        tensor (torch.Tensor): The tensor to reduce.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    # If not distributed, just return the mean.
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    # All-reduce the sum across GPUs/nodes
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # Divide by world size
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


class SetCriterion(nn.Module):
    """SetCriterion class for Sparse4D.

    This class defines the criterion for the Sparse4D model.
    It computes the classification and regression losses for the model.
    """

    def __init__(self, model_config, instance_bank):
        """ Create the criterion.

        Args:
            model_config (dict): Model configuration.
            instance_bank (InstanceBank): Instance bank.
        """
        super().__init__()
        sampler_config = model_config["head"]["sampler"]
        self.sampler = SparseBox3DTarget(
            num_dn_groups=sampler_config["num_dn_groups"],
            num_temp_dn_groups=sampler_config["num_temp_dn_groups"],
            dn_noise_scale=sampler_config["dn_noise_scale"],
            max_dn_gt=sampler_config["max_dn_gt"],
            add_neg_dn=sampler_config["add_neg_dn"],
            cls_weight=sampler_config["cls_weight"],
            box_weight=sampler_config["box_weight"],
            reg_weights=sampler_config["reg_weights"],
            use_temporal_align=model_config["use_temporal_align"],
        )
        head_config = model_config["head"]
        self.use_reid_sampling = head_config["use_reid_sampling"]
        self.reg_weights = head_config["reg_weights"]
        self.cls_threshold_to_reg = head_config["cls_threshold_to_reg"]
        self.cls_loss_config = head_config["loss"]["cls"]
        self.loss_cls = FocalLoss(gamma=self.cls_loss_config["gamma"], alpha=self.cls_loss_config["alpha"], loss_weight=self.cls_loss_config["loss_weight"])
        self.reg_loss_config = model_config["head"]["loss"]["reg"]
        self.loss_reg = SparseBox3DLoss(box_weight=self.reg_loss_config["box_weight"], valid_vel_weight=head_config["valid_vel_weight"])
        self.id_loss_config = model_config["head"]["loss"]["id"]
        self.loss_id = CrossEntropyLabelSmooth(num_ids=self.id_loss_config["num_ids"])
        self.loss_visibility = nn.BCELoss()
        self.use_temporal_align = model_config["use_temporal_align"]
        self.num_single_frame_decoder = head_config["num_single_frame_decoder"]
        self.loss_depth = DenseDepthLoss()
        self.instance_bank = instance_bank

    def forward(self, raw_model_outs, data, feature_maps=None):
        """Computes loss."""
        # ===================== prediction losses ======================
        model_outs, depths = raw_model_outs
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]

        reid_features = model_outs["reid_feature"] if self.use_reid_sampling else [None] * len(cls_scores)
        pred_visibility_scores = model_outs["visibility_scores"] if self.use_reid_sampling else [None] * len(cls_scores)
        predicted_ids = model_outs["predicted_id"] if self.use_reid_sampling else [None] * len(cls_scores)

        if self.use_temporal_align:
            gt_index_mapping_prev = self.instance_bank.get_gt_index_mapping()  # use the same mapping for different decoders
        else:
            gt_index_mapping_prev = None

        output = {}
        gt_index_mapping_curr = gt_index_mapping_prev
        for decoder_idx, (cls, reg, qt, _, pred_visibility_score, predicted_id) in enumerate(
            zip(cls_scores, reg_preds, quality, reid_features, pred_visibility_scores, predicted_ids)
        ):
            if self.use_temporal_align:
                if gt_index_mapping_prev is None:  # first frame: Hungarian only
                    use_hungarian_only = True
                    update_gt_indices = False
                    update_gt_index_mapping = True
                else:  # other frames
                    if decoder_idx + 1 == self.num_single_frame_decoder:  # first decoder
                        # query_indices = None
                        use_hungarian_only = True
                        update_gt_indices = False
                        update_gt_index_mapping = False
                    else:  # other decoders
                        # query_indices = cached_query_indices
                        use_hungarian_only = False
                        update_gt_indices = True
                        update_gt_index_mapping = True
            else:
                use_hungarian_only = True
                update_gt_indices = False
                update_gt_index_mapping = False

            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights, _, asset_id_target, visibility_score_target, gt_index_mapping_curr = self.sampler.sample(
                cls,
                reg,
                data["gt_labels_3d"],
                data["gt_bboxes_3d"],
                data["instance_id"],
                data["asset_id"],
                data["gt_visibility"] if "gt_visibility" in data else None,
                gt_index_mapping_curr,
                use_hungarian_only=use_hungarian_only,
                update_gt_indices=update_gt_indices,
                update_gt_index_mapping=update_gt_index_mapping,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = torch.max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)),
                torch.tensor(1.0, dtype=reg.dtype, device=reg.device)
            )

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            if self.use_reid_sampling:
                asset_id_target_reshaped = asset_id_target.reshape(-1)
                predicted_id = predicted_id.reshape(-1, predicted_id.shape[-1])
                num_ids = predicted_id.shape[-1]
                valid_mask = (asset_id_target_reshaped >= 0) & (asset_id_target_reshaped < num_ids)
                predicted_id_valid = predicted_id[valid_mask]
                asset_id_valid = asset_id_target_reshaped[valid_mask]
                if valid_mask.sum() == 0:
                    id_loss = predicted_id_valid.new_tensor(0.0)
                else:
                    id_loss = self.loss_id(predicted_id_valid, asset_id_valid)

                output[f"loss_id_{decoder_idx}"] = id_loss

                visibility_loss = self.loss_visibility(pred_visibility_score, visibility_score_target)
                output[f"loss_visibility_{decoder_idx}"] = visibility_loss

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if self.use_temporal_align:
            self.instance_bank.cache_gt_index_mapping(gt_index_mapping_curr)
            query_indices = self.instance_bank.get_cached_query_indices()
            self.instance_bank.update_query_indices_in_cached_gt_index_mapping(query_indices)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if ("temp_dn_valid_mask" in model_outs and decoder_idx == self.num_single_frame_decoder):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )

            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        output["loss_dense_depth"] = self.loss_depth(depths, data["gt_depth"])
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """Prepare for denoising loss."""
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss, as described in:
    Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

    This version replicates the functionality commonly found in mmcv,
    supporting sigmoid-based focal loss, alpha, gamma, reduction, and
    optional use of an avg_factor instead of a standard mean across the batch.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """
        Args:
            use_sigmoid (bool): If True, uses a sigmoid + binary focal loss.
                                If False, uses softmax + cross-entropy focal loss.
            gamma (float): Exponent of the modulating factor (1 - pt).
            alpha (float): Weighting factor for positive examples.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'.
            loss_weight (float): Multiplied by the final loss value.
        """
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None):
        """
        Forward computation of focal loss.

        Args:
            pred (Tensor): Model predictions of shape (N, C) or (N, ) if binary.
                           Typically these are raw logits (not probabilities).
            target (Tensor): Corresponding ground-truth labels, same shape as pred
                             if use_sigmoid=True for multi-label.
                             If it is multi-class with use_sigmoid=False, then `target`
                             can be shape (N,) with class indices in [0, C-1].
            weight (Tensor, optional): Per-sample weighting (broadcastable to pred).
            avg_factor (int or float, optional): If set, this will be used to
                                                 normalize the total loss instead
                                                 of dividing by the batch size.
        """
        num_classes = pred.size(1)
        valid_mask = (target >= 0) & (target < num_classes)  # ignore negative/out-of-range
        if not valid_mask.any():
            # No valid samples, return zero loss
            return pred.new_tensor(0.0)

        # Filter pred, target, and weight by this mask
        pred = pred[valid_mask]
        valid_target = target[valid_mask]
        if weight is not None:
            weight = weight[valid_mask]

        # Now safe to do one_hot
        one_hot_target = F.one_hot(valid_target, num_classes=num_classes).float()

        log_pt = F.binary_cross_entropy_with_logits(
            pred,
            one_hot_target,              # <-- use the one-hot
            reduction='none'
        )
        p = torch.sigmoid(pred)
        pt = p * one_hot_target + (1 - p) * (1 - one_hot_target)

        focal_weight = ((self.alpha * one_hot_target + (1 - self.alpha) * (1 - one_hot_target)) * ((1 - pt) ** self.gamma))

        loss = focal_weight * log_pt

        # Apply per-sample weighting if provided (e.g., for imbalance in data).
        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        # Handle reduction
        if self.reduction == 'mean':
            # If the user provides avg_factor, we divide the sum by avg_factor
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # If 'none', we just return the per-element loss.

        # Multiply by the user-specified scalar to get final focal loss
        loss = self.loss_weight * loss
        return loss


class DenseDepthLoss(nn.Module):
    """DenseDepthLoss class."""

    def __init__(self, loss_weight=0.2, max_depth=60):
        super(DenseDepthLoss, self).__init__()
        self.loss_weight = loss_weight
        self.max_depth = max_depth

    def forward(self, depth_preds, gt_depths):
        """Calculate depth prediction loss.

        Args:
            depth_preds: Predicted depth maps
            gt_depths: Ground truth depth maps

        Returns:
            Depth loss value
        """
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            # Reshape predictions and ground truth
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)

            # Filter valid points
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]

            # Clip predicted depth to valid range
            pred = torch.clip(pred, 0.0, self.max_depth)

            # Calculate L1 loss with full precision
            error = torch.abs(pred - gt).sum()
            _loss = (error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight)

            loss = loss + _loss

        return loss


class GaussianFocalLoss(nn.Module):
    """
    A simplified version of Gaussian Focal Loss from the CornerNet/CenterNet family.
    Typically used for heatmap-based object center detection.
    Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0):
        """
        Args:
            alpha (float): Exponent for the (1 - prob) or prob terms (often called `alpha` in cornernet).
            gamma (float): Exponent for modulating factor (sometimes called `beta` or `gamma`).
            reduction (str): 'none' | 'mean' | 'sum'.
            loss_weight (float): Overall scalar on the final loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Model output logits of shape [N, C, H, W], or similar.
            target (Tensor): Ground-truth heatmap in [0, 1], same shape as pred.
                             1 indicates the peak (center), 0 indicates background,
                             fractional values near 1 can represent Gaussian distribution around the center.
        Returns:
            torch.Tensor: Loss (scalar or per-element).
        """
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights

        if self.reduction == 'mean':
            loss = (pos_loss + neg_loss).mean()
        elif self.reduction == 'sum':
            loss = (pos_loss + neg_loss).sum()

        return self.loss_weight * loss


class SparseBox3DLoss(nn.Module):
    """SparseBox3DLoss class."""

    def __init__(
        self,
        box_weight=0.25,
        cls_allow_reverse=None,
        valid_vel_weight=-1,
    ):
        """Initialize SparseBox3DLoss.

        Args:
            box_weight (float): Weight for box loss.
            cls_allow_reverse (list): List of classes that allow reverse.
            valid_vel_weight (float): Weight for valid velocity.
        """
        super().__init__()

        self.loss_box = WeightedL1(
            loss_weight=box_weight,
            reduction='mean'
        )
        self.loss_cns = SigmoidCrossEntropy(reduction='mean')
        self.loss_yns = GaussianFocalLoss(alpha=2.0, gamma=4.0, loss_weight=1.0)
        self.cls_allow_reverse = cls_allow_reverse
        self.valid_vel_weight = valid_vel_weight

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        """Forward pass for SparseBox3DLoss.

        Args:
            box (torch.Tensor): Predicted box.
            box_target (torch.Tensor): Target box.
            weight (torch.Tensor): Weight for box loss.
            avg_factor (int): Average factor.
            suffix (str): Suffix for loss names.
            quality (torch.Tensor): Quality of the box.
            cls_target (torch.Tensor): Target class.
            **kwargs: Additional arguments.
        """
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (torch.nn.functional.cosine_similarity(box_target[..., [SIN_YAW, COS_YAW]], box[..., [SIN_YAW, COS_YAW]], dim=-1) < 0)
            if_reverse = (torch.isin(cls_target, cls_target.new_tensor(self.cls_allow_reverse)) & if_reverse)
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        if self.valid_vel_weight > 0:
            box_loss = self.loss_box(
                box[:, :8], box_target[:, :8], weight=weight[:, :8], avg_factor=avg_factor
            )
            vel_loss = self.loss_box(
                box[:, 8:], box_target[:, 8:], weight=weight[:, 8:], avg_factor=avg_factor
            )
            vel_weights = torch.norm(box[:, 8:], p=2, dim=-1) > 1e-3
            vel_weights = torch.where(vel_weights, torch.tensor(self.valid_vel_weight), torch.tensor(1.0))
            output[f"loss_box{suffix}"] = box_loss * vel_weights
            output[f"loss_box_vel{suffix}"] = vel_loss * vel_weights
        else:
            box_loss = self.loss_box(
                box, box_target, weight=weight, avg_factor=avg_factor
            )
            output[f"loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(box_target[..., [SIN_YAW, COS_YAW]], box[..., [SIN_YAW, COS_YAW]], dim=-1) > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target)
            output[f"loss_yns{suffix}"] = yns_loss

            if self.valid_vel_weight > 0:
                output[f"loss_cns{suffix}"] = cns_loss * vel_weights
                output[f"loss_yns{suffix}"] = yns_loss * vel_weights

        return output


def normalize(x, axis=-1):
    """Normalize a Tensor to unit length along the specified dimension.

    Args:
        x (torch.Tensor): The data to normalize.
        axis (int, optional): The axis along which to normalize. Defaults to -1.

    Returns:
        torch.Tensor: The normalized data.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """Compute the euclidean distance between two tensors.

    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The euclidean distance between x and y.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    """

    def __init__(self, num_ids=67, epsilon=0.1, use_gpu=True):
        """Initialize the CrossEntropyLabelSmooth class.

        Args:
            num_ids (int): Number of ids.
            epsilon (float, optional): Smoothing factor. Defaults to 0.1.
            use_gpu (bool, optional): Whether to use gpu for computation. Defaults to True.
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_ids = num_ids
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """Compute the loss based on inputs and targets.

        Args:
            inputs (torch.Tensor): Prediction matrix (before softmax) with shape (batch_size, num_ids).
            targets (torch.Tensor): Ground truth labels with shape (num_ids).

        Returns:
            list: Loss values.
        """
        # Ensure targets are of Long type as required by cross_entropy
        targets = targets.long()

        return F.cross_entropy(
            inputs, targets, label_smoothing=self.epsilon, reduction='mean'
        )


class WeightedL1(nn.Module):
    """Weighted L1 loss.

    This class implements a weighted L1 loss function.
    """

    def __init__(self, loss_weight=0.25, reduction='mean'):
        """Initialize WeightedL1.

        Args:
            loss_weight (float): Weight for L1 loss.
            reduction (str): Reduction method.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        """
        pred: (N, *), the predicted box
        target: (N, *), the ground-truth
        weight: optional weighting per-element, shape (N,) or broadcastable
        avg_factor: optional scalar for normalizing
        """
        if target.numel() == 0:
            return pred.sum() * 0

        assert pred.size() == target.size(), f"Incorrect shape of pred: {pred.size()} and target: {target.size()} in weighted L1 loss"
        loss = torch.abs(pred - target)
        if weight is not None:
            loss = loss * weight  # broadcast or elementwise

        # Sum or average
        if self.reduction == 'mean':
            loss = loss.sum() if avg_factor is None else loss.sum() / avg_factor
        elif self.reduction == 'sum':
            loss = loss.sum()

        # Multiply by the config weight
        return self.loss_weight * loss


class SigmoidCrossEntropy(nn.Module):
    """Sigmoid cross entropy loss.

    This class implements a sigmoid cross entropy loss function.
    """

    def __init__(self, reduction='mean'):
        """Initialize SigmoidCrossEntropy.

        Args:
            reduction (str): Reduction method.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, weight=None, avg_factor=None):
        """
        logits: (N, 1 or N, C) raw predicted scores
        targets: same shape, in [0,1]
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            loss = loss.sum() if avg_factor is None else loss.sum() / avg_factor
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
