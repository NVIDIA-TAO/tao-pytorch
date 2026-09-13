# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Criterion loss functions for Sparse4D."""

import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.target import SparseBox3DTarget
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.decoder import decode_box
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import (
    X,
    Y,
    Z,
    SIN_YAW,
    COS_YAW,
    CNS,
    YNS,
)
from nvidia_tao_pytorch.cv.sparse4d.model import loose_to_tight_loss as ltt_loss
from nvidia_tao_pytorch.cv.sparse4d.model import loose_to_tight_match as ltt_match
from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import LooseToTightMLP
from nvidia_tao_pytorch.cv.sparse4d.model.sv_aux_head import SVAuxClassifier


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across the initialized distributed group."""
    if not torch.distributed.is_available():
        return tensor
    if not torch.distributed.is_initialized():
        return tensor

    reduced = tensor.clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    return reduced / torch.distributed.get_world_size()


class SetCriterion(nn.Module):
    """SetCriterion class for Sparse4D.

    This class defines the criterion for the Sparse4D model.
    It computes the classification and regression losses for the model.
    """

    def __init__(
        self,
        model_config,
        instance_bank,
        class_names=None,
        sync_positive_counts=True,
    ):
        """Create the criterion.

        Args:
            model_config (dict): Model configuration.
            instance_bank (InstanceBank): Instance bank.
            class_names (Sequence[str], optional): Ordered detector taxonomy
                from ``dataset.classes``.
            sync_positive_counts (bool): Whether main and DN positive counts
                may use distributed collectives. This must be false when an
                active route-specific loss can make ranks leave the criterion
                before the 3D loss path.
        """
        super().__init__()
        self.sync_positive_counts = bool(sync_positive_counts)
        self.class_names = None if class_names is None else list(class_names)
        if self.class_names is not None:
            if not self.class_names:
                raise ValueError("dataset.classes must contain at least one class")
            if any(not isinstance(name, str) or not name for name in self.class_names):
                raise ValueError("dataset.classes must contain non-empty strings")
            if len(set(self.class_names)) != len(self.class_names):
                raise ValueError("dataset.classes must not contain duplicates")
        self.detector_num_classes = (
            len(self.class_names) if self.class_names is not None else None
        )
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
        self.loss_cls = FocalLoss(
            gamma=self.cls_loss_config["gamma"],
            alpha=self.cls_loss_config["alpha"],
            loss_weight=self.cls_loss_config["loss_weight"],
        )
        self.reg_loss_config = model_config["head"]["loss"]["reg"]
        self.loss_reg = SparseBox3DLoss(
            box_weight=self.reg_loss_config["box_weight"],
            valid_vel_weight=head_config["valid_vel_weight"],
        )
        self.id_loss_config = model_config["head"]["loss"]["id"]
        self.loss_id = CrossEntropyLabelSmooth(num_ids=self.id_loss_config["num_ids"])
        self.loss_visibility = nn.BCELoss()
        self.use_temporal_align = model_config["use_temporal_align"]
        self.num_single_frame_decoder = head_config["num_single_frame_decoder"]
        self.loss_depth = DenseDepthLoss()
        self.instance_bank = instance_bank

        # Loose-to-tight geometric 2D distillation. The frozen MLP deliberately
        # lives in a plain list so it is excluded from TAO checkpoints and DDP.
        ltt_cfg = head_config.get("loose_to_tight", {}) or {}
        self.ltt_enable = bool(ltt_cfg.get("enable", False))
        self.ltt_loss_weight = float(ltt_cfg.get("loss_weight", 0.1))
        configured_ltt_classes = int(ltt_cfg.get("num_classes", 0))
        if configured_ltt_classes < 0:
            raise ValueError("model.head.loose_to_tight.num_classes cannot be negative")
        self.ltt_num_classes = self.detector_num_classes or configured_ltt_classes or 7
        self.ltt_tight_l1_weight = float(ltt_cfg.get("tight_l1_weight", 1.0))
        self.ltt_containment_weight = float(ltt_cfg.get("containment_weight", 1.0))
        self.ltt_box2d_key = ltt_cfg.get("box2d_key", "gt_boxes_2d_visible")
        self.ltt_occ_key = ltt_cfg.get("occ_key", "gt_occ_weight")
        self.ltt_instance_id_key = ltt_cfg.get("instance_id_key", "instance_id")
        self.ltt_ego2cam_key = ltt_cfg.get("ego2cam_key", "cam2world_transform")
        self.ltt_min_gt_area = float(ltt_cfg.get("min_gt_area", 1.0))
        self.ltt_eps = float(ltt_cfg.get("eps", 0.1))
        self.ltt_pseudo_enable = bool(ltt_cfg.get("pseudo_enable", False))
        self.ltt_det_box_key = ltt_cfg.get("det_box_key", "det_boxes_2d")
        self.ltt_det_cls_key = ltt_cfg.get("det_cls_key", "det_classes_2d")
        self.ltt_det_score_key = ltt_cfg.get("det_score_key", "det_scores_2d")
        self.ltt_has3dgt_key = ltt_cfg.get("has_3d_gt_key", "has_3d_gt")
        self.ltt_giou_thr = float(ltt_cfg.get("giou_thr", 0.3))
        self.ltt_cost_giou = float(ltt_cfg.get("cost_giou", 2.0))
        self.ltt_cost_l1 = float(ltt_cfg.get("cost_l1", 1.0))
        self.ltt_cost_cls = float(ltt_cfg.get("cost_cls", 1.0))
        self.ltt_det_score_thr = float(ltt_cfg.get("det_score_thr", 0.0))
        self.ltt_min_cams = int(ltt_cfg.get("min_cams", 1))
        self.ltt_dedup_dist = float(ltt_cfg.get("dedup_dist", 0.0))
        self.ltt_class_gate = bool(ltt_cfg.get("class_gate", True))
        self.ltt_pseudo_box_weight = float(ltt_cfg.get("pseudo_box_weight", 0.1))
        self.ltt_pseudo_cls_weight = float(ltt_cfg.get("pseudo_cls_weight", 1.0))
        self.ltt_sv_depth_weight = float(ltt_cfg.get("sv_depth_weight", 0.0))
        self.ltt_sv_size_weight = float(ltt_cfg.get("sv_size_weight", 0.25))
        self.ltt_sv_yaw_weight = float(ltt_cfg.get("sv_yaw_weight", 0.0))
        self.sv_scene_keywords = list(model_config.get("sv_scene_keywords", []) or [])
        if (
            (self.ltt_enable or self.ltt_pseudo_enable) and
            self.detector_num_classes is not None and
            configured_ltt_classes not in (0, self.detector_num_classes)
        ):
            raise ValueError(
                "model.head.loose_to_tight.num_classes must be zero or match "
                f"dataset.classes: configured={configured_ltt_classes}, "
                f"dataset={self.detector_num_classes}"
            )
        self._ltt_mlp = [None]
        if self.ltt_enable:
            checkpoint = ltt_cfg.get("mlp_ckpt", "")
            if not checkpoint:
                raise FileNotFoundError(
                    "Loose-to-tight distillation is enabled, but model.head."
                    "loose_to_tight.mlp_ckpt is empty."
                )
            try:
                mlp = LooseToTightMLP.load(checkpoint, map_location="cpu", freeze=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loose-to-tight distillation requires a trained MLP checkpoint; "
                    f"not found: {checkpoint}"
                ) from exc
            expected_num_classes = (
                self.detector_num_classes or configured_ltt_classes or None
            )
            if (
                expected_num_classes is not None and
                mlp.num_classes != expected_num_classes
            ):
                raise ValueError(
                    "Loose-to-tight checkpoint class count does not match the "
                    f"detector taxonomy: expected={expected_num_classes}, "
                    f"checkpoint={mlp.num_classes}"
                )
            if self.class_names is not None:
                checkpoint_classes = mlp.class_names
                if checkpoint_classes is None:
                    raise ValueError(
                        "Loose-to-tight checkpoint is missing class_names metadata; "
                        f"expected dataset.classes={self.class_names!r}"
                    )
                if list(checkpoint_classes) != self.class_names:
                    raise ValueError(
                        "Loose-to-tight checkpoint class_names/order does not match "
                        f"dataset.classes: expected={self.class_names!r}, "
                        f"checkpoint={list(checkpoint_classes)!r}"
                    )
            self.ltt_num_classes = mlp.num_classes
            self._ltt_mlp = [mlp]

        # Calibration-free SV2D uses an image-plane auxiliary classifier owned by
        # the criterion. This keeps loss modules out of the prediction-only head.
        sv_cfg = model_config.get("sv_aux_head", {}) or {}
        self.sv_aux_enable = bool(sv_cfg.get("enable", False))
        configured_sv_classes = int(sv_cfg.get("num_classes", 0))
        if configured_sv_classes < 0:
            raise ValueError("model.sv_aux_head.num_classes cannot be negative")
        reference_num_classes = self.detector_num_classes or (
            self.ltt_num_classes if self.ltt_enable else None
        )
        if (
            self.sv_aux_enable and
            reference_num_classes is not None and
            configured_sv_classes not in (0, reference_num_classes)
        ):
            raise ValueError(
                "model.sv_aux_head.num_classes must be zero or match "
                f"dataset.classes: configured={configured_sv_classes}, "
                f"dataset={reference_num_classes}"
            )
        self.sv_aux_num_classes = reference_num_classes or configured_sv_classes
        self.sv_aux_head = None
        if self.sv_aux_enable:
            if not self.sv_aux_num_classes:
                raise ValueError(
                    "model.sv_aux_head.num_classes is zero, but dataset.classes "
                    "was not provided to derive the detector taxonomy"
                )
            self.sv_aux_head = SVAuxClassifier(
                in_channels=int(sv_cfg.get("in_channels", 256)),
                num_classes=self.sv_aux_num_classes,
                roi_size=int(sv_cfg.get("roi_size", 7)),
                hidden_dim=int(sv_cfg.get("hidden_dim", 256)),
                fpn_strides=tuple(sv_cfg.get("fpn_strides", [4, 8, 16, 32])),
                use_level=int(sv_cfg.get("use_level", 1)),
                loss_weight=float(sv_cfg.get("loss_weight", 1.0)),
                det_box_key=sv_cfg.get("det_box_key", "det_boxes_2d"),
                det_cls_key=sv_cfg.get("det_cls_key", "det_classes_2d"),
                min_box_size=float(sv_cfg.get("min_box_size", 2.0)),
            )

    def _positive_count(self, count):
        """Globally average a count only when every rank reaches this path."""
        if self.sync_positive_counts:
            count = reduce_mean(count)
        return count.clamp_min(1.0)

    def _synchronized_dn_positive_counts(self, model_outs, reference):
        """Reduce regular and temporal DN counts with fixed participation.

        DN tensors can be absent on a rank with no local ground truth. Packing
        both counts into one collective before the DN early return ensures that
        every baseline rank participates exactly once.
        """
        if not self.sync_positive_counts:
            return {}

        prefixes = ("", "temp_")
        local_counts = []
        for prefix in prefixes:
            valid_mask = model_outs.get(f"{prefix}dn_valid_mask")
            count = reference.new_zeros(())
            if valid_mask is not None:
                count = valid_mask.sum().to(
                    device=reference.device, dtype=reference.dtype
                )
            local_counts.append(count)

        mean_counts = reduce_mean(torch.stack(local_counts))
        return {
            prefix: count.clamp_min(1.0) for prefix, count in zip(prefixes, mean_counts)
        }

    def forward(self, raw_model_outs, data, feature_maps=None):
        """Computes loss."""
        # ===================== prediction losses ======================
        if len(raw_model_outs) == 3:
            model_outs, depths, raw_feature_maps = raw_model_outs
        else:
            model_outs, depths = raw_model_outs
            raw_feature_maps = feature_maps
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        loss_zero = self._prediction_zero(cls_scores, reg_preds, depths)

        reid_features = (
            model_outs["reid_feature"]
            if self.use_reid_sampling
            else [None] * len(cls_scores)
        )
        pred_visibility_scores = (
            model_outs["visibility_scores"]
            if self.use_reid_sampling
            else [None] * len(cls_scores)
        )
        predicted_ids = (
            model_outs["predicted_id"]
            if self.use_reid_sampling
            else [None] * len(cls_scores)
        )

        # Real frames have no 3D labels. Supervise their projected queries with
        # per-camera RT-DETR detections, or use the calibration-free SV ROI head.
        real_batch = self._is_real_batch(data)
        sv_batch = real_batch and self._is_sv_batch(data)
        if real_batch and (
            (self.ltt_enable and self.ltt_pseudo_enable) or
            (sv_batch and self.sv_aux_head is not None)
        ):
            output = {}
            if self.ltt_enable and self.ltt_pseudo_enable:
                for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
                    output.update(
                        self._loss_2d_pseudo(
                            decoder_idx,
                            reg[..., : len(self.reg_weights)],
                            cls,
                            data,
                        )
                    )
            else:
                for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
                    output[f"loss_sv_head_{decoder_idx}"] = (
                        cls.sum() + reg.sum()
                    ) * 0.0

            if sv_batch and self.sv_aux_head is not None:
                output = {key: value * 0.0 for key, value in output.items()}
                if raw_feature_maps is None:
                    raise RuntimeError(
                        "SV auxiliary supervision is enabled, but Sparse4D did not "
                        "return raw FPN feature maps."
                    )
                output.update(self.sv_aux_head.loss(raw_feature_maps, data))

            self._add_dense_depth_loss(
                output,
                depths,
                data.get("gt_depth"),
                skip=sv_batch,
            )
            return self._finite_losses(output, loss_zero)

        if self.use_temporal_align:
            gt_index_mapping_prev = (
                self.instance_bank.get_gt_index_mapping()
            )  # use the same mapping for different decoders
        else:
            gt_index_mapping_prev = None

        output = {}
        gt_index_mapping_curr = gt_index_mapping_prev
        for decoder_idx, (
            cls,
            reg,
            qt,
            _,
            pred_visibility_score,
            predicted_id,
        ) in enumerate(
            zip(
                cls_scores,
                reg_preds,
                quality,
                reid_features,
                pred_visibility_scores,
                predicted_ids,
            )
        ):
            if self.use_temporal_align:
                if gt_index_mapping_prev is None:  # first frame: Hungarian only
                    use_hungarian_only = True
                    update_gt_indices = False
                    update_gt_index_mapping = True
                else:  # other frames
                    if (
                        decoder_idx + 1 == self.num_single_frame_decoder
                    ):  # first decoder
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
            (
                cls_target,
                reg_target,
                reg_weights,
                instance_id_target,
                asset_id_target,
                visibility_score_target,
                gt_index_mapping_curr,
            ) = self.sampler.sample(
                cls,
                reg,
                data["gt_labels_3d"],
                data["gt_bboxes_3d"],
                data.get("instance_id"),
                data.get("asset_id"),
                data["gt_visibility"] if "gt_visibility" in data else None,
                gt_index_mapping_curr,
                use_hungarian_only=use_hungarian_only,
                update_gt_indices=update_gt_indices,
                update_gt_index_mapping=update_gt_index_mapping,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = self._positive_count(torch.sum(mask).to(dtype=reg.dtype))

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            if self.ltt_enable:
                output.update(
                    self._loss_box_2d(
                        decoder_idx,
                        reg,
                        cls,
                        mask,
                        instance_id_target,
                        data,
                        num_pos,
                    )
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_input = cls.clamp(min=-50.0, max=50.0)
            cls_loss = self.loss_cls(cls_input, cls_target, avg_factor=num_pos)
            if not bool(torch.isfinite(cls_loss).all()):
                warnings.warn(
                    f"loss_cls_{decoder_idx} is non-finite; skipping its gradient",
                    RuntimeWarning,
                )
                cls_loss = (
                    torch.nan_to_num(cls_input, nan=0.0, posinf=0.0, neginf=0.0).sum() *
                    0.0
                )

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
                if predicted_id is None:
                    id_loss = cls.sum() * 0.0
                else:
                    predicted_id = predicted_id.reshape(-1, predicted_id.shape[-1])
                    if asset_id_target is None:
                        id_loss = predicted_id.sum() * 0.0
                    else:
                        asset_ids = asset_id_target.reshape(-1)
                        valid_id = (asset_ids >= 0) & (
                            asset_ids < predicted_id.shape[-1]
                        )
                        if bool(valid_id.any()):
                            id_loss = self.loss_id(
                                predicted_id[valid_id], asset_ids[valid_id]
                            )
                        else:
                            id_loss = predicted_id.sum() * 0.0

                output[f"loss_id_{decoder_idx}"] = id_loss

                if pred_visibility_score is None:
                    visibility_loss = cls.sum() * 0.0
                elif visibility_score_target is None:
                    visibility_loss = pred_visibility_score.sum() * 0.0
                else:
                    valid_visibility = (
                        torch.isfinite(pred_visibility_score) &
                        torch.isfinite(visibility_score_target) &
                        (visibility_score_target >= 0) &
                        (visibility_score_target <= 1)
                    )
                    if bool(valid_visibility.any()):
                        visibility_loss = self.loss_visibility(
                            pred_visibility_score[valid_visibility],
                            visibility_score_target[valid_visibility].to(
                                pred_visibility_score.dtype
                            ),
                        )
                    else:
                        visibility_loss = pred_visibility_score.sum() * 0.0
                output[f"loss_visibility_{decoder_idx}"] = visibility_loss

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if self.use_temporal_align:
            self.instance_bank.cache_gt_index_mapping(gt_index_mapping_curr)
            query_indices = self.instance_bank.get_cached_query_indices()
            self.instance_bank.update_query_indices_in_cached_gt_index_mapping(
                query_indices
            )
        dn_positive_counts = self._synchronized_dn_positive_counts(
            model_outs, reg_preds[0]
        )

        if "dn_prediction" not in model_outs:
            zero = cls_scores[0].sum() * 0.0
            has_velocity_loss = self.loss_reg.valid_vel_weight > 0
            for decoder_idx in range(len(cls_scores)):
                output[f"loss_cls_dn_{decoder_idx}"] = zero
                output[f"loss_box_dn_{decoder_idx}"] = zero
                if has_velocity_loss:
                    output[f"loss_box_vel_dn_{decoder_idx}"] = zero
            self._add_dense_depth_loss(output, depths, data.get("gt_depth"))
            return self._finite_losses(output, loss_zero)

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
        ) = self.prepare_for_dn_loss(
            model_outs,
            avg_factor=dn_positive_counts.get(""),
        )
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            if (
                "temp_dn_valid_mask" in model_outs and
                decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(
                    model_outs,
                    prefix="temp_",
                    avg_factor=dn_positive_counts.get("temp_"),
                )

            dn_cls_input = cls.flatten(end_dim=1)[dn_valid_mask].clamp(
                min=-50.0, max=50.0
            )
            cls_loss = self.loss_cls(
                dn_cls_input,
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            if not bool(torch.isfinite(cls_loss).all()):
                warnings.warn(
                    f"loss_cls_dn_{decoder_idx} is non-finite; skipping its gradient",
                    RuntimeWarning,
                )
                cls_loss = (
                    torch.nan_to_num(
                        dn_cls_input, nan=0.0, posinf=0.0, neginf=0.0
                    ).sum() *
                    0.0
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
        self._add_dense_depth_loss(output, depths, data.get("gt_depth"))
        return self._finite_losses(output, loss_zero)

    @staticmethod
    def _per_sample(value, batch_index):
        """Select one sample from a list or batched tensor."""
        return value[batch_index]

    @staticmethod
    def _to_device(value, device, dtype=None):
        """Convert tensor-like side-cache data without breaking existing tensors."""
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

    def _is_real_batch(self, data):
        """Return whether every sample is explicitly marked as lacking 3D GT."""
        values = data.get(self.ltt_has3dgt_key)
        if values is None:
            metas = data.get("img_metas")
            if isinstance(metas, (list, tuple)):
                values = [meta.get(self.ltt_has3dgt_key) for meta in metas]
        if values is None:
            return False
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().reshape(-1).tolist()
        elif not isinstance(values, (list, tuple)):
            values = [values]
        return bool(values) and all(
            value is not None and not bool(value) for value in values
        )

    def _add_dense_depth_loss(self, output, depth_preds, gt_depths, skip=False):
        """Add dense-depth supervision or a graph-connected route-safe zero."""
        if depth_preds is None or gt_depths is None:
            return
        if skip:
            # Calibration-free SV2D samples have no metric camera/depth
            # contract. Keep the depth branch in the graph without treating
            # any placeholder values as supervision.
            output["loss_dense_depth"] = self._prediction_zero([], [], depth_preds)
            return
        output["loss_dense_depth"] = self.loss_depth(depth_preds, gt_depths)

    def _is_sv_batch(self, data):
        """Return whether all samples belong to a calibration-free SV2D scene."""
        if not self.sv_scene_keywords:
            return False
        names = data.get("scene_name")
        if names is None:
            metas = data.get("img_metas")
            if isinstance(metas, (list, tuple)):
                names = [meta.get("scene_name") for meta in metas]
        if names is None:
            return False
        if isinstance(names, str):
            names = [names]
        return bool(names) and all(
            name is not None and
            any(keyword in str(name) for keyword in self.sv_scene_keywords)
            for name in names
        )

    @staticmethod
    def _scale_gradient(value, weight):
        """Scale only the backward gradient while preserving the forward value."""
        if weight >= 1.0:
            return value
        detached = value.detach()
        if weight <= 0.0:
            return detached
        return detached + weight * (value - detached)

    def _validate_ltt_prediction_classes(self, cls):
        """Require detector logits and the frozen LTT taxonomy to agree."""
        prediction_classes = int(cls.shape[-1])
        if (
            self.detector_num_classes is not None and
            prediction_classes != self.detector_num_classes
        ):
            raise ValueError(
                "Sparse4D classification logits do not match dataset.classes: "
                f"logits={prediction_classes}, dataset={self.detector_num_classes}"
            )
        if prediction_classes != self.ltt_num_classes:
            raise ValueError(
                "Sparse4D classification logits do not match the loose-to-tight "
                f"checkpoint: logits={prediction_classes}, "
                f"checkpoint={self.ltt_num_classes}"
            )

    def _sv_mask_boxes(self, boxes7, ego2cam):
        """Mask unobservable depth/yaw gradients for a virtual single camera."""
        rotation = ego2cam[:3, :3]
        translation = ego2cam[:3, 3]
        center_cam = boxes7[..., :3] @ rotation.transpose(-1, -2) + translation
        center_cam = torch.cat(
            [
                center_cam[..., :2],
                self._scale_gradient(center_cam[..., 2:3], self.ltt_sv_depth_weight),
            ],
            dim=-1,
        )
        center = (center_cam - translation) @ rotation
        extent = self._scale_gradient(boxes7[..., 3:6], self.ltt_sv_size_weight)
        yaw = self._scale_gradient(boxes7[..., 6:7], self.ltt_sv_yaw_weight)
        return torch.cat([center, extent, yaw, boxes7[..., 7:]], dim=-1)

    def _loss_box_2d(
        self,
        decoder_idx,
        reg,
        cls,
        mask,
        instance_id_target,
        data,
        num_pos,
    ):
        """Occlusion-aware LTT loss for synthetic samples with matched 3D GT."""
        key = f"loss_box_2d_{decoder_idx}"
        zero = {key: reg.sum() * 0.0}
        mlp = self._ltt_mlp[0]
        if mlp is None or instance_id_target is None:
            return zero
        self._validate_ltt_prediction_classes(cls)

        box2d = data.get(self.ltt_box2d_key)
        occ = data.get(self.ltt_occ_key)
        gt_instance_ids = data.get(self.ltt_instance_id_key)
        projection = data.get("projection_mat")
        image_wh = data.get("image_wh")
        ego2cam = data.get(self.ltt_ego2cam_key)
        if any(
            value is None
            for value in (
                box2d,
                occ,
                gt_instance_ids,
                projection,
                image_wh,
                ego2cam,
            )
        ):
            return zero

        device = reg.device
        mlp = mlp.to(device)
        total = reg.new_zeros(())
        for batch_index in range(reg.shape[0]):
            positive = mask[batch_index]
            if not bool(positive.any()):
                continue

            boxes7 = decode_box(reg[batch_index][positive].float())[:, :7]
            class_id = cls[batch_index][positive].argmax(dim=-1)
            positive_instance_ids = instance_id_target[batch_index][positive]
            gt_ids = self._to_device(
                self._per_sample(gt_instance_ids, batch_index),
                device,
                torch.long,
            ).reshape(-1)
            visible = self._to_device(
                self._per_sample(box2d, batch_index), device, torch.float32
            )
            occ_weight = self._to_device(
                self._per_sample(occ, batch_index), device, torch.float32
            )
            if gt_ids.numel() == 0:
                continue

            matches = positive_instance_ids[:, None] == gt_ids[None, :]
            found = matches.any(dim=-1)
            if not bool(found.any()):
                continue
            rows = matches.to(torch.int64).argmax(dim=-1)[found]
            boxes7 = boxes7[found]
            class_id = class_id[found]
            visible = visible[rows]
            occ_weight = occ_weight[rows]

            projection_b = self._to_device(
                self._per_sample(projection, batch_index), device, torch.float32
            )
            image_wh_b = self._to_device(
                self._per_sample(image_wh, batch_index), device, torch.float32
            )
            ego2cam_b = self._to_device(
                self._per_sample(ego2cam, batch_index), device, torch.float32
            )
            for camera_index in range(projection_b.shape[0]):
                loose, features, valid = ltt_loss.loose_and_features(
                    boxes7,
                    class_id,
                    projection_b[camera_index],
                    ego2cam_b[camera_index],
                    image_wh_b[camera_index],
                    self.ltt_num_classes,
                    eps=self.ltt_eps,
                )
                target = visible[:, camera_index]
                area = (target[:, 2] - target[:, 0]).clamp_min(0) * (
                    target[:, 3] - target[:, 1]
                ).clamp_min(0)
                keep = valid & (area > self.ltt_min_gt_area)
                if not bool(keep.any()):
                    continue
                scales = mlp(features[keep].float()).detach().to(loose.dtype)
                prediction = LooseToTightMLP.apply_to_loose(loose[keep], scales)
                diagonal = torch.linalg.vector_norm(image_wh_b[camera_index]).expand(
                    int(keep.sum())
                )
                total = total + ltt_loss.loose_to_tight_2d_loss(
                    prediction,
                    target[keep],
                    occ_weight[keep, camera_index],
                    diagonal,
                    tight_l1_weight=self.ltt_tight_l1_weight,
                    containment_weight=self.ltt_containment_weight,
                )
        return {key: self.ltt_loss_weight * total / num_pos}

    def _loss_2d_pseudo(self, decoder_idx, reg, cls, data):
        """LTT projection and Hungarian pseudo-label losses for real frames."""
        box_key = f"loss_box_2d_pseudo_{decoder_idx}"
        cls_key = f"loss_cls_pseudo_{decoder_idx}"
        output = {
            box_key: torch.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0,
            cls_key: torch.nan_to_num(cls, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0,
        }
        mlp = self._ltt_mlp[0]
        if mlp is None:
            return output
        self._validate_ltt_prediction_classes(cls)

        valid_samples = self._pseudo_valid_samples(data, reg.shape[0])
        if not any(valid_samples):
            return output

        det_boxes = data.get(self.ltt_det_box_key)
        det_classes = data.get(self.ltt_det_cls_key)
        det_scores = data.get(self.ltt_det_score_key)
        projection = data.get("projection_mat")
        image_wh = data.get("image_wh")
        ego2cam = data.get(self.ltt_ego2cam_key)
        if any(
            value is None
            for value in (
                det_boxes,
                det_classes,
                det_scores,
                projection,
                image_wh,
                ego2cam,
            )
        ):
            return output

        device = reg.device
        mlp = mlp.to(device)
        is_sv = self._is_sv_batch(data)
        background = cls.shape[-1]
        box_total = output[box_key]
        box_count = 0
        cls_logits = []
        cls_targets = []

        for batch_index in range(reg.shape[0]):
            if not valid_samples[batch_index]:
                continue
            boxes7 = decode_box(reg[batch_index].float())[:, :7]
            query_probability = cls[batch_index].sigmoid()
            query_class = query_probability.argmax(dim=-1)
            projection_b = self._to_device(
                self._per_sample(projection, batch_index), device, torch.float32
            )
            image_wh_b = self._to_device(
                self._per_sample(image_wh, batch_index), device, torch.float32
            )
            ego2cam_b = self._to_device(
                self._per_sample(ego2cam, batch_index), device, torch.float32
            )
            boxes_by_camera = self._per_sample(det_boxes, batch_index)
            classes_by_camera = self._per_sample(det_classes, batch_index)
            scores_by_camera = self._per_sample(det_scores, batch_index)
            amodal_by_camera = []
            matches_by_camera = []

            for camera_index in range(projection_b.shape[0]):
                projected_boxes = (
                    self._sv_mask_boxes(boxes7, ego2cam_b[camera_index])
                    if is_sv
                    else boxes7
                )
                loose, features, valid = ltt_loss.loose_and_features(
                    projected_boxes,
                    query_class,
                    projection_b[camera_index],
                    ego2cam_b[camera_index],
                    image_wh_b[camera_index],
                    self.ltt_num_classes,
                    eps=self.ltt_eps,
                )
                scales = mlp(features.float()).detach().to(loose.dtype)
                amodal = LooseToTightMLP.apply_to_loose(loose, scales)
                amodal_by_camera.append(amodal)

                boxes = self._to_device(
                    boxes_by_camera[camera_index], device, torch.float32
                ).reshape(-1, 4)
                classes = self._to_device(
                    classes_by_camera[camera_index], device, torch.long
                ).reshape(-1)
                scores = self._to_device(
                    scores_by_camera[camera_index], device, torch.float32
                ).reshape(-1)
                invalid_classes = torch.logical_or(classes < 0, classes >= background)
                if bool(invalid_classes.any()):
                    invalid_ids = torch.unique(classes[invalid_classes]).tolist()
                    taxonomy = (
                        self.class_names
                        if self.class_names is not None
                        else f"{background} detector classes"
                    )
                    raise ValueError(
                        "Pseudo-label class IDs are outside the detector taxonomy "
                        f"at batch={batch_index}, cam={camera_index}: "
                        f"invalid={invalid_ids}, taxonomy={taxonomy!r}"
                    )
                valid_detection = torch.isfinite(boxes).all(dim=-1) & torch.isfinite(
                    scores
                )
                boxes = boxes[valid_detection]
                classes = classes[valid_detection]
                scores = scores[valid_detection]
                diagonal = float(torch.linalg.vector_norm(image_wh_b[camera_index]))
                matches_by_camera.append(
                    ltt_match.match_camera(
                        amodal,
                        valid,
                        query_probability,
                        boxes,
                        classes,
                        scores,
                        diagonal,
                        giou_thr=self.ltt_giou_thr,
                        w_giou=self.ltt_cost_giou,
                        w_l1=self.ltt_cost_l1,
                        w_cls=self.ltt_cost_cls,
                        score_thr=self.ltt_det_score_thr,
                        class_gate=self.ltt_class_gate,
                    )
                )

            keep_pairs, class_target, _ = ltt_match.aggregate_consistency(
                matches_by_camera,
                boxes7.shape[0],
                boxes7=boxes7,
                min_cams=self.ltt_min_cams,
                dedup_dist=self.ltt_dedup_dist,
                device=device,
            )
            for query_index, camera_index, target_box, score in keep_pairs:
                prediction = amodal_by_camera[camera_index][query_index].view(1, 4)
                target = target_box.to(device=device, dtype=torch.float32).view(1, 4)
                confidence = prediction.new_tensor([score])
                diagonal = torch.linalg.vector_norm(image_wh_b[camera_index]).view(1)
                box_total = box_total + ltt_loss.loose_to_tight_2d_loss(
                    prediction,
                    target,
                    confidence,
                    diagonal,
                    tight_l1_weight=self.ltt_tight_l1_weight,
                    containment_weight=self.ltt_containment_weight,
                )
                box_count += 1

            cls_logits.append(cls[batch_index])
            cls_targets.append(
                torch.where(
                    class_target >= 0,
                    class_target,
                    torch.full_like(class_target, background),
                )
            )

        logits = torch.cat(cls_logits, dim=0).clamp(min=-50.0, max=50.0)
        targets = torch.cat(cls_targets, dim=0)
        num_positive = (targets < background).sum().to(logits.dtype).clamp(min=1.0)
        classification_loss = self.loss_cls(logits, targets, avg_factor=num_positive)
        if not bool(torch.isfinite(classification_loss).all()):
            warnings.warn(
                f"{cls_key} is non-finite; skipping its gradient", RuntimeWarning
            )
            classification_loss = (
                torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            )
        output[box_key] = self.ltt_pseudo_box_weight * box_total / max(box_count, 1)
        output[cls_key] = self.ltt_pseudo_cls_weight * classification_loss
        return output

    @staticmethod
    def _pseudo_valid_samples(data, batch_size):
        """Resolve per-sample cache validity, preserving legacy callers."""
        values = data.get("has_2d_pseudo")
        if values is None:
            return [True] * batch_size
        if isinstance(values, torch.Tensor):
            resolved = values.detach().reshape(-1).cpu().tolist()
        elif isinstance(values, np.ndarray):
            resolved = values.reshape(-1).tolist()
        elif isinstance(values, (list, tuple)):
            resolved = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise ValueError(
                            "has_2d_pseudo entries must be scalar booleans"
                        )
                    value = value.detach().cpu().item()
                resolved.append(value)
        else:
            resolved = [values]
        if len(resolved) == 1 and batch_size != 1:
            resolved *= batch_size
        if len(resolved) != batch_size:
            raise ValueError(
                "has_2d_pseudo must contain one value per batch sample; "
                f"got {len(resolved)} for batch size {batch_size}"
            )
        return [bool(value) for value in resolved]

    @staticmethod
    def _prediction_zero(cls_scores, reg_preds, depths=None):
        """Build a backward-safe zero from tensors upstream of all loss maths."""
        tensors = list(cls_scores) + list(reg_preds)
        if torch.is_tensor(depths):
            tensors.append(depths)
        elif depths is not None:
            tensors.extend(value for value in depths if torch.is_tensor(value))
        if not tensors:
            return torch.zeros(())
        return sum(
            torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            for tensor in tensors
        )

    @staticmethod
    def _finite_losses(output, zero_ref=None):
        """Replace residual non-finite values with a pre-loss graph zero.

        Multiplying an already-invalid loss by zero is not backward-safe: its
        Jacobian can still contain NaN. ``zero_ref`` must therefore come from a
        finite, upstream prediction tensor rather than the failed loss operation.
        """
        for key, value in list(output.items()):
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                warnings.warn(
                    f"{key} is non-finite; zeroing this batch", RuntimeWarning
                )
                output[key] = (
                    zero_ref if zero_ref is not None else value.detach().new_zeros(())
                )
        return output

    def prepare_for_dn_loss(self, model_outs, prefix="", avg_factor=None):
        """Prepare for denoising loss."""
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        local_count = torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)
        # Never issue a collective from this data-dependent helper. The forward
        # path supplies its fixed-participation global factor when that is safe;
        # mixed-route training falls back to this rank-local denominator.
        num_dn_pos = (
            local_count
            if avg_factor is None
            else avg_factor.to(device=local_count.device, dtype=local_count.dtype)
        ).clamp_min(1.0)
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

    def __init__(
        self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0
    ):
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

    def forward(self, pred, target, weight=None, avg_factor=None):
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
        # Sparse4D follows the MMDetection sigmoid-focal convention: target C is
        # background (an all-zero one-hot row), while negative and >C labels are
        # ignored. The previous TAO implementation dropped every background query.
        valid_mask = (target >= 0) & (target <= num_classes)
        if not bool(valid_mask.any()):
            return pred.sum() * 0.0

        # Filter pred, target, and weight by this mask
        pred = pred[valid_mask]
        valid_target = target[valid_mask].long()
        if weight is not None:
            weight = weight[valid_mask]

        one_hot_target = pred.new_zeros((pred.shape[0], num_classes))
        foreground = valid_target < num_classes
        if bool(foreground.any()):
            one_hot_target[foreground] = F.one_hot(
                valid_target[foreground], num_classes=num_classes
            ).to(dtype=pred.dtype)

        log_pt = F.binary_cross_entropy_with_logits(
            pred, one_hot_target, reduction="none"  # <-- use the one-hot
        )
        p = torch.sigmoid(pred)
        pt = p * one_hot_target + (1 - p) * (1 - one_hot_target)

        focal_weight = (
            self.alpha * one_hot_target + (1 - self.alpha) * (1 - one_hot_target)
        ) * ((1 - pt) ** self.gamma)

        loss = focal_weight * log_pt

        # Apply per-sample weighting if provided (e.g., for imbalance in data).
        if weight is not None:
            weight = weight.float()
            if weight.ndim == 1:
                weight = weight[:, None]
            loss = loss * weight

        # Handle reduction
        if self.reduction == "mean":
            # If the user provides avg_factor, we divide the sum by avg_factor
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif self.reduction == "sum":
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
                torch.logical_and(torch.isfinite(gt), gt > 0.0),
                torch.isfinite(pred),
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]

            # Clip predicted depth to valid range
            pred = torch.clip(pred, 0.0, self.max_depth)

            # Calculate L1 loss with full precision
            error = torch.abs(pred - gt).sum()
            _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight

            loss = loss + _loss

        return loss


class GaussianFocalLoss(nn.Module):
    """
    A simplified version of Gaussian Focal Loss from the CornerNet/CenterNet family.
    Typically used for heatmap-based object center detection.
    Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction="mean", loss_weight=1.0):
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

        if self.reduction == "mean":
            loss = (pos_loss + neg_loss).mean()
        elif self.reduction == "sum":
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

        self.loss_box = WeightedL1(loss_weight=box_weight, reduction="mean")
        self.loss_cns = SigmoidCrossEntropy(reduction="mean")
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
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                ) <
                0
            )
            if_reverse = (
                torch.isin(cls_target, cls_target.new_tensor(self.cls_allow_reverse)) &
                if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        if self.valid_vel_weight > 0:
            box_loss = self.loss_box(
                box[:, :8],
                box_target[:, :8],
                weight=weight[:, :8],
                avg_factor=avg_factor,
            )
            vel_loss = self.loss_box(
                box[:, 8:],
                box_target[:, 8:],
                weight=weight[:, 8:],
                avg_factor=avg_factor,
            )
            vel_weights = torch.norm(box[:, 8:], p=2, dim=-1) > 1e-3
            vel_weights = torch.where(
                vel_weights, torch.tensor(self.valid_vel_weight), torch.tensor(1.0)
            )
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
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                ) >
                0
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
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
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
            inputs, targets, label_smoothing=self.epsilon, reduction="mean"
        )


class WeightedL1(nn.Module):
    """Weighted L1 loss.

    This class implements a weighted L1 loss function.
    """

    def __init__(self, loss_weight=0.25, reduction="mean"):
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

        assert (
            pred.size() == target.size()
        ), f"Incorrect shape of pred: {pred.size()} and target: {target.size()} in weighted L1 loss"
        loss = torch.abs(pred - target)
        if weight is not None:
            loss = loss * weight  # broadcast or elementwise

        # Sum or average
        if self.reduction == "mean":
            loss = loss.sum() if avg_factor is None else loss.sum() / avg_factor
        elif self.reduction == "sum":
            loss = loss.sum()

        # Multiply by the config weight
        return self.loss_weight * loss


class SigmoidCrossEntropy(nn.Module):
    """Sigmoid cross entropy loss.

    This class implements a sigmoid cross entropy loss function.
    """

    def __init__(self, reduction="mean"):
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
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            loss = loss.sum() if avg_factor is None else loss.sum() / avg_factor
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
