# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration-free single-view auxiliary classifier for Sparse4D."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class SVAuxClassifier(nn.Module):
    """Classify SV2D boxes from a selected pre-format FPN feature level.

    Sparse4D's geometric head requires calibrated cameras. This auxiliary head
    instead applies RoIAlign directly to image-plane FPN features, allowing
    calibration-free single-view examples to update the shared backbone and FPN.

    Args:
        in_channels: Number of channels in the selected FPN level.
        num_classes: Number of output classes.
        roi_size: Spatial output size used by RoIAlign.
        hidden_dim: Width of the classifier's hidden layer.
        fpn_strides: Input-pixel stride for each FPN level.
        use_level: Index of the FPN level to pool from.
        loss_weight: Scale applied to the classification loss.
        det_box_key: Batch key for per-camera boxes in input-pixel ``xyxy``.
        det_cls_key: Batch key for per-camera integer class labels.
        min_box_size: Minimum box width and height in input pixels.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        roi_size: int = 7,
        hidden_dim: int = 256,
        fpn_strides: Sequence[int] = (4, 8, 16, 32),
        use_level: int = 1,
        loss_weight: float = 1.0,
        det_box_key: str = "det_boxes_2d",
        det_cls_key: str = "det_classes_2d",
        min_box_size: float = 2.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or roi_size <= 0 or hidden_dim <= 0:
            raise ValueError("SV auxiliary head dimensions must be positive")
        if not fpn_strides:
            raise ValueError("fpn_strides must contain at least one level")
        if use_level < 0 or use_level >= len(fpn_strides):
            raise ValueError(
                f"use_level must index fpn_strides, got {use_level} for "
                f"{len(fpn_strides)} levels"
            )
        if fpn_strides[use_level] <= 0:
            raise ValueError("the selected FPN stride must be positive")
        if min_box_size < 0:
            raise ValueError("min_box_size must be non-negative")

        self.num_classes = int(num_classes)
        self.roi_size = int(roi_size)
        self.use_level = int(use_level)
        self.spatial_scale = 1.0 / float(fpn_strides[self.use_level])
        self.loss_weight = float(loss_weight)
        self.det_box_key = det_box_key
        self.det_cls_key = det_cls_key
        self.min_box_size = float(min_box_size)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * self.roi_size * self.roi_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.loss_cls = nn.CrossEntropyLoss()

    @staticmethod
    def _camera_items(value, minimum_stacked_dims: int) -> List:
        """Normalize one sample's fixed- or variable-length camera values."""
        if isinstance(value, (list, tuple)):
            return list(value)
        ndim = getattr(value, "ndim", None)
        if ndim is not None and ndim >= minimum_stacked_dims:
            return [value[index] for index in range(len(value))]
        return [value]

    @staticmethod
    def _as_tensor(value, *, dtype, device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _gather_rois(
        self,
        det_boxes,
        det_cls,
        batch_size: int,
        num_cams: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build RoIAlign rows ``[batch_camera, x1, y1, x2, y2]`` and labels."""
        rois = []
        labels = []
        for batch_index in range(batch_size):
            sample_boxes = self._camera_items(det_boxes[batch_index], 3)
            sample_classes = self._camera_items(det_cls[batch_index], 2)
            camera_count = min(num_cams, len(sample_boxes), len(sample_classes))
            for camera_index in range(camera_count):
                boxes = self._as_tensor(
                    sample_boxes[camera_index], dtype=torch.float32, device=device
                ).reshape(-1, 4)
                classes = self._as_tensor(
                    sample_classes[camera_index], dtype=torch.long, device=device
                ).reshape(-1)
                if boxes.shape[0] != classes.shape[0]:
                    raise ValueError(
                        "SV boxes/classes length mismatch at "
                        f"batch={batch_index}, cam={camera_index}: "
                        f"{boxes.shape[0]} vs {classes.shape[0]}"
                    )
                if boxes.numel() == 0:
                    continue

                invalid_classes = torch.logical_or(
                    classes < 0, classes >= self.num_classes
                )
                if bool(invalid_classes.any()):
                    invalid_ids = torch.unique(classes[invalid_classes]).tolist()
                    raise ValueError(
                        "SV auxiliary pseudo-label class IDs are outside the "
                        f"configured taxonomy [0, {self.num_classes - 1}] at "
                        f"batch={batch_index}, cam={camera_index}: {invalid_ids}"
                    )

                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                keep = (
                    torch.isfinite(boxes).all(dim=1) &
                    (widths >= self.min_box_size) &
                    (heights >= self.min_box_size)
                )
                if not bool(keep.any()):
                    continue

                boxes = boxes[keep]
                classes = classes[keep]
                roi_batch_index = boxes.new_full(
                    (boxes.shape[0], 1), float(batch_index * num_cams + camera_index)
                )
                rois.append(torch.cat((roi_batch_index, boxes), dim=1))
                labels.append(classes)

        if not rois:
            return None, None
        return torch.cat(rois, dim=0), torch.cat(labels, dim=0)

    def loss(
        self, fpn_list: Sequence[torch.Tensor], data: Dict
    ) -> Dict[str, torch.Tensor]:
        """Return the weighted SV auxiliary classification loss."""
        if self.use_level >= len(fpn_list):
            raise ValueError(
                f"SV auxiliary head requested FPN level {self.use_level}, but only "
                f"{len(fpn_list)} levels were provided"
            )
        feature = fpn_list[self.use_level]
        if feature.ndim != 5:
            raise ValueError(
                "SV auxiliary features must have shape [B, N, C, H, W], "
                f"got {tuple(feature.shape)}"
            )

        batch_size, num_cams = feature.shape[:2]
        feature_2d = feature.flatten(0, 1).contiguous().float()
        zero = torch.nan_to_num(feature_2d, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        zero = zero + sum(
            torch.nan_to_num(parameter, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            for parameter in self.classifier.parameters()
            if parameter.requires_grad
        )
        det_boxes = data.get(self.det_box_key)
        det_cls = data.get(self.det_cls_key)
        if det_boxes is None or det_cls is None:
            return {"loss_sv_aux_cls": zero}

        rois, labels = self._gather_rois(
            det_boxes, det_cls, batch_size, num_cams, feature_2d.device
        )
        if rois is None:
            return {"loss_sv_aux_cls": zero}

        pooled = roi_align(
            feature_2d,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        logits = self.classifier(pooled.flatten(1))
        loss = self.loss_cls(logits, labels)
        if not bool(torch.isfinite(loss)):
            loss = zero
        return {"loss_sv_aux_cls": self.loss_weight * loss}
