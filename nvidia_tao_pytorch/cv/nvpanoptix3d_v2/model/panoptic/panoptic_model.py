# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint 3D reconstruction and open-vocabulary panoptic segmentation.

A feed-forward model over unposed multi-view images. VGGT is the 3D backbone;
its tokens are concatenated with DINOv2's, projected, and refined by a
three-block per-view spatial mixer. The decoder reads one native stride-14
memory plus stride-2 LoftUp mask features, and a metric scale head converts
VGGT's normalised depth to metric.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import (
    MetricScaleHead,
    apply_metric_scale,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.panoptic_decoder import (
    NVPanoptix3Dv2PanopticDecoder,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt import VGGT


class NVPanoptix3Dv2Panoptic(nn.Module):
    """
    Main NVPanoptix3Dv2 model.

    Combines a VGGT backbone (3D reconstruction + multi-scale features) with a
    multi-view panoptic segmentation decoder.

    Forward pipeline:
      1. VGGT Backbone -> DINOv2 feats, VGGT feats, depth, 3D points, cameras
      2. PanopticDecoder -> concat projection -> spatial mixer -> LoftUp ->
         2D-PE Mask Transformer
      3. MetricScaleHead(first 5 views) -> one scene scale,
         applied to metric depth and world points for every view
    """

    def __init__(
        self,
        vggt_backbone: VGGT,
        panoptic_decoder: NVPanoptix3Dv2PanopticDecoder,
        metric_depth_head: Optional[MetricScaleHead] = None,
    ):
        super().__init__()

        self.vggt_backbone = vggt_backbone
        self.panoptic_decoder = panoptic_decoder
        self.metric_depth_head = metric_depth_head

    def freeze_vggt_weights(self):
        """Freeze the complete VGGT backbone."""
        for param in self.vggt_backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        true_shape: torch.Tensor,
        classes: List[str],
        outdevice: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            images:     [B, S, 3, H, W]  input images in [0, 1]
            true_shape: [B, S, 2]         compatibility metadata; spatial
                                          layout comes from ``images``
            classes:    list of class names for open-vocabulary prediction
            outdevice:  optional target device for outputs

        Returns:
            (panoptic_output, geometry_output)

            panoptic_output:
              pred_logits:  [B, Q, C]       class predictions
              pred_masks:   [B, S, Q, H, W] mask predictions
              aux_outputs:  list of intermediate predictions

            geometry_output:
              depth:            [B, S, H, W, 1]   relative (normalised) depth
              depth_conf:       [B, S, H, W]
              world_points:     [B, S, H, W, 3]   normalised world points (from frozen point_head)
              world_points_conf:[B, S, H, W]
              pose_enc:         [B, S, 9]
              metric_scale_params: dict with ``log_s``, ``scale``, ``intrinsics``
                                   (if metric_depth_head is enabled)
              metric_depth:       [B, S, H, W, 1]   (if metric_depth_head) — corrected
              metric_points:      [B, S, H, W, 3]   (if metric_depth_head) — scaled world_points
              intrinsics:         [B, S, 3, 3]      reconstructed intrinsics (metric scale)
        """
        expected_shape = (images.shape[0], images.shape[1], 2)
        if tuple(true_shape.shape) != expected_shape:
            raise ValueError(
                f"true_shape must have shape {expected_shape}, got "
                f"{tuple(true_shape.shape)}"
            )

        backbone_out = self.vggt_backbone(images)

        dino_feats = backbone_out["dino_feats"]
        vggt_feats = backbone_out["vggt_feats"]
        depth = backbone_out.get("depth")
        world_points = backbone_out.get("world_points")

        panout = self.panoptic_decoder(
            dino_feats=dino_feats,
            vggt_feats=vggt_feats,
            images=images,
            classes=classes,
            outdevice=outdevice,
        )

        geometry_output = {
            k: backbone_out[k]
            for k in ("depth", "depth_conf", "world_points", "world_points_conf", "pose_enc")
            if k in backbone_out
        }

        if self.metric_depth_head is not None and depth is not None:
            pose_enc = backbone_out.get("pose_enc")
            if pose_enc is not None:
                _, _, H_d, W_d, _ = depth.shape

                metric_scale_params = self.metric_depth_head(
                    vggt_feats=vggt_feats,
                    rel_depth=depth,
                    pose_enc=pose_enc,
                    image_size_hw=(H_d, W_d),
                )
                scale = metric_scale_params["scale"]  # [B, 1]

                geometry_output["metric_scale_params"] = metric_scale_params
                geometry_output["intrinsics"] = metric_scale_params["intrinsics"]

                # The head estimates one scene-level scale correction from
                # its fixed five-view context. Apply it to every view,
                # including views outside that context at inference.
                metric_depth = apply_metric_scale(depth, metric_scale_params)
                geometry_output["metric_depth"] = metric_depth

                # Scale VGGT's globally-aligned world_points to metric.
                # VGGT's point_head produces multi-view-consistent points in a
                # shared coordinate frame; applying the same predicted scale
                # to depth and points keeps both metric outputs consistent.
                if world_points is not None:
                    metric_points = world_points * scale[:, :, None, None, None]
                    geometry_output["metric_points"] = metric_points

        return panout, geometry_output

    def set_vocab(self, class_names: List[str], device=None):
        """Pre-compute text embeddings for a fixed class vocabulary."""
        self.panoptic_decoder.text_encoder.set_vocab(class_names, device=device)
