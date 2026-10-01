# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric scale head for NVPanoptix3Dv2.

VGGT predicts depth in normalised scene coordinates (mean depth ~ 1). This head
regresses one global correction per scene,

    d_metric = s * d_rel + b,    s = exp(log_s) > 0

from a ``[B, 13 + C]`` descriptor of the scene's absolute physical scale:

     4  camera baseline     mean/std/min/max of ``||t_i - t_j||`` over view pairs
     2  translation norm    mean/std of ``||t_i||``
     5  depth distribution  mean/std/median/p10/p90 of ``d_rel``, view-averaged
     2  focal length        mean/std of ``(fx + fy) / 2``
     C  scene token         VGGT patch tokens mean-pooled over views and patches

Exponentiating ``log_s`` keeps the scale positive without clamping. The shift
``b`` is regressed only when ``predict_shift`` is set; disabling it gives the
scale-only correction ``d_metric = s * d_rel`` used by the panoptic variant.
It lives in depth units and is never applied to 3D points.
"""

from typing import Dict

import torch
import torch.nn as nn

SCALAR_FEAT_DIM = 13  # 4 baseline + 2 translation + 5 depth + 2 focal


def mean_std(values: torch.Tensor) -> torch.Tensor:
    """Mean and population std over the last axis, stacked as ``[..., 2]``.

    ``unbiased=False`` keeps the std well-defined when the axis holds a single
    element, which happens for a one-view context.
    """
    return torch.stack(
        [values.mean(dim=-1), values.std(dim=-1, unbiased=False)], dim=-1,
    )


def pose_encoding_to_intrinsics(
    pose_encoding: torch.Tensor,
    image_size_hw,
) -> torch.Tensor:
    """Reconstruct camera intrinsics from VGGT FoV pose components.

    Args:
        pose_encoding: VGGT camera encoding shaped ``[B, S, 9]``. Its final
            two values are the vertical and horizontal fields of view.
        image_size_hw: Input image ``(height, width)``.

    Returns:
        Camera intrinsics shaped ``[B, S, 3, 3]``.

    Raises:
        ValueError: If the pose encoding does not end in nine values.
    """
    if pose_encoding.shape[-1] != 9:
        raise ValueError(
            "VGGT pose encoding must end in 9 values, got "
            f"shape {tuple(pose_encoding.shape)}"
        )

    height, width = image_size_hw
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    intrinsics = pose_encoding.new_zeros(pose_encoding.shape[:2] + (3, 3))
    intrinsics[..., 0, 0] = (width / 2.0) / torch.tan(fov_w / 2.0)
    intrinsics[..., 1, 1] = (height / 2.0) / torch.tan(fov_h / 2.0)
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0
    return intrinsics


class MetricScaleHead(nn.Module):
    """Predict one global metric scale correction for VGGT depth.

    Args:
        scene_token_dim: Channel dim of the pooled VGGT patch token (2048 for
            the 1024-dim aggregator).
        hidden_dims: Hidden widths of the MLP.
        depth_min: Lower clamp applied when computing the depth statistics,
            avoiding percentile blow-up at zero depth.
        depth_max: Upper clamp applied when computing the depth statistics.
        metric_context_views: Deterministic view prefix used to estimate scale,
            so 20/50-view inference sees the training-time view-count
            distribution. Inputs with fewer views use all of them.
        predict_shift: Also regress the additive shift ``b``.
    """

    def __init__(
        self,
        scene_token_dim: int = 2048,
        hidden_dims=(256, 64),
        depth_min: float = 1e-3,
        depth_max: float = 1e3,
        metric_context_views: int = 5,
        predict_shift: bool = True,
    ):
        super().__init__()
        if metric_context_views <= 0:
            raise ValueError(
                f"metric_context_views must be positive, got {metric_context_views}"
            )
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.metric_context_views = int(metric_context_views)
        self.predict_shift = bool(predict_shift)

        layers = []
        in_dim = SCALAR_FEAT_DIM + scene_token_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
            in_dim = hidden
        # Zero-init the output layer so the head starts as the identity
        # (s = 1, b = 0) before training.
        out_layer = nn.Linear(in_dim, 2 if self.predict_shift else 1)
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        self.mlp = nn.Sequential(*layers, out_layer)

    def depth_stats(self, rel_depth: torch.Tensor) -> torch.Tensor:
        """Per-view depth distribution stats averaged over views, ``[B, 5]``.

        Returns (mean, std, median, p10, p90). ONNX export swaps this method for
        a sort-based equivalent, since ``torch.quantile`` has no lowering -- see
        ``nvpanoptix3d_v2.export.onnx_exporter.depth_stats_export_safe``.

        Args:
            rel_depth: ``[B, S, H, W, 1]`` or ``[B, S, H, W]`` normalised depth.
        """
        if rel_depth.dim() == 5:
            rel_depth = rel_depth.squeeze(-1)
        batch, views = rel_depth.shape[:2]
        d = rel_depth.reshape(batch, views, -1).clamp(self.depth_min, self.depth_max)

        # Ordered to match the stack below, so the result unpacks directly.
        qs = torch.tensor([0.5, 0.1, 0.9], device=d.device, dtype=d.dtype)
        median, p10, p90 = torch.quantile(d, qs, dim=-1)  # each [B, S]

        per_view = torch.stack(
            [d.mean(dim=-1), d.std(dim=-1, unbiased=False), median, p10, p90], dim=-1,
        )
        return per_view.mean(dim=1)

    def forward(
        self,
        vggt_feats: torch.Tensor,
        rel_depth: torch.Tensor,
        pose_enc: torch.Tensor,
        image_size_hw,
    ) -> Dict[str, torch.Tensor]:
        """Predict the per-scene scale from a fixed view prefix.

        Args:
            vggt_feats: ``[B, S, P, C]`` VGGT patch tokens (frozen).
            rel_depth: ``[B, S, H, W, 1]`` or ``[B, S, H, W]`` VGGT depth in
                normalised units (frozen).
            pose_enc: ``[B, S, 9]`` VGGT pose encoding (frozen).
            image_size_hw: ``(H, W)`` for the FoV -> focal length conversion.

        Returns:
            Dict with ``log_s`` and ``scale`` ``[B, 1]``, ``intrinsics``
            ``[B, S, 3, 3]`` for *all* views, and ``b`` ``[B, 1]`` when
            ``predict_shift`` is set.

        Raises:
            ValueError: If there are no views, or the three inputs disagree on
                the view count.
        """
        n_views = vggt_feats.shape[1]
        if n_views == 0:
            raise ValueError("Metric scale estimation requires at least one view")
        if rel_depth.shape[1] != n_views or pose_enc.shape[1] != n_views:
            raise ValueError(
                "Metric-head view counts must match: "
                f"vggt_feats={n_views}, rel_depth={rel_depth.shape[1]}, "
                f"pose_enc={pose_enc.shape[1]}"
            )

        # Decode every camera -- intrinsics are a per-view downstream output.
        # Only the context prefix feeds the scale features.
        intrinsics = pose_encoding_to_intrinsics(pose_enc.float(), image_size_hw)
        context = slice(0, min(n_views, self.metric_context_views))

        # Features are built under no_grad: the head must never backprop into
        # VGGT, so its input is a constant.
        with torch.no_grad():
            translations = pose_enc[:, context, :3].float()  # [B, S, 3]
            n_context = translations.shape[1]
            if n_context < 2:
                baseline = translations.new_zeros(translations.shape[0], 4)
            else:
                # Strict upper-triangular pairs of ||t_i - t_j||, [B, C(S,2)].
                dist = (translations.unsqueeze(2) - translations.unsqueeze(1)).norm(dim=-1)
                iu, ju = torch.triu_indices(
                    n_context, n_context, offset=1, device=dist.device,
                )
                pairs = dist[:, iu, ju]
                baseline = torch.stack(
                    [
                        pairs.mean(dim=-1), pairs.std(dim=-1, unbiased=False),
                        pairs.amin(dim=-1), pairs.amax(dim=-1),
                    ],
                    dim=-1,
                )

            intr = intrinsics[:, context]
            focals = 0.5 * (intr[..., 0, 0] + intr[..., 1, 1])  # [B, S]

            feats = torch.cat(
                [
                    baseline,                                          # [B, 4]
                    mean_std(translations.norm(dim=-1)),              # [B, 2]
                    self.depth_stats(rel_depth[:, context].float()),  # [B, 5]
                    mean_std(focals),                                 # [B, 2]
                    vggt_feats[:, context].float().mean(dim=(1, 2)),   # [B, C]
                ],
                dim=-1,
            )

        out = self.mlp(feats)
        log_s = out[:, 0:1]
        params = {"log_s": log_s, "scale": log_s.exp(), "intrinsics": intrinsics}
        if self.predict_shift:
            params["b"] = out[:, 1:2]
        return params


def apply_metric_scale(
    rel_depth: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the global correction ``d_metric = s * d_rel + b``.

    ``b`` is only present when the head was built with ``predict_shift``;
    otherwise this reduces to ``d_metric = s * d_rel``.

    Args:
        rel_depth: ``[B, S, H, W, 1]`` or ``[B, S, H, W]`` VGGT depth.
        params: Dict from :meth:`MetricScaleHead.forward`, carrying ``scale``
            ``[B, 1]`` and optionally ``b`` ``[B, 1]``.

    Returns:
        ``metric_depth``, shaped like ``rel_depth``.

    Raises:
        ValueError: If ``rel_depth`` is neither 4D nor 5D.
    """
    if rel_depth.dim() not in (4, 5):
        raise ValueError(
            f"rel_depth must be 4D or 5D, got shape {tuple(rel_depth.shape)}"
        )

    # ``scale``/``b`` are [B, 1]; broadcast over the view and pixel axes.
    shape = (-1, 1) + (1,) * (rel_depth.dim() - 2)
    metric_depth = params["scale"].reshape(shape) * rel_depth
    shift = params.get("b")
    if shift is not None:
        metric_depth = metric_depth + shift.reshape(shape)
    return metric_depth
