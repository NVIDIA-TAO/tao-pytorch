# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO backbone implementation for Detectron2-style integration.

Wraps the TAO :class:`~nvidia_tao_pytorch.cv.backbone_v2.radio.RADIO` model for
OneFormer / Detectron2-style backbones. Follows the ViTDet Simple Feature Pyramid
strategy to derive multi-scale features from a single-scale ViT backbone
(arXiv:2203.16527).

Exports:
    ChannelLayerNorm: Per-spatial-location normalization over channels for (B, C, H, W).
    D2RADIO: Multi-scale backbone reading ``cfg.model.backbone.radio`` and optional
        ``cfg.model.backbone.freeze_at``.
"""

import torch
import torch.nn as nn
from addict import Dict
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension for (B, C, H, W) tensors.

    Standard nn.LayerNorm normalizes the last dims which is wrong for
    feature maps. This variant normalizes per-pixel across channels,
    matching the ConvNeXt / ViTDet convention.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize channel-wise LayerNorm.

        Args:
            num_channels (int): Number of channels ``C`` for inputs of shape (N, C, H, W).
            eps (float): Epsilon added inside the square root for the variance. Default: 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Forward function."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class D2RADIO(RADIO):
    """RADIO backbone with ViTDet-style multi-scale feature adaptation.

    RADIO produces single-scale spatial tokens at stride = patch_size (16).

        res3 (stride  8): ConvTranspose2d 2x upsample, channels C → C//2
        res4 (stride 16): native scale (identity)
        res5 (stride 32): MaxPool2d 2x downsample

    Channel reduction at finer scales follows ViTDet: finer features are
    lower-dimensional since they encode local/boundary patterns.  MaxPool
    at the coarsest level is parameter-free and selects strongest activations.
    The downstream pixel decoder's own input_proj handles per-level projection.
    """

    def __init__(self, cfg, input_shape):
        """Build D2RADIO from config.

        Args:
            cfg: Global experiment config. Must define ``cfg.model.backbone.radio`` with
                fields used by :class:`~nvidia_tao_pytorch.cv.backbone_v2.radio.RADIO`
                (``resolution``, ``backbone``, ``summary_idxs``, ``window_size``,
                ``num_teacher``, ``cpe_max_size``, ``register_multiple``, ``use_checkpoint``,
                ``out_features``). Optional ``cfg.model.backbone.freeze_at`` (int) maps to
                ``freeze_at`` on the base RADIO constructor when non-negative.
            input_shape: Declared input shape for the backbone (Detectron2 convention).
                Retained for API compatibility; construction uses ``cfg`` only.
        """
        radio_cfg = cfg.model.backbone.radio
        backbone_cfg = cfg.model.backbone
        _freeze_at = getattr(backbone_cfg, "freeze_at", -1)
        freeze_at = None if _freeze_at < 0 else [_freeze_at]
        super().__init__(
            resolution=radio_cfg.resolution,
            backbone=radio_cfg.backbone,
            summary_idxs=radio_cfg.summary_idxs,
            window_size=radio_cfg.window_size,
            num_teacher=radio_cfg.num_teacher,
            cpe_max_size=radio_cfg.cpe_max_size,
            register_multiple=radio_cfg.register_multiple,
            activation_checkpoint=radio_cfg.use_checkpoint,
            freeze_at=freeze_at,
        )

        self._out_features = radio_cfg.out_features
        C = self.radio.radio.model.embed_dim

        self._out_feature_strides = {
            "res3": self.patch_size // 2,
            "res4": self.patch_size,
            "res5": self.patch_size * 2,
        }
        self._out_feature_channels = {
            "res3": C // 2,
            "res4": C,
            "res5": C,
        }

        if "res3" in self._out_features:
            self.upsample_proj = nn.Sequential(
                nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2),
                ChannelLayerNorm(C // 2),
                nn.GELU(),
            )
        if "res5" in self._out_features:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward function."""
        assert (
            x.dim() == 4
        ), f"RADIO takes an input of shape (N, C, H, W). Got {x.shape} instead!"

        _, spatial_tokens = self.radio(x)
        B, _, C = spatial_tokens.shape
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        res4 = spatial_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)

        outputs = {}
        if "res4" in self._out_features:
            outputs["res4"] = res4
        if "res3" in self._out_features:
            outputs["res3"] = self.upsample_proj(res4)
        if "res5" in self._out_features:
            outputs["res5"] = self.downsample(res4)
        return outputs

    def output_shape(self):
        """Get output feature shape."""
        backbone_feature_shape = dict()
        for name in self._out_features:
            if name in self._out_feature_channels:
                backbone_feature_shape[name] = Dict({
                    'channel': self._out_feature_channels[name],
                    'stride': self._out_feature_strides[name]
                })
        return backbone_feature_shape

    @property
    def size_divisibility(self):
        """Get size divisibility."""
        return self.patch_size * 2
