# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiNAT backbone wrapper for OneFormer."""

from addict import Dict

from nvidia_tao_pytorch.cv.backbone_v2.dinat import DiNAT


class D2DiNAT(DiNAT):
    """DiNAT backbone for Detectron2 / OneFormer integration.

    Thin wrapper around the shared DiNAT backbone in backbone_v2.
    Reads config from ``cfg.model.backbone.dinat.*`` and exposes
    ``output_shape()`` / ``size_divisibility`` for the OneFormer head.
    """

    def __init__(self, cfg, input_shape):
        embed_dim = cfg.model.backbone.dinat.embed_dim
        mlp_ratio = cfg.model.backbone.dinat.mlp_ratio
        depths = cfg.model.backbone.dinat.depths
        num_heads = cfg.model.backbone.dinat.num_heads
        drop_path_rate = cfg.model.backbone.dinat.drop_path_rate
        kernel_size = cfg.model.backbone.dinat.kernel_size
        out_indices = cfg.model.backbone.dinat.out_indices
        dilations = cfg.model.backbone.dinat.dilations

        super().__init__(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            kernel_size=kernel_size,
            out_indices=out_indices,
            dilations=dilations,
        )

        self._out_features = cfg.model.backbone.dinat.out_features

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """Forward function."""
        assert (
            x.dim() == 4
        ), f"DiNAT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        """Get output feature shape."""
        return {
            name: Dict({
                "channel": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name],
            })
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        """Get size divisibility."""
        return 32
