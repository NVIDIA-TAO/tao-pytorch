# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EfficientViT backbone builder."""
from addict import Dict
from torch import nn

from nvidia_tao_pytorch.core.decorators import experimental
from nvidia_tao_pytorch.cv.backbone_v2.efficientvit import (
    efficientvit_b0,
    efficientvit_b1,
    efficientvit_b2,
    efficientvit_b3,
    efficientvit_l0,
    efficientvit_l1,
    efficientvit_l2,
    efficientvit_l3,
)

arch = {
    'b0': (efficientvit_b0, [8, 16, 32, 64, 128]),
    'b1': (efficientvit_b1, [16, 32, 64, 128, 256]),
    'b2': (efficientvit_b2, [24, 48, 96, 192, 384]),
    'b3': (efficientvit_b3, [32, 64, 128, 256, 512]),
    'l0': (efficientvit_l0, [32, 64, 128, 256, 512]),
    'l1': (efficientvit_l1, [32, 64, 128, 256, 512]),
    'l2': (efficientvit_l2, [32, 64, 128, 256, 512]),
    'l3': (efficientvit_l3, [64, 128, 256, 512, 1024]),
}


@experimental("EfficientViT is not extensively tested and using it may result in lower accuracy.")
class EfficientViT(nn.Module):
    """EfficientViT."""

    def __init__(self, cfg, **kwargs):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.model.backbone.efficientvit.name
        self.model = arch[self.model_name][0](**kwargs)
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = [4, 8, 16, 32]

    def forward(self, x):
        """Forward pass."""
        y = self.model(x)
        z = dict()
        for k, v in y.items():
            if 'stage' in k and len(k) == 6:
                z['res' + str(int(k[-1]) + 1)] = v
        return z

    def output_shape(self):
        """Get output shape."""
        backbone_feature_shape = dict()
        for i, name in enumerate(self._out_features):
            backbone_feature_shape[name] = Dict(
                {'channel': arch[self.model_name][1][i + 1],
                 'stride': self._out_feature_strides[i]}
            )
        return backbone_feature_shape

    @property
    def size_divisibility(self):
        """Size divisibility."""
        return 32
