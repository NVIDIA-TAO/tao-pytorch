# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import timm
from einops import rearrange

from detectron2.modeling.backbone.backbone import Backbone


class TIMMBackbone(Backbone):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .
    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        *,
        features_only=False,
        pretrained=True,
        in_channels=3,
        num_classes=0,
        seq2spatial=False,
        stride,
        out_feature="last_feat",
        **kwargs,
    ):
        super().__init__()
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            **kwargs,
        )

        if features_only:
            assert not seq2spatial, "seq2spatial is not supported for features_only=True"

        self.features_only = features_only
        self.seq2spatial = seq2spatial
        self.stride = stride

        self._out_feature_channels = {
            out_feature: self.timm_model.num_features,
        }
        self._out_feature_strides = {
            out_feature: stride,
        }
        self._out_features = list(self._out_feature_channels.keys())
        self._size_divisibility = stride

    def forward(self, x):
        height, width = x.shape[-2:]

        if self.features_only:
            return self.timm_model(x)
        else:
            feat = self.timm_model.forward_features(x)

        if self.seq2spatial:
            feat = feat[:, 1:]
            feat = rearrange(
                feat, "b (h w) c -> b c h w", h=height // self.stride, w=width // self.stride
            )

        return {self._out_features[0]: feat}
