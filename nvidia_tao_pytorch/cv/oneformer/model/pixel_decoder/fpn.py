# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature Pyramid Network implementation for pixel decoder."""

import logging
from typing import Dict, Optional, Union, Callable
from torch import nn
from torch.nn import functional as F

from nvidia_tao_pytorch.cv.oneformer.model.transformer_decoder.position_encoding import (
    PositionEmbeddingSine,
)
from nvidia_tao_pytorch.cv.oneformer.model.transformer_decoder.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from nvidia_tao_pytorch.cv.mask2former.model.pixel_decoder.msdeformattn import (
    MSDeformAttnPixelDecoder,
)


class ShapeSpec:
    """Specification for tensor shapes."""

    def __init__(self, channels=None, stride=None, **kwargs):
        # self.channels = channels
        self.channel = channels
        self.stride = stride
        for k, v in kwargs.items():
            setattr(self, k, v)


def configurable(init_method):
    """Decorator to make init method configurable."""
    def wrapped_init(self, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], "model"):
            cfg = args[0]
            remaining_args = args[1:]
            from_config_kwargs = self.__class__.from_config(cfg, *remaining_args)
            from_config_kwargs.update(kwargs)
            init_method(self, **from_config_kwargs)
        else:
            init_method(self, *args, **kwargs)

    return wrapped_init


def xavier_fill(module):
    """Initialize module weights with Xavier initialization."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


def get_norm(norm, out_channels):
    """Get normalization layer by name."""
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    elif norm == "GN":
        return nn.GroupNorm(32, out_channels)
    elif norm == "LN":
        return nn.LayerNorm(out_channels)
    elif norm == "" or norm is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


class Conv2d(nn.Module):
    """2D Convolution layer with normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.norm = None
        if norm is not None:
            if isinstance(norm, str):
                self.norm = get_norm(norm, out_channels)
            else:
                self.norm = norm

        self.activation = activation

    def forward(self, x):
        """Forward pass through Conv2d layer."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def build_pixel_decoder(cfg, input_shape):
    """Build pixel decoder from configuration."""
    name = cfg.model.sem_seg_head.pixel_decoder_name

    if name == "BasePixelDecoder":
        model = BasePixelDecoder(
            input_shape=input_shape,
            conv_dim=cfg.model.sem_seg_head.convs_dim,
            mask_dim=cfg.model.sem_seg_head.mask_dim,
            norm=cfg.model.sem_seg_head.norm,
        )
    elif name == "TransformerEncoderPixelDecoder":
        model = TransformerEncoderPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=cfg.model.one_former.dropout,
            transformer_nheads=cfg.model.one_former.nheads,
            transformer_dim_feedforward=cfg.model.one_former.dim_feedforward,
            transformer_enc_layers=cfg.model.sem_seg_head.transformer_enc_layers,
            transformer_pre_norm=cfg.model.one_former.pre_norm,
            conv_dim=cfg.model.sem_seg_head.convs_dim,
            mask_dim=cfg.model.sem_seg_head.mask_dim,
            norm=cfg.model.sem_seg_head.norm,
        )
    elif name == "MSDeformAttnPixelDecoder":
        model = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=cfg.model.one_former.dropout,
            transformer_nheads=cfg.model.one_former.nheads,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=cfg.model.sem_seg_head.transformer_enc_layers,
            conv_dim=cfg.model.sem_seg_head.convs_dim,
            mask_dim=cfg.model.sem_seg_head.mask_dim,
            transformer_in_features=cfg.model.sem_seg_head.in_features,
            common_stride=cfg.model.sem_seg_head.common_stride,
            norm=cfg.model.sem_seg_head.norm,
            export=cfg.model.export,
        )
    else:
        raise ValueError(f"Unknown pixel decoder: {name}")

    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


class BasePixelDecoder(nn.Module):
    """Base pixel decoder for feature pyramid networks."""

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == "" or norm is None
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                xavier_fill(output_conv.conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                xavier_fill(lateral_conv.conv)
                xavier_fill(output_conv.conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        xavier_fill(self.mask_features.conv)

        self.oneformer_num_feature_levels = 3

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        """Create BasePixelDecoder from configuration."""
        ret = {}
        ret["input_shape"] = {
            k: v
            for k, v in input_shape.items()
            if k in cfg.model.sem_seg_head.in_features
        }
        ret["conv_dim"] = cfg.model.sem_seg_head.convs_dim
        ret["mask_dim"] = cfg.model.sem_seg_head.mask_dim
        ret["norm"] = cfg.model.sem_seg_head.norm
        return ret

    def forward_features(self, features):
        """Extract multi-scale features from FPN backbone."""
        multi_scale_features = []
        num_cur_levels = 0
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.oneformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), None, multi_scale_features

    def forward(self, features, targets=None):
        """Forward pass through pixel decoder."""
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module."
        )
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    """Transformer encoder module for pixel decoder."""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """Forward pass through transformer encoder."""
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoderPixelDecoder(BasePixelDecoder):
    """Pixel decoder with transformer encoder."""

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        xavier_fill(self.input_proj.conv)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        use_bias = norm == "" or norm is None
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        xavier_fill(output_conv.conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        """Create TransformerEncoderPixelDecoder from configuration."""
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.model.one_former.dropout
        ret["transformer_nheads"] = cfg.model.one_former.nheads
        ret["transformer_dim_feedforward"] = cfg.model.one_former.dim_feedforward
        ret["transformer_enc_layers"] = cfg.model.sem_seg_head.transformer_enc_layers
        ret["transformer_pre_norm"] = cfg.model.one_former.pre_norm
        return ret

    def forward_features(self, features):
        """Extract multi-scale features with transformer encoder."""
        multi_scale_features = []
        num_cur_levels = 0
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.oneformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), transformer_encoder_features, multi_scale_features

    def forward(self, features, targets=None):
        """Forward pass through transformer encoder pixel decoder."""
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module."
        )
        return self.forward_features(features)
