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
"""Swin Transformer backbone implementation for OneFormer."""

from addict import Dict
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.backbone_v2.swin import SwinTransformer


class D2SwinTransformer(SwinTransformer):
    """Swin Transformer backbone for Detectron2 integration."""

    def __init__(self, cfg, input_shape):

        pretrain_img_size = cfg.model.backbone.swin.pretrain_img_size
        patch_size = cfg.model.backbone.swin.patch_size
        in_chans = 3
        embed_dim = cfg.model.backbone.swin.embed_dim
        depths = cfg.model.backbone.swin.depths
        num_heads = cfg.model.backbone.swin.num_heads
        window_size = cfg.model.backbone.swin.window_size
        mlp_ratio = cfg.model.backbone.swin.mlp_ratio
        qkv_bias = cfg.model.backbone.swin.qkv_bias
        qk_scale = cfg.model.backbone.swin.qk_scale
        drop_rate = cfg.model.backbone.swin.drop_rate
        attn_drop_rate = cfg.model.backbone.swin.attn_drop_rate
        drop_path_rate = cfg.model.backbone.swin.drop_path_rate
        norm_layer = nn.LayerNorm
        ape = cfg.model.backbone.swin.ape
        patch_norm = cfg.model.backbone.swin.patch_norm
        use_checkpoint = cfg.model.backbone.swin.use_checkpoint

        super().__init__(
            img_size=pretrain_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            activation_checkpoint=use_checkpoint,
        )

        self._out_features = cfg.model.backbone.swin.out_features

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_inter_features[0],
            "res3": self.num_inter_features[1],
            "res4": self.num_inter_features[2],
            "res5": self.num_inter_features[3],
        }

    def forward(self, x):
        """Forward function."""
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_inter_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out

        for k in outs.keys():
            if k in self._out_features:
                outputs[k] = outs[k]
        return outputs

    def output_shape(self):
        """Get output feature shape."""
        backbone_feature_shape = dict()
        for name in self._out_features:
            backbone_feature_shape[name] = Dict({'channel': self._out_feature_channels[name], 'stride': self._out_feature_strides[name]})
        return backbone_feature_shape

    @property
    def size_divisibility(self):
        """Get size divisibility."""
        return 32
