# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submodule contains small module classes for various parts of FounationStereo"""

import math
import itertools
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.convolution_helper import (Conv, Conv3dNormActReduced)

torch.backends.cuda.enable_flash_sdp(True)


class FlashMultiheadAttention(nn.Module):
    """Multihead Attention with Flash Attention."""

    def __init__(self, embed_dim, num_heads, qkv_layout: str = 'standard'):
        """Initializes FlashMultiheadAttention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            qkv_layout: Reshape layout for Q/K/V before SDPA. 'standard'
                (default) reshapes to (B, num_heads, L, head_dim) per the
                PyTorch ``scaled_dot_product_attention`` contract.
                'upstream_compat' reshapes to (B, L, num_heads, head_dim) —
                a non-standard layout that the FFS commercial bp2 ckpt was
                trained with (heads-as-sequence semantics).
        """
        super().__init__()
        if qkv_layout not in ('standard', 'upstream_compat'):
            raise ValueError(
                f"qkv_layout must be 'standard' or 'upstream_compat', got {qkv_layout!r}")
        self.qkv_layout = qkv_layout
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1, -1)):
        """Forward pass of FlashMultiheadAttention.

        Args:
            query: Query tensor (B, L, C).
            key: Key tensor (B, L, C).
            value: Value tensor (B, L, C).
            attn_mask: Attention mask.
            window_size: Window size for attention.

        Returns:
            Output tensor.
        """
        B, L, _ = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        if self.qkv_layout == 'standard':
            Q = Q.view(Q.size(0), self.num_heads, Q.size(1), self.head_dim)
            K = K.view(K.size(0), self.num_heads, K.size(1), self.head_dim)
            V = V.view(V.size(0), self.num_heads, V.size(1), self.head_dim)
        else:  # 'upstream_compat'
            Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
            K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
            V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        if torch.onnx.is_in_onnx_export():
            # ONNX trace requires the math backend.
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(Q, K, V)
        else:
            # train / inference: let PyTorch auto-select.
            attn_output = F.scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.reshape(B, L, -1)
        output = self.out_proj(attn_output)

        return output


class FeatureAtt(nn.Module):
    """
    Computes multiplicative feature matching between the cost volume and input feature.

    This module implements a feature attention mechanism that uses a 2D feature map
    to "attend" to a 3D cost volume. It learns a per-channel multiplicative
    attention mask for the cost volume based on the input feature.

    Args:
        cv_chan (int): The number of channels in the cost volume.
        feat_chan (int): The number of channels in the input feature map.
        norm_type (str, optional): Normalization for the inner Conv. Defaults
            to ``'instance2d'``. Pass ``'batch2d'`` to load upstream ckpts
            that store BatchNorm at this position.
    """

    def __init__(self, cv_chan, feat_chan, norm_type: str = 'instance2d'):
        super().__init__()

        # A small sub-network to produce the attention map from the feature map.
        # It consists of a 1x1 convolution to reduce the number of channels,
        # followed by another 1x1 convolution to match the cost volume's channel
        # dimension.
        self.feat_att = nn.Sequential(
            Conv(feat_chan, feat_chan // 2,
                 conv_type='conv2d', norm_type=norm_type,
                 relu=True, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1))

    def forward(self, cv, feat):
        """
        Defines the computation performed at every call.

        The forward pass computes a 2D attention map from the feature map,
        expands it to a 3D tensor to match the dimensions of the cost volume,
        applies a sigmoid activation, and then performs a element-wise
        multiplication with the cost volume.

        Args:
            cv (torch.Tensor): The cost volume tensor of shape
                `(B, C, D, H, W)`, where `B` is the batch size, `C` is the
                number of channels, `D` is the disparity/depth dimension,
                and `H` and `W` are the height and width.
            feat (torch.Tensor): The input feature map tensor of shape
                `(B, C_feat, H, W)`, where `B` is the batch size, `C_feat` is the
                number of channels, and `H` and `W` are the height and width.

        Returns:
            torch.Tensor: The attended cost volume tensor, with the same shape
                as the input cost volume `(B, C, D, H, W)`.
        """
        # Generate the 2D attention map from the feature map
        feat_att = self.feat_att(feat)

        # Unsqueeze the attention map to match the cost volume's dimensions
        # This adds a new dimension for the disparity/depth dimension.
        # Shape changes from (B, cv_chan, H, W) to (B, cv_chan, 1, H, W)
        feat_att = feat_att.unsqueeze(2)

        # Apply a sigmoid activation to the attention map to constrain values
        # between 0 and 1.
        # Perform element-wise multiplication with the cost volume.
        # This acts as a per-channel, per-pixel multiplicative gate for the cost volume.
        cv = torch.sigmoid(feat_att) * cv
        return cv


class FlashAttentionTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Flash Attention."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout=0.1,
        act=nn.GELU,
        norm=nn.LayerNorm,
        qkv_layout: str = 'standard',
    ):
        """Initializes FlashAttentionTransformerEncoderLayer.

        Args:x.shape
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
            act: Activation function.
            norm: Normalization layer.
            qkv_layout: Forwarded to FlashMultiheadAttention.
        """
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads, qkv_layout=qkv_layout)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        """Forward pass of FlashAttentionTransformerEncoderLayer.

        Args:
            src: Input tensor.
            src_mask: Mask for source tensor.
            window_size: Window size for attention.

        Returns:
            Output tensor.
        """
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class HourGlass(nn.Module):
    """HourGlass network module for feature extraction and aggregation."""

    def __init__(self, cfg, in_channels, feat_dims=None,
                 agg_1_conv_type: str = 'deconv',
                 atts_max_len_divisor: int = 4,
                 feat_att_norm_type: str = 'instance2d',
                 conv_patch_padding: Optional[List[int]] = None,
                 qkv_layout: str = 'standard'):
        """Initializes HourGlass.

        Args:
            cfg: Configuration dictionary.
            in_channels: Number of input channels.
            feat_dims: List of feature dimensions.
            agg_1_conv_type: Convolution kind for the first ``agg_1`` layer.
                ``'deconv'`` (default) → ``deconv3d``. ``'conv'`` → ``conv3d``.
            conv_patch_padding (Optional[List[int]], optional): Padding for the
                ``conv_patch`` Conv3d. None (default) keeps the legacy
                ``[1, 1, 0]``. Pass ``[0, 0, 0]`` to match the training repo's
                FFS forward.
            atts_max_len_divisor: Divisor applied to ``cfg['max_disparity']`` to
                size the cost-volume attention's ``max_len``. 4 matches the
                post-upsample disparity dimension; 16 pairs with
                ``atts_before_upsample=True``.
            feat_att_norm_type: Normalization for the inner ``FeatureAtt`` Conv,
                propagated to all five ``feature_att_*`` children. Defaults to
                ``'instance2d'``; pass ``'batch2d'`` for upstream ckpts that
                store BatchNorm at the FeatureAtt position.
        """
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(
            Conv(
                in_channels,
                in_channels * 2,
                conv_type="conv3d",
                norm_type="batch3d",
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
        )

        self.conv2 = nn.Sequential(
            Conv(
                in_channels * 2,
                in_channels * 4,
                conv_type="conv3d",
                norm_type="batch3d",
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
        )

        self.conv3 = nn.Sequential(
            Conv(
                in_channels * 4,
                in_channels * 6,
                conv_type="conv3d",
                norm_type="batch3d",
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(
                in_channels * 6, in_channels * 6, kernel_size=3, kernel_disp=17
            ),
        )

        self.conv3_up = Conv(
            in_channels * 6,
            in_channels * 4,
            conv_type="deconv3d",
            norm_type="batch3d",
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv2_up = Conv(
            in_channels * 4,
            in_channels * 2,
            conv_type="deconv3d",
            norm_type="batch3d",
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv1_up = Conv(
            in_channels * 2,
            in_channels,
            conv_type="deconv3d",
            norm_type="batch3d",
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv_out = nn.Sequential(
            Conv3dNormActReduced(
                in_channels, in_channels, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels, in_channels, kernel_size=3, kernel_disp=17
            ),
        )

        self.agg_0 = nn.Sequential(
            Conv(
                in_channels * 8,
                in_channels * 4,
                conv_type="conv3d",
                norm_type="batch3d",
                relu=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17
            ),
        )

        agg_1_full_conv_type = "deconv3d" if agg_1_conv_type == 'deconv' else "conv3d"
        self.agg_1 = nn.Sequential(
            Conv(
                in_channels * 4,
                in_channels * 2,
                conv_type=agg_1_full_conv_type,
                norm_type="batch3d",
                relu=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
            Conv3dNormActReduced(
                in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17
            ),
        )

        self.atts = nn.ModuleDict(
            {
                'cost_vol_disp': CostVolumeDisparityAttention(
                    d_model=in_channels,
                    nhead=4,
                    dim_feedforward=in_channels,
                    norm_first=False,
                    num_transformer=4,
                    max_len=self.cfg["max_disparity"] // atts_max_len_divisor,
                    qkv_layout=qkv_layout,
                )
            }
        )

        if conv_patch_padding is None:
            conv_patch_padding = [1, 1, 0]
        self.conv_patch = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=4,
                padding=conv_patch_padding,
                groups=in_channels,
            ),
            nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels * 2, feat_dims[1], norm_type=feat_att_norm_type)
        self.feature_att_16 = FeatureAtt(in_channels * 4, feat_dims[2], norm_type=feat_att_norm_type)
        self.feature_att_32 = FeatureAtt(in_channels * 6, feat_dims[3], norm_type=feat_att_norm_type)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, feat_dims[2], norm_type=feat_att_norm_type)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, feat_dims[1], norm_type=feat_att_norm_type)

    def forward(self, x, features, atts_before_upsample: bool = False):
        """Forward pass of HourGlass.

        Args:
            x: Input tensor.
            features: List of feature tensors.
            atts_before_upsample: When False (default), the cost-volume
                attention runs on the trilinear-upsampled cost volume. When
                True, the attention runs on the downsampled patches and the
                upsample happens after.

        Returns:
            Output tensor.
        """
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        if atts_before_upsample:
            x = self.atts["cost_vol_disp"](x)
            x = F.interpolate(x, scale_factor=4, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=4, mode="trilinear", align_corners=False)
            x = self.atts["cost_vol_disp"](x)
        conv = conv + x
        conv = self.conv_out(conv)
        return conv


class PositionalEmbedding(nn.Module):
    """Positional Embedding Module."""

    def __init__(self, d_model, max_len=512):
        """Initializes PositionalEmbedding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(int(max_len), int(d_model)).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (N,1)
        div_term = (
            (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[
                None
            ]
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # (N, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x, resize_embed=False):
        """Forward pass of PositionalEmbedding.

        Args:
            x: Input tensor (B, N, D).
            resize_embed: Whether to resize positional embedding.

        Returns:
            Input tensor with added positional embeddings.
        """
        self.pe = self.pe.to(x.device).to(x.dtype)
        pe = self.pe
        if pe.shape[1] < x.shape[1]:
            if resize_embed:
                pe = F.interpolate(
                    pe.permute(0, 2, 1), size=x.shape[1], mode="linear", align_corners=False
                ).permute(0, 2, 1)
            else:
                raise RuntimeError(f"x:{x.shape}, pe:{pe.shape}")

        return x + pe[:, : x.size(1)]


class DualAttentionMlp(nn.Module):
    """Computes Dual Attention: self and cross attention"""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.0, act=nn.ReLU):
        """Initializes DualAttentionMlp.

        Args:
            d_model: The number of expected features in the input (required).
            nhead: The number of heads in the multiheadattention models (required).
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            act: Activation function.
        """
        super().__init__()
        self.iea = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.iea_r = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cea = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm1r = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.linear1r = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, src, ref, mask_cross, mask_self):
        """Forward pass of DualAttentionMlp.

        Args:
            src: The sequence to the encoder (required).
            ref: The sequence to the reference encoder (required).
            mask_cross: The mask for cross attention.
            mask_self: The mask for self attention.

        Returns:
            Tuple of output src and output ref.
        """
        if mask_cross is not None:
            assert mask_cross.dtype == torch.bool
        if mask_self is not None:
            assert mask_self.dtype == torch.bool

        # Self-attention
        with torch.cuda.amp.autocast(enabled=False):
            src_iea, _ = self.iea(
                src.float(), src.float(), value=src.float(), attn_mask=mask_self
            )
        out = self.norm1(src + self.linear1(src_iea))

        with torch.cuda.amp.autocast(enabled=False):
            ref_iea, _ = self.iea_r(
                ref.float(), ref.float(), value=ref.float(), attn_mask=mask_self
            )
        out_ref = self.norm1r(ref + self.linear1r(ref_iea))

        # Cross-attention
        with torch.cuda.amp.autocast(enabled=False):
            src_cea, _ = self.cea(
                out.float(), out_ref.float(), value=out_ref.float(), attn_mask=mask_cross
            )
        out = self.norm2(out + self.linear2(src_cea))

        return out, out_ref


class AttentionCorrelation(nn.Module):
    """Attention Correlation Module."""

    def __init__(
        self,
        d_in,
        d_model,
        n_att,
        dim_feedforward=1024,
        nhead=8,
        max_len=512,
        use_ignore_mask=False,
    ):
        """Initializes AttentionCorrelation.

        Args:
            d_in: Input feature dimension.
            d_model: Model dimension.
            n_att: Number of attention layers.
            dim_feedforward: Feedforward network dimension.
            nhead: Number of attention heads.
            max_len: Maximum sequence length for positional embedding.
            use_ignore_mask: Whether to use ignore mask.
        """
        super().__init__()
        self.ignore_mask = None
        self.use_ignore_mask = use_ignore_mask
        self.conv0 = nn.Conv2d(d_in, d_model, kernel_size=1)
        self.att = nn.ModuleList()
        for _ in range(n_att):
            self.att.append(
                DualAttentionMlp(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0,
                    act=nn.GELU,
                )
            )
        self.pos_embed = PositionalEmbedding(d_model, max_len=max_len)
        self.conv1 = nn.Conv2d(d_model, d_in, kernel_size=1)

    def create_mask(self, W):
        """Creates ignore mask.

        Args:
            W: Width of the mask.
        """
        self.ignore_mask = torch.zeros((W, W), dtype=torch.bool)
        for w in range(W):
            mask = torch.ones((W), dtype=torch.bool, requires_grad=False)
            mask[: w + 1] = 0
            self.ignore_mask[w] = mask

    def forward(self, xl, xr, bs=None):
        """Forward pass of AttentionCorrelation.

        Args:
            xl: Left input tensor (B, C, H, W).
            xr: Right input tensor (B, C, H, W).
            bs: Batch size (unused).

        Returns:
            Tuple of processed left and right tensors.
        """
        xl = self.conv0(xl)
        xr = self.conv0(xr)
        B, C, H, W = xl.shape
        xl = xl.permute(0, 2, 3, 1).reshape(B * H, W, C).contiguous()
        xr = xr.permute(0, 2, 3, 1).reshape(B * H, W, C).contiguous()
        if self.use_ignore_mask:
            if self.ignore_mask is None or self.ignore_mask.shape[0] != W:
                self.create_mask(W)
                self.ignore_mask = self.ignore_mask.to(xl.device).bool()

        if isinstance(self.pos_embed, PositionalEmbedding):
            xl = self.pos_embed(xl, resize_embed=True)
            xr = self.pos_embed(xr, resize_embed=True)
        else:
            xl = self.pos_embed(xl)
            xr = self.pos_embed(xr)

        x_src = xl
        x_ref = xr
        for i in range(len(self.att)):
            x_src, x_ref = self.att[i](
                src=x_src, ref=x_ref, mask_cross=self.ignore_mask, mask_self=None
            )

        x_src = x_src.reshape(B * H, W, -1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,d_model,H,W)
        x_ref = x_ref.reshape(B * H, W, -1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,d_model,H,W)
        x_src = self.conv1(x_src)
        x_ref = self.conv1(x_ref)
        return x_src, x_ref


class CostVolumeDisparityAttention(nn.Module):
    """Cost Volume Disparity Attention Module."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        act=nn.GELU,
        norm_first=False,
        num_transformer=6,
        max_len=512,
        resize_embed=False,
        qkv_layout: str = 'standard',
    ):
        """Initializes CostVolumeDisparityAttention.

        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
            act: Activation function.
            norm_first: Whether to apply layer norm before attention.
            num_transformer: Number of transformer layers.
            max_len: Maximum sequence length for positional embedding.
            resize_embed: Whether to resize positional embedding.
            qkv_layout: Forwarded to FlashMultiheadAttention via the
                transformer encoder layer. 'standard' (default) for TAO
                regular FS / mono paths; 'upstream_compat' for the FFS
                commercial bp2 ckpt.
        """
        super().__init__()
        self.resize_embed = resize_embed
        self.sa = nn.ModuleList([])
        for _ in range(num_transformer):
            self.sa.append(
                FlashAttentionTransformerEncoderLayer(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dim_feedforward=dim_feedforward,
                    act=act,
                    dropout=dropout,
                    qkv_layout=qkv_layout,
                )
            )
        self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)

    def forward(self, cv, window_size=(-1, -1)):
        """Forward pass of CostVolumeDisparityAttention.

        Args:
            cv: Cost volume tensor (B, C, D, H, W) where D is max disparity.
            window_size: Window size for attention.

        Returns:
            Processed cost volume tensor.
        """
        x = cv
        B, C, D, H, W = x.shape
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)  # N x 28 x 128 x 168 x 256
        x = self.pos_embed0(x, resize_embed=self.resize_embed)  # !NOTE No resize since disparity is pre-determined
        for i in range(len(self.sa)):
            x = self.sa[i](x, window_size=window_size)
        x = x.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        return x


class ChannelAttentionEnhancement(nn.Module):
    """Channel Attention Enhancement Module."""

    def __init__(self, in_planes, ratio=16, mid_channels: Optional[int] = None):
        """Initializes ChannelAttentionEnhancement.

        From selective-IGEV.

        Args:
            in_planes: Input channel dimension.
            ratio: Reduction ratio for the hidden dimension.
            mid_channels: Explicit override for the hidden (mid) dimension.
                None (default) keeps the legacy ``in_planes // 16`` formula.
                When set, ``ratio`` is ignored.
        """
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if mid_channels is None:
            mid_channels = in_planes // 16
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of ChannelAttentionEnhancement.

        Args:
            x: Input tensor.

        Returns:
            Channel attention weights.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionExtractor(nn.Module):
    """Spatial Attention Extractor Module."""

    def __init__(self, kernel_size=7):
        """Initializes SpatialAttentionExtractor.

        From selective-IGEV.

        Args:
            kernel_size: Size of the convolutional kernel.
        """
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of SpatialAttentionExtractor.

        Args:
            x: Input tensor.

        Returns:
            Spatial attention weights.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)


class LayerNorm2d(torch.nn.LayerNorm):
    """LayerNorm for channels_first tensors with 2D spatial dimensions (N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        """Initializes LayerNorm2d.

        Args:
            normalized_shape (int): Channel dimension.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of LayerNorm2d.

        Routes through ``F.layer_norm`` via channels-last permute so ONNX
        export emits a single ``LayerNormalization`` op.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Normalized output of the same shape, in the input
            tensor's dtype.
        """
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(),
            self.normalized_shape, self.weight, self.bias, self.eps,
        ).permute(0, 3, 1, 2).contiguous()


class LayerNorm3d(torch.nn.LayerNorm):
    """LayerNorm for channels_first tensors with 3D spatial dimensions (N, C, D, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of LayerNorm3d.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        return (
            F.layer_norm(
                x.permute(0, 2, 3, 4, 1).contiguous(),
                self.normalized_shape, self.weight, self.bias, self.eps)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )


class CenterPadding(nn.Module):
    """
    Applies center padding to a tensor to make its height and width dimensions
    multiples of a given factor.

    This module is particularly useful in deep learning pipelines where network
    architectures, such as those with multiple downsampling layers, require
    input tensor dimensions to be exact multiples of a specific number. The
    padding is applied symmetrically to the top/bottom and left/right of the
    tensor, ensuring that the original content remains centered.

    Attributes:
        multiple (int): The integer factor by which the height and width of
                        the input tensor will be made divisible.
    """

    def __init__(self, multiple: int):
        """
        Initializes the CenterPadding module.

        Args:
            multiple (int): The integer multiple for padding.
        """
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> tuple[int, int]:
        """
        Calculates the padding amounts for a single dimension.

        Args:
            size (int): The current size of the dimension.

        Returns:
            tuple[int, int]: A tuple containing the padding for the left/top
                             and right/bottom sides of the dimension.
        """
        # Calculate the new size by rounding up to the nearest multiple.
        new_size = math.ceil(size / self.multiple) * self.multiple

        # Calculate the total padding needed.
        pad_size = new_size - size

        # Distribute the padding as evenly as possible.
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left

        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor.

        This method calculates the necessary padding for the height and width
        dimensions of the input tensor and applies it symmetrically. The padding
        is performed in an inference mode, meaning no gradients are computed,
        which is efficient for this type of preprocessing.

        Args:
            x (torch.Tensor): The input tensor to be padded, with shape
                              `[..., H, W]`.

        Returns:
            torch.Tensor: The padded tensor with height and width dimensions
                          that are multiples of `self.multiple`.
        """
        # Calculate padding for height and width, in reverse order (W, H)
        # to match the F.pad format [left, right, top, bottom].
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))

        # Apply padding using replicate mode to fill with existing boundary values.
        # F.pad expects the padding to be applied to the last dimensions first.
        return F.pad(x, pads, mode='replicate')
