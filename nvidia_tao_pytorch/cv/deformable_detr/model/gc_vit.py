
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Backbone GCViT model definition. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath

from nvidia_tao_pytorch.cv.backbone.gc_vit import (
    _to_channel_first, WindowAttentionGlobal, Mlp,
    WindowAttention, PatchEmbed, ReduceSize, GlobalQueryGen
)


def window_partition(x, window_size):
    """Window partions.

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Window reversal.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # Casting to int leads to error
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class GCViTBlock(nn.Module):
    """GCViT block based on: "Hatamizadeh et al.,

    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_rel_pos_bias=False
                 ):
        """Initialize GCViT Block.

        Args:
            dim (int): feature size dimension.
            num_heads (int): number of heads in each stage.
            window_size (int): window size in each stage.
            mlp_ratio (float): MLP ratio.
            qkv_bias (bool): bool argument for query, key, value learnable bias.
            qk_scale (bool): bool argument to scaling query, key.
            drop (float): dropout rate.
            attn_drop (float): attention dropout rate.
            drop_path (float): drop path rate.
            act_layer (nn.Module): type of activation layer.
            attention (nn.Module): type of attention layer
            norm_layer (nn.Module): normalization layer.
            layer_scale (float): layer scaling coefficient.
            use_rel_pos_bias (bool): whether to use relative positional bias.
        """
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              use_rel_pos_bias=use_rel_pos_bias
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

    def forward(self, x, q_global):
        """Forward function."""
        _, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GCViTLayer(nn.Module):
    """GCViT layer based on: "Hatamizadeh et al.,

    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 image_resolution,
                 num_heads,
                 window_size,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_rel_pos_bias=False):
        """Initialize GCViT Layer.

        Args:
            dim (int): feature size dimension.
            depths (int): number of layers in each stage.
            input_resolution (int): input image resolution
            image_resolution (int): input image resolution
            num_heads (int): number of heads in each stage.
            window_size (tuple): window size in each stage.
            downsample (bool): bool argument to downsample.
            mlp_ratio (float): MLP ratio.
            qkv_bias (bool): bool argument for query, key, value learnable bias.
            qk_scale (bool): bool argument to scaling query, key.
            attn_drop (float): attention dropout rate.
            drop (float): dropout rate.
            drop_path (float): drop path rate.
            norm_layer (nn.Module): normalization layer.
            layer_scale (float): layer scaling coefficient.
            use_rel_pos_bias (bool): whether to use relative positional bias.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale,
                       use_rel_pos_bias=use_rel_pos_bias)
            for i in range(depth)])
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def forward(self, x):
        """Foward function."""
        q_global = self.q_global_gen(_to_channel_first(x))
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class GCViT(nn.Module):
    """GCViT model based on: "Hatamizadeh et al.,

    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depths,
                 mlp_ratio,
                 num_heads,
                 window_size=(7, 7, 14, 7),
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_rel_pos_bias=True,
                 activation_checkpoint=True,
                 **kwargs):
        """Initialize GCViT model.

        Args:
            dim (int): feature size dimension.
            depths (int): number of layers in each stage.
            mlp_ratio (float): MLP ratio.
            num_heads (int): number of heads in each stage.
            window_size (tuple): window size in each stage.
            resolution (int): input image resolution
            drop_path_rate (float): drop path rate.
            qkv_bias (bool): bool argument for query, key, value learnable bias.
            qk_scale (bool): bool argument to scaling query, key.
            drop_rate (float): dropout rate.
            attn_drop_rate (float): attention dropout rate.
            norm_layer (nn.Module): normalization layer.
            layer_scale (float): layer scaling coefficient.
            out_indices (list): list of block indices to return as feature.
            frozen_stages (int): stage to freeze.
            use_rel_pos_bias (bool): whether to use relative positional bias.
            activation_checkpoint (bool): bool argument for activiation checkpointing.
        """
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = dim
        self.num_features = [int(dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio
        self.activation_checkpoint = activation_checkpoint
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution,
                               use_rel_pos_bias=use_rel_pos_bias)
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages

        for level in self.levels:
            for block in level.blocks:
                w_ = block.attn.window_size[0]
                relative_position_bias_table_pre = block.attn.relative_position_bias_table
                L1, nH1 = relative_position_bias_table_pre.shape
                L2 = (2 * w_ - 1) * (2 * w_ - 1)
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pre.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                relative_position_bias_table_pretrained_resized = relative_position_bias_table_pretrained_resized.view(nH1, L2).permute(1, 0)
                block.attn.relative_position_bias_table = torch.nn.Parameter(relative_position_bias_table_pretrained_resized)

    def _freeze_stages(self):
        """Freeze some blocks"""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Freeze some blocks during training"""
        super(GCViT, self).train(mode)
        self._freeze_stages()

    def forward_embeddings(self, x):
        """Compute patch embedding"""
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        """Returns features with normalization"""
        outs = {}
        for idx, level in enumerate(self.levels):
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x, xo = level(x)
            else:
                x, xo = checkpoint.checkpoint(level, x, use_reentrant=True)

            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(xo)
                outs[f'p{idx}'] = x_out.permute(0, 3, 1, 2).contiguous()
        return outs

    def forward(self, x):
        """Forward function"""
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

    def forward_features(self, x):
        """Extract features"""
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)


def gc_vit_xxtiny(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-XXTiny model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[2, 2, 6, 2],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_xtiny(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-XTiny model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 6, 5],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_tiny(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-Tiny model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_small(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-Small model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[3, 6, 12, 24],
                  window_size=[7, 7, 14, 7],
                  dim=96,
                  mlp_ratio=2,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  layer_scale=1e-5,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_base(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-Base model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[4, 8, 16, 32],
                  window_size=[7, 7, 14, 7],
                  dim=128,
                  mlp_ratio=2,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  layer_scale=1e-5,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_large(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-Large model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[6, 12, 24, 48],
                  window_size=[7, 7, 14, 7],
                  dim=192,
                  mlp_ratio=2,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  layer_scale=1e-5,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


def gc_vit_large_384(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """GCViT-Large Input Resolution 384 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.
    """
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[6, 12, 24, 48],
                  window_size=[12, 12, 24, 12],
                  dim=192,
                  mlp_ratio=2,
                  drop_path_rate=0.2,
                  out_indices=out_indices,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  frozen_stages=-1,
                  layer_scale=1e-5,
                  activation_checkpoint=activation_checkpoint,
                  **kwargs)

    return model


gc_vit_model_dict = {
    'gc_vit_xxtiny': gc_vit_xxtiny,
    'gc_vit_xtiny': gc_vit_xtiny,
    'gc_vit_tiny': gc_vit_tiny,
    'gc_vit_small': gc_vit_small,
    'gc_vit_base': gc_vit_base,
    'gc_vit_large': gc_vit_large,
    'gc_vit_large_384': gc_vit_large_384,
}
