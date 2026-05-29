# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer"""

import math
from typing import Optional, Type

import torch
from torch import nn
from timm.layers import GluMlp, Mlp
from timm.models.vision_transformer import VisionTransformer

from nvidia_tao_pytorch.ssl.nvdinov2.model.layers.block import DropPathBlock, NestedTensorBlock


class SwiGLUFused(GluMlp):
    """A Fused SwiGLU that reduce the hidden features by 1/3 and round to the nearest multiple of 8"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        **kwargs,
    ):
        """
        SwiGLUFused module.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features. If None, defaults to in_features.
            out_features (Optional[int]): Number of output features. If None, defaults to in_features.
            act_layer (Callable, optional): Activation layer to use. Defaults to nn.SiLU.
            norm_layer (Optional[Callable], optional): Normalization layer to use. Defaults to None.
            bias (bool, optional): Whether to use bias in the layers. Defaults to True.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features * 2 / 3) + 7
        hidden_features -= hidden_features % 8
        hidden_features *= 2  # for timm's GluMlp

        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            norm_layer=norm_layer,
            bias=bias,
            drop=drop,
            gate_last=False,
            **kwargs,  # forward device/dtype kwargs added in timm 1.x
        )


class DinoV2VisionTransformer(VisionTransformer):
    """Vision Transformer for NVDINOv2"""

    def __init__(
        self,
        *args,
        block_fn: Type[nn.Module] = NestedTensorBlock,
        drop_path_schedule: str = "uniform",
        register_tokens: int = 4,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.SiLU,
        mlp_layer: Type[nn.Module] = Mlp,
        use_custom_attention: bool = True,
        **kwargs,
    ):
        """
        Vision Transformer for NVDINOv2

        Args:
            *args: Variable length argument list for the parent class.
            block_fn (Type[nn.Module]): Block type to use in the transformer. Defaults to NestedTensorBlock.
            drop_path_schedule (str): Schedule for dropout path. Default is 'uniform'.
            register_tokens (int): Number of register tokens to be added. Default is 4.
            embed_dim (int): Dimensionality of the embedding space. Default is 1024.
            depth (int): Number of layers in the transformer. Default is 24.
            num_heads (int): Number of attention heads in the transformer. Default is 16.
            mlp_ratio (float): Ratio of the hidden dimension in the MLP to the embedding dimension. Default is 4.0.
            qkv_bias (bool): Whether to add a bias term to Q, K, and V. Default is True.
            qk_norm (bool): Whether to apply normalization to Q and K. Default is False.
            init_values (Optional[float]): Initial values for layers. Default is None.
            proj_drop_rate (float): Dropout rate for the projection layer. Default is 0.0.
            attn_drop_rate (float): Dropout rate for the attention layer. Default is 0.0.
            drop_path_rate (float): Dropout rate for the drop path layer. Default is 0.0.
            norm_layer (Type[nn.Module]): Normalization layer to use. Default is nn.LayerNorm.
            act_layer (Type[nn.Module]): Activation layer to use. Default is nn.SiLU.
            mlp_layer (Type[nn.Module]): MLP layer to use. Default is Mlp.
            use_custom_attention (bool): Whether to use memory_efficient_attention.
            **kwargs: Additional keyword arguments for the parent class.
        """
        assert block_fn in [NestedTensorBlock, DropPathBlock], f"Invalid block_fn: {block_fn}. Must be one of [NestedTensorBlock, DropPathBlock]."

        super().__init__(
            *args,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            **kwargs,
        )
        self.patch_embed.img_size = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.num_register_tokens = register_tokens
        if register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, register_tokens, self.embed_dim)
            )

        if drop_path_schedule == 'linear':
            # stochastic depth decay rule
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_schedule == 'uniform':
            dpr = [drop_path_rate] * depth

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                use_custom_attention=use_custom_attention
            )
            for i in range(depth)])

        self.n_blocks = len(self.blocks)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate position embeddings"""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8

        w0, h0 = w0 + 0.1, h0 + 0.1

        # We need fp32 for the interpolation
        reshaped_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def patch_pos_embed(self, x, masks=None):
        """Generate patch embeddings with positional encoding."""
        B, _, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling (B, HW, C)
        if masks is not None:
            x = torch.where(masks[..., None], self.mask_token.to(x.dtype), x)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        # add register tokens
        if self.num_register_tokens > 0:
            x = torch.cat((x, self.register_tokens.expand(B, -1, -1)), dim=1)

        return self.pos_drop(x)

    def forward(
        self, x, masks=None, keep_last_n_layers=None, keep_level: str = "chunk"
    ):
        """Forward the input to get features."""
        if isinstance(x, (tuple, list)):
            x = [
                self.norm_pre(self.patch_drop(self.patch_pos_embed(i, masks=j)))
                for i, j in zip(x, masks)
            ]

            x = self.blocks(x)

            all_x = x
            output = []
            for x_, mask in zip(all_x, masks):
                # Remove register tokens
                if self.num_register_tokens > 0:
                    x_ = x_[:, : -self.num_register_tokens]

                x_norm = self.norm(x_)
                output.append(
                    {
                        "x_norm_clstoken": x_norm[:, 0],
                        "x_norm_patchtokens": x_norm[:, 1:],
                        "prenorm_x": x_,
                        "masks": mask,
                    }
                )
            return output

        x = self.patch_pos_embed(x, masks=masks)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []

        if keep_last_n_layers is None:
            x = self.blocks(x)
        else:
            x = self.blocks[:-keep_last_n_layers](x)

            for module in self.blocks[-keep_last_n_layers:]:
                x = module(x)
                features.append(self.norm(x))

            assert (
                keep_last_n_layers is None or len(features) == keep_last_n_layers
            ), f"len(features)={len(features)} != keep_last_n_layers={keep_last_n_layers}"

        # Remove register tokens
        if self.num_register_tokens > 0:
            x = x[:, : -self.num_register_tokens]

        x_norm = self.norm(x)

        return {
            "features": features,
            "prenorm_x": x,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "masks": masks,
        }
