# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Vision Transformer.

Strategy A: subclass ``nvdinov2``'s :class:`DinoV2VisionTransformer` and change only what
DINOv3 requires:

* **No absolute positional embedding.** ``pos_embed`` is dropped; ``patch_pos_embed`` no
  longer adds it and ``interpolate_pos_encoding`` is a no-op. Position is encoded by 2D
  axial RoPE inside attention.
* **RoPE blocks.** ``self.blocks`` is rebuilt with :class:`RoPENestedTensorBlock`, which
  rotates Q/K of patch tokens before the xformers memory-efficient attention call.
* **Per-type FFN.** The MLP class is passed in by the param map (``Mlp`` for ViT-B/L,
  ``SwiGLUFusedFull`` for ViT-H+/7B).

Everything else (iBOT mask token, register tokens, nested-tensor batching, FSDP wrapping,
the multi-crop ``forward`` contract returning ``x_norm_clstoken`` / ``x_norm_patchtokens``)
is inherited.

**Register-token placement (open decision #1, resolved for v1):** register tokens stay at
the **end** of the sequence — ``[CLS, patches, registers]`` — matching the inherited
``nvdinov2`` layout. This keeps ``patch_pos_embed`` / ``forward`` a minimal override and
makes the RoPE exclusion trivial (``[CLS]`` at index 0, registers at the tail). Meta/timm
DINOv3 place registers right after ``[CLS]``; the step-4 checkpoint remapper reorders them
into this end-of-sequence layout.
"""

from typing import Optional, Type

import torch
from torch import nn
from timm.layers import GluMlp, Mlp

from nvidia_tao_pytorch.ssl.nvdinov2.model.vit import DinoV2VisionTransformer
from nvidia_tao_pytorch.ssl.nvdinov2.model.layers.block import NestedTensorBlock
from nvidia_tao_pytorch.ssl.dinov3.model.layers.block import RoPENestedTensorBlock
from nvidia_tao_pytorch.ssl.dinov3.model.layers.rope import RoPE2D


class SwiGLUFusedFull(GluMlp):
    """Fused SwiGLU with the full inner width (no LLaMA-style 2/3 reduction).

    DINOv3 ViT-H+/7B use SwiGLU whose inner width is ``mlp_ratio * dim`` per branch: the public
    timm ``vit_*_patch16_dinov3`` checkpoints store ``fc1_g``/``fc1_x`` of that width and ``fc2``
    of matching in-width. timm's ``Block`` passes ``hidden_features = int(dim * mlp_ratio)`` (one
    branch's width); ``GluMlp`` expects the fused (gate + value) width, so it is doubled here and
    the gating activation is SiLU. ``gate_last=False`` gives the fused layout ``[gate, value]``,
    which the checkpoint remapper matches by concatenating ``fc1_g`` then ``fc1_x``.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        **kwargs,
    ):
        """Build a full-width fused SwiGLU (gate-first), doubling the per-branch hidden width."""
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features *= 2  # gate + value branches fused into one fc1 (no 2/3 reduction)
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=nn.SiLU,  # SwiGLU gating non-linearity (ignore the ViT MLP act_layer)
            norm_layer=norm_layer,
            bias=bias,
            drop=drop,
            gate_last=False,
            **kwargs,
        )


class DinoV3VisionTransformer(DinoV2VisionTransformer):
    """Vision Transformer for DINOv3 (RoPE, no absolute pos-embed)."""

    def __init__(
        self,
        *args,
        rope_theta: float = 100.0,
        register_tokens: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        drop_path_schedule: str = "uniform",
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        mlp_layer: Type[nn.Module] = Mlp,
        use_custom_attention: bool = True,
        **kwargs,
    ):
        """Build a DINOv3 ViT.

        Args:
            *args: Forwarded to :class:`DinoV2VisionTransformer`.
            rope_theta (float): Frequency base for 2D axial RoPE.
            register_tokens (int): Number of register tokens (ViT-B uses 4).
            embed_dim (int): Embedding dimension.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): MLP hidden ratio.
            qkv_bias (bool): Whether QKV projections use bias.
            qk_norm (bool): Whether to normalize Q and K.
            init_values (Optional[float]): LayerScale init value.
            proj_drop_rate (float): Projection dropout.
            attn_drop_rate (float): Attention dropout.
            drop_path_rate (float): Stochastic-depth rate.
            drop_path_schedule (str): 'uniform' or 'linear' drop-path schedule.
            norm_layer (Type[nn.Module]): Normalization layer.
            act_layer (Type[nn.Module]): Activation layer.
            mlp_layer (Type[nn.Module]): FFN class (Mlp for ViT-B/L, SwiGLUFusedFull for H+/7B).
            use_custom_attention (bool): Whether to use memory_efficient_attention.
            **kwargs: Forwarded to :class:`DinoV2VisionTransformer` (e.g. img_size, patch_size).
        """
        # Build the DINOv2 ViT first (NestedTensorBlock passes its attn-class guard); we then
        # replace pos_embed and the blocks with the DINOv3 (RoPE) variants below.
        super().__init__(
            *args,
            block_fn=NestedTensorBlock,
            drop_path_schedule=drop_path_schedule,
            register_tokens=register_tokens,
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
            use_custom_attention=use_custom_attention,
            **kwargs,
        )

        # DINOv3 has no absolute positional embedding (RoPE encodes position in attention).
        # Setting the registered parameter to None removes it from state_dict and the optimizer.
        self.pos_embed = None

        # 2D axial RoPE generator. CLS at index 0; registers appended at the tail.
        head_dim = embed_dim // num_heads
        self.rope = RoPE2D(
            head_dim=head_dim,
            theta=rope_theta,
            num_prefix_tokens=1,
            num_register_tokens=register_tokens,
        )

        # Rebuild blocks with the RoPE-aware NestedTensorBlock (same drop-path schedule).
        if drop_path_schedule == 'linear':
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        self.blocks = nn.Sequential(*[
            RoPENestedTensorBlock(
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
                use_custom_attention=use_custom_attention,
            )
            for i in range(depth)
        ])
        self.n_blocks = len(self.blocks)

    def interpolate_pos_encoding(self, x, w, h):
        """No-op: DINOv3 has no absolute positional embedding (RoPE handles position)."""
        return 0

    def patch_pos_embed(self, x, masks=None):
        """Embed patches and assemble the token sequence **without** additive pos-embed.

        Args:
            x (torch.Tensor): Input image batch shaped ``[B, C, H, W]``.
            masks (torch.Tensor, optional): iBOT mask-token positions.

        Returns:
            torch.Tensor: Token sequence ``[CLS, patches, registers]`` after dropout.
        """
        B, _, _, _ = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling (B, HW, C)
        if masks is not None:
            x = torch.where(masks[..., None], self.mask_token.to(x.dtype), x)

        # add the [CLS] token to the embedded patch tokens (no positional embedding added)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add register tokens at the end of the sequence
        if self.num_register_tokens > 0:
            x = torch.cat((x, self.register_tokens.expand(B, -1, -1)), dim=1)

        return self.pos_drop(x)

    def _build_rope(self, x):
        """Build the per-crop RoPE ``(sin, cos)`` table from an image batch.

        Args:
            x (torch.Tensor): Image batch shaped ``[B, C, H, W]``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(sin, cos)`` each shaped ``[N, head_dim]``
            for this crop's token sequence (fp32).
        """
        _, _, h, w = x.shape
        ps = self.patch_embed.patch_size
        grid_h = h // ps[0]
        grid_w = w // ps[1]
        return self.rope(grid_h, grid_w, x.device, torch.float32)

    def forward(self, x, masks=None, keep_last_n_layers=None, keep_level: str = "chunk"):
        """Forward pass returning DINOv2-style feature dicts, with RoPE threaded into blocks.

        Mirrors :meth:`DinoV2VisionTransformer.forward` but (a) skips additive pos-embed and
        (b) iterates the blocks manually so the per-crop RoPE table can be passed in (an
        ``nn.Sequential`` call cannot forward the extra ``rope`` argument).

        Args:
            x (torch.Tensor | list | tuple): A single image batch or a list of crop batches.
            masks (torch.Tensor | list, optional): iBOT masks (per crop for the list path).
            keep_last_n_layers (int, optional): If set, also return normed features from the
                last ``n`` blocks (single-tensor path only).
            keep_level (str): Unused; kept for signature parity.

        Returns:
            dict | list[dict]: Feature dict(s) with ``x_norm_clstoken`` /
            ``x_norm_patchtokens`` / ``prenorm_x`` / ``masks``.
        """
        if isinstance(x, (tuple, list)):
            rope_list = [self._build_rope(i) for i in x]
            x = [
                self.norm_pre(self.patch_drop(self.patch_pos_embed(i, masks=j)))
                for i, j in zip(x, masks)
            ]

            for blk in self.blocks:
                x = blk(x, rope=rope_list)

            all_x = x
            output = []
            for x_, mask in zip(all_x, masks):
                # Remove register tokens (appended at the end)
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

        rope = self._build_rope(x)
        x = self.patch_pos_embed(x, masks=masks)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []

        if keep_last_n_layers is None:
            for blk in self.blocks:
                x = blk(x, rope=rope)
        else:
            for blk in self.blocks[:-keep_last_n_layers]:
                x = blk(x, rope=rope)

            for blk in self.blocks[-keep_last_n_layers:]:
                x = blk(x, rope=rope)
                features.append(self.norm(x))

            assert (
                keep_last_n_layers is None or len(features) == keep_last_n_layers
            ), f"len(features)={len(features)} != keep_last_n_layers={keep_last_n_layers}"

        # Remove register tokens (appended at the end)
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
