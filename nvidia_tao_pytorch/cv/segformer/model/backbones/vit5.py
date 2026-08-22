# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT-5 backbone and ViT-Adapter feature pyramid for SegFormer.

The ViT-5 architecture and checkpoint-compatible module names are adapted from
https://github.com/wangf3014/ViT-5 (Apache-2.0).  The dense adapter is TAO-specific:
it keeps ViT-5's ``[CLS, image patches, registers]`` token order, interpolates the
released 224-pixel absolute position embedding for larger inputs, and couples the
patch tokens to the four-scale ViT-Adapter spatial prior.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, PatchEmbed, trunc_normal_
from torch import nn
from torch.nn.init import normal_
from torch.utils.checkpoint import checkpoint

from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.segformer.model.backbones.adapter_modules import (
    Extractor,
    Injector,
    SpatialPriorModule,
    deform_inputs,
)


def _rotate_half(x):
    """Rotate adjacent feature pairs by 90 degrees for RoPE."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbedding(nn.Module):
    """Device-safe two-dimensional rotary embedding used by ViT-5."""

    def __init__(self, dim, pt_seq_len=14, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.pt_seq_len = pt_seq_len
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        seq_len = x.shape[1]
        grid = math.isqrt(seq_len)
        if grid * grid != seq_len:
            raise ValueError(f"ViT-5 RoPE requires a square patch grid, got {seq_len} tokens")
        t = torch.arange(grid, device=x.device, dtype=torch.float32)
        t = t / grid * self.pt_seq_len
        freqs = torch.einsum("i,j->ij", t, self.freqs.float())
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        y_freqs = freqs[:, None, :].expand(grid, grid, -1)
        x_freqs = freqs[None, :, :].expand(grid, grid, -1)
        freqs = torch.cat((y_freqs, x_freqs), dim=-1).reshape(seq_len, 1, -1)
        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)
        return x * cos + _rotate_half(x) * sin


class RMSNorm(nn.Module):
    """Checkpoint-compatible ViT-5 RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ViT5Attention(nn.Module):
    """ViT-5 attention using memory-efficient PyTorch SDPA at 1024 resolution."""

    def __init__(
        self,
        dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_size=14,
        num_registers=4,
        reg_theta=100.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_registers = num_registers
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = VisionRotaryEmbedding(head_dim // 2, rope_size)
        self.rope_reg = VisionRotaryEmbedding(
            head_dim // 2, math.isqrt(num_registers), theta=reg_theta
        )
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        batch, tokens, channels = x.shape
        reg_idx = tokens - self.num_registers
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.unbind(dim=2)
        q_dtype = q.dtype
        q = self.q_norm(q).to(q_dtype)
        k = self.k_norm(k).to(q_dtype)
        q = torch.cat((q[:, :1], self.rope(q[:, 1:reg_idx]), q[:, reg_idx:]), dim=1)
        k = torch.cat((k[:, :1], self.rope(k[:, 1:reg_idx]), k[:, reg_idx:]), dim=1)
        q = torch.cat((q[:, :reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1)
        k = torch.cat((k[:, :reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class ViT5Block(nn.Module):
    """Checkpoint-compatible ViT-5 transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=1e-4):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = ViT5Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ViT5InteractionBlock(nn.Module):
    """ViT-Adapter interaction that preserves ViT-5's trailing registers."""

    def __init__(
        self,
        dim,
        num_heads=16,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.ModuleList([
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    with_cp=with_cp,
                )
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(
        self,
        x,
        c,
        blocks,
        deform_inputs1,
        deform_inputs2,
        H,
        W,
        num_registers=4,
    ):
        cls_token = x[:, :1]
        patch_tokens = x[:, 1:-num_registers]
        registers = x[:, -num_registers:]
        patch_tokens = self.injector(
            query=patch_tokens,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        x = torch.cat((cls_token, patch_tokens, registers), dim=1)
        for block in blocks:
            if self.with_cp and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        patch_tokens = x[:, 1:-num_registers]
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=patch_tokens,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=patch_tokens,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c


class ViT5Adapter(BackboneBase):
    """ViT-5 encoder coupled to a four-scale ViT-Adapter pyramid."""

    def __init__(
        self,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pretrained_resolution=224,
        resolution=1024,
        patch_size=16,
        num_registers=4,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=None,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=(0, 1, 2, 3),
        activation_checkpoint=False,
        freeze_at=None,
        **kwargs,
    ):
        if isinstance(resolution, tuple):
            if resolution[0] != resolution[1]:
                raise ValueError("ViT-5 currently requires square inputs")
            resolution = resolution[0]
        if resolution % 32 != 0:
            raise ValueError(f"Input resolution ({resolution}) must be divisible by 32")
        super().__init__(
            in_chans=3,
            num_classes=0,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.pretrained_grid_size = pretrained_resolution // patch_size
        self.interaction_indexes = interaction_indexes or [
            [0, 5], [6, 11], [12, 17], [18, 23]
        ]
        self.patch_embed = PatchEmbed(
            img_size=pretrained_resolution,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            strict_img_size=False,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            ViT5Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.reg_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_vit_weights)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        normal_(self.level_embed)
        self.spm = SpatialPriorModule(
            in_channel=3,
            patch_size=patch_size,
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            out_indices=return_idx,
        )
        self.spm.apply(self._init_adapter_weights)
        self.interactions = nn.ModuleList([
            ViT5InteractionBlock(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=(i == len(self.interaction_indexes) - 1),
                with_cp=activation_checkpoint,
            )
            for i in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_adapter_weights)
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up.apply(self._init_adapter_weights)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)
        self.apply(self._init_deform_weights)

    def _init_vit_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_adapter_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(module):
        if isinstance(module, MSDeformAttn):
            module._reset_parameters()

    def get_stage_dict(self):
        stages = {0: self.patch_embed}
        stages.update({index + 1: block for index, block in enumerate(self.blocks)})
        return stages

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "reg_token"}

    def freeze_backbone(self):
        """Freeze the pretrained encoder while retaining trainable dense adapters."""
        super().freeze_backbone()
        if self.freeze_at == "all":
            self.level_embed.requires_grad = True
            adapter_modules = (
                self.spm,
                self.interactions,
                self.up,
                self.norm1,
                self.norm2,
                self.norm3,
                self.norm4,
            )
            for module in adapter_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
                module.train()

    def load_state_dict(self, state_dict, **kwargs):
        """Load an official ViT-5 checkpoint while leaving adapters initialized."""
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        return nn.Module.load_state_dict(self, state_dict, **kwargs)

    def _position_embedding(self, patch_h, patch_w):
        if (patch_h, patch_w) == (self.pretrained_grid_size, self.pretrained_grid_size):
            return self.pos_embed
        pos_embed = self.pos_embed.reshape(
            1, self.pretrained_grid_size, self.pretrained_grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        return F.interpolate(
            pos_embed,
            size=(patch_h, patch_w),
            mode="bicubic",
            align_corners=False,
        ).flatten(2).transpose(1, 2)

    def _patch_tokens(self, image):
        patches = self.patch_embed(image)
        patch_h = image.shape[-2] // self.patch_size
        patch_w = image.shape[-1] // self.patch_size
        patches = patches + self._position_embedding(patch_h, patch_w)
        batch = patches.shape[0]
        cls_token = self.cls_token.expand(batch, -1, -1)
        registers = self.reg_token.expand(batch, -1, -1)
        return torch.cat((cls_token, patches, registers), dim=1), patch_h, patch_w

    def _add_level_embed(self, c2, c3, c4):
        return (
            c2 + self.level_embed[0],
            c3 + self.level_embed[1],
            c4 + self.level_embed[2],
        )

    def forward_feature_pyramid(self, image, indices=None, **kwargs):
        deform_inputs1, deform_inputs2 = deform_inputs(image, patch_size=self.patch_size)
        c1, c2, c3, c4 = self.spm(image)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat((c2, c3, c4), dim=1)
        height, width = image.shape[-2] // 16, image.shape[-1] // 16
        x, patch_h, patch_w = self._patch_tokens(image)
        batch = x.shape[0]
        outputs = []
        for interaction, indexes in zip(self.interactions, self.interaction_indexes):
            x, c = interaction(
                x,
                c,
                self.blocks[indexes[0]:indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                height,
                width,
                num_registers=self.num_registers,
            )
            patches = x[:, 1:-self.num_registers]
            outputs.append(
                patches.transpose(1, 2).reshape(
                    batch, self.embed_dim, patch_h, patch_w
                ).contiguous()
            )
        c2_len, c3_len = c2.size(1), c3.size(1)
        c2 = c[:, :c2_len].transpose(1, 2).reshape(
            batch, self.embed_dim, height * 2, width * 2
        ).contiguous()
        c3 = c[:, c2_len:c2_len + c3_len].transpose(1, 2).reshape(
            batch, self.embed_dim, height, width
        ).contiguous()
        c4 = c[:, c2_len + c3_len:].transpose(1, 2).reshape(
            batch, self.embed_dim, height // 2, width // 2
        ).contiguous()
        c1 = self.up(c2) + c1
        target_features = (c1, c2, c3, c4)
        target_features = tuple(
            target + F.interpolate(
                source, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            for target, source in zip(target_features, outputs)
        )
        return [
            norm(feature)
            for norm, feature in zip(
                (self.norm1, self.norm2, self.norm3, self.norm4), target_features
            )
        ]

    def forward_pre_logits(self, x):
        return self.forward_feature_pyramid(x)[-1]

    def forward(self, x):
        return self.forward_feature_pyramid(x)


def vit5_large_patch16_224(
    return_idx=(0, 1, 2, 3),
    resolution=1024,
    freeze_at=None,
    activation_checkpoint=False,
    **kwargs,
):
    """ViT-5 Large/16 using the released 224-pixel ImageNet-1K checkpoint."""
    return ViT5Adapter(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pretrained_resolution=224,
        resolution=resolution,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )
