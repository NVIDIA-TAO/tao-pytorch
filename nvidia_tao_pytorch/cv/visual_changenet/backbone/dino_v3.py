# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ViT-Adapter backbone for Visual ChangeNet.

DINOv3 timm models are ``timm.models.eva.Eva`` instances that use rotary position
embeddings (RoPE) instead of a learned ``pos_embed`` and carry ``num_prefix_tokens``
(1 cls + 4 register) prefix tokens. The RoPE tensor is computed once per forward from
the patch grid and applied inside every block to the patch tokens, so the existing
``ViTAdapter``/``CRADIOAdapter`` block-iteration paths (which are RoPE-unaware) cannot
be reused directly. ``DINOV3InteractionBlock`` keeps the prefix tokens alongside the
patch tokens and forwards the RoPE tensor into each block; the prefix (summary) tokens
are optionally fused into the feature pyramid exactly like ``CRADIOAdapter`` does for
RADIO summary tokens.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_, trunc_normal_

from nvidia_tao_pytorch.cv.backbone_v2.dino_v3 import (
    create_dinov3_model,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vits16,
    dinov3_vits16plus,
    validate_dinov3_checkpoint,
)
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import FrozenBatchNorm2d
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.visual_changenet.backbone.adapter_modules import (
    Extractor,
    Injector,
    SpatialPriorModule,
    deform_inputs,
)


def _freeze_bn_norm(module):
    """Recursively replace BatchNorm2d with FrozenBatchNorm2d.

    Mirrors ``BackboneBase._freeze_bn_norm`` (an instance method, so not directly
    reusable here: the wrapped timm model is not a ``BackboneBase``); replacement
    survives later ``.train()`` calls, matching ``CRADIOAdapter``'s export mode.
    """
    if isinstance(module, nn.BatchNorm2d):
        return FrozenBatchNorm2d(module.num_features)
    for name, child in module.named_children():
        new_child = _freeze_bn_norm(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module


class DINOV3InteractionBlock(nn.Module):
    """ViT-Adapter InteractionBlock for RoPE-based DINOv3 (timm Eva) blocks.

    Mirrors ``adapter_modules.InteractionBlock`` but (1) keeps the ``num_summary``
    prefix tokens (cls + register) alongside the patch tokens while running the ViT
    blocks and (2) forwards the RoPE tensor into each block. The Injector/Extractor
    only operate on the patch tokens, matching the standard ViT-Adapter design.
    """

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        """Initialize the DINOv3 interaction block.

        Args:
            dim (int): The feature dimension.
            num_heads (int): Parallel attention heads of MultiScaleDeformableAttention.
            n_points (int): The number of sampling points for each query in each head.
            norm_layer (nn.Module): norm layer.
            drop (float): Dropout probability after the feed forward layer.
            drop_path (float): Stochastic depth rate.
            with_cffn (bool): The option to use ConvFFN for the extractor.
            cffn_ratio (float): Expansion ratio of the ConvFFN hidden layer channels.
            init_values (float): Init value of LayerScale in the injector.
            deform_ratio (float): The expansion ratio of value_proj in the deform attention.
            extra_extractor (bool): Whether to append extra extractors in this block.
            with_cp (bool): Use activation checkpointing in the injector/extractor.
        """
        super().__init__()

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
            self.extra_extractors = nn.Sequential(
                *[
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
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, rope=None, num_summary=0):
        """Run the injector, a slice of RoPE ViT blocks and the extractor.

        Args:
            x (torch.Tensor): tokens of shape [bs, num_summary + H*W, dim] (prefix + patches).
            c (torch.Tensor): adapter (SPM) features for the injector.
            blocks (nn.Module): slice of DINOv3 (Eva) transformer blocks.
            deform_inputs1 (list): deform inputs for the injector.
            deform_inputs2 (list): deform inputs for the extractor.
            H (int): feature height (patch grid rows).
            W (int): feature width (patch grid cols).
            rope (torch.Tensor): rotary position embedding for the patch tokens.
            num_summary (int): number of prefix tokens (cls + register) to keep untouched.

        Returns:
            tuple: (updated tokens, updated adapter features)
        """
        x_summary = x[:, :num_summary]
        x_feat = x[:, num_summary:]
        x_feat = self.injector(
            query=x_feat,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        x = torch.cat([x_summary, x_feat], dim=1)
        for blk in blocks:
            x = blk(x, rope=rope)
        x_feat = x[:, num_summary:]

        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x_feat,
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
                    feat=x_feat,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c


class DINOV3Adapter(nn.Module):
    """ViT-Adapter (https://arxiv.org/abs/2205.08534) wrapping a timm DINOv3 (Eva) model.

    Structured like ``CRADIOAdapter`` (SPM + interactions + optional summary-token fusion
    into every pyramid level) but drives the RoPE-based DINOv3 blocks through
    ``DINOV3InteractionBlock`` and loads bare timm checkpoints into the wrapped
    ``self.vit`` model.
    """

    def __init__(
        self,
        dino_model,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        drop_path_rate=0.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        out_indices=None,
        activation_checkpoint=False,
        add_summary=True,
        freeze_at=None,
        export=False,
        **kwargs,
    ):
        """DINOv3 ViT-Adapter constructor.

        Args:
            dino_model (nn.Module): The timm DINOv3 (Eva) model to wrap.
            conv_inplane (int): The hidden dimension of Conv2D in the SPM.
            n_points (int): Number of sampling points per query in the deform attention.
            deform_num_heads (int): Parallel attention heads of the deform attention.
            init_values (float): Init value of LayerScale in the injector.
            interaction_indexes (list): The block-index ranges of each interaction block.
            with_cffn (bool): Whether to use ConvFFN in the extractor.
            cffn_ratio (float): Expansion ratio of the ConvFFN hidden layer channels.
            deform_ratio (float): The expansion ratio of value_proj in the deform attention.
            drop_path_rate (float): Stochastic depth rate for the interaction blocks.
            add_vit_feature (bool): Whether to add ViT features to the adapter features.
            use_extra_extractor (bool): Whether to use extra extractors in the last block.
            out_indices (list): List of SPM output indices to return as feature.
            activation_checkpoint (bool): Use activation checkpointing in the interactions.
            add_summary (bool): Fuse the prefix (cls + register) tokens into every pyramid
                level, mirroring ``CRADIOAdapter``'s summary-token handling.
            freeze_at (str or list): ``"all"`` freezes the wrapped ViT (the adapter modules
                stay trainable, matching the frozen-backbone recipes of the other adapters).
            export (bool): Whether to enable export mode (freeze BatchNorm statistics).
        """
        super().__init__()

        if out_indices is None:
            out_indices = [0, 1, 2, 3]
        self.vit = dino_model
        self.patch_size = dino_model.patch_embed.patch_size[0]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_summary = add_summary
        self.num_summary = dino_model.num_prefix_tokens
        embed_dim = dino_model.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(in_channel=3,
                                      patch_size=self.patch_size,
                                      inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            DINOV3InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                   init_values=init_values, drop_path=drop_path_rate,
                                   norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                   extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                                   with_cp=activation_checkpoint)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        if self.add_summary:
            self.fc_summary = nn.Linear(self.num_summary * embed_dim, embed_dim)
            self.conv1 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv2 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv3 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv4 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv1.apply(self._init_weights)
            self.conv2.apply(self._init_weights)
            self.conv3.apply(self._init_weights)
            self.conv4.apply(self._init_weights)

        if freeze_at == "all":
            # Frozen-backbone recipe: only the wrapped ViT is frozen; the adapter
            # modules (SPM / interactions / norms / summary fusion) stay trainable.
            for param in self.vit.parameters():
                param.requires_grad = False

        if export:
            _freeze_bn_norm(self)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _get_patch_embed(self, x):
        """Compute patch embedding, prefix tokens and RoPE for the input.

        Args:
            x (torch.Tensor): input image tensor [B, 3, H, W].

        Returns:
            tuple: (tokens [B, num_summary + Hp*Wp, C], rope, Hp, Wp)
        """
        feat = self.vit.patch_embed(x)  # [B, Hp, Wp, C] (dynamic_img_size)
        patch_h, patch_w = feat.shape[1], feat.shape[2]
        tokens, rope = self.vit._pos_embed(feat)  # prefix tokens prepended; rope for patches
        tokens = self.vit.norm_pre(tokens)
        return tokens, rope, patch_h, patch_w

    def forward_feature_pyramid(self, x):
        """Forward function returning the 4-level feature pyramid."""
        deform_inputs1, deform_inputs2 = deform_inputs(x, patch_size=self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        _, _, H, W = x.shape
        H, W = H // self.patch_size, W // self.patch_size

        # Patch embedding + RoPE (prefix tokens kept for RoPE alignment).
        x_tokens, rope, patch_h, patch_w = self._get_patch_embed(x)
        bs, _, dim = x_tokens.shape
        num_summary = self.num_summary

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blocks = self.vit.blocks[indexes[0]:indexes[-1] + 1]
            x_tokens, c = layer(x_tokens, c, blocks, deform_inputs1, deform_inputs2,
                                H, W, rope=rope, num_summary=num_summary)
            patch_tokens = x_tokens[:, num_summary:]
            outs.append(patch_tokens.transpose(1, 2).view(bs, dim, patch_h, patch_w).contiguous())

        # Split & Reshape
        n2, n3 = c2.size(1), c3.size(1)
        c2 = c[:, 0:n2, :]
        c3 = c[:, n2:n2 + n3, :]
        c4 = c[:, n2 + n3:, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        if self.add_summary:
            summary = x_tokens[:, :num_summary].reshape(bs, -1)
            summary = self.fc_summary(summary)
            summary = summary.unsqueeze(2).unsqueeze(3)
            c1 = torch.cat([summary.expand(-1, -1, c1.shape[2], c1.shape[3]), c1], dim=1)
            c2 = torch.cat([summary.expand(-1, -1, c2.shape[2], c2.shape[3]), c2], dim=1)
            c3 = torch.cat([summary.expand(-1, -1, c3.shape[2], c3.shape[3]), c3], dim=1)
            c4 = torch.cat([summary.expand(-1, -1, c4.shape[2], c4.shape[3]), c4], dim=1)
            c1 = self.conv1(c1)
            c2 = self.conv2(c2)
            c3 = self.conv3(c3)
            c4 = self.conv4(c4)

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def forward(self, x):
        """Return the multi-scale feature pyramid."""
        return self.forward_feature_pyramid(x)

    def load_state_dict(self, state_dict, **kwargs):
        """Route a bare timm DINOv3 state dict into the wrapped ``self.vit`` model.

        Pretrained / SSL->timm-converted DINOv3 checkpoints use bare timm keys
        (``blocks.*``, ``cls_token``, ``reg_token``, ``patch_embed.*``, ``norm.*``).
        The wrapped model lives under ``self.vit``, so prefix those keys with ``vit.``.
        The prefixing is all-or-nothing: it only triggers when no key is already an
        adapter/``vit.`` key.
        """
        validate_dinov3_checkpoint(
            state_dict,
            allow_partial=not kwargs.get("strict", True),
        )
        adapter_prefixes = (
            "vit.", "spm.", "interactions.", "level_embed", "up.",
            "norm1.", "norm2.", "norm3.", "norm4.",
            "fc_summary.", "conv1.", "conv2.", "conv3.", "conv4.",
        )
        if state_dict and not any(k.startswith(adapter_prefixes) for k in state_dict):
            state_dict = {f"vit.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, **kwargs)


def _make_dinov3_adapter(
    timm_name, interaction_indexes,
    out_indices=None, resolution=224, activation_checkpoint=False,
    use_summary_token=True, pretrained=False, **kwargs
):
    """Build a DINOv3 ViT-Adapter with the family-shared adapter hyper-parameters.

    Setting pretrained fetches timm's matching Hugging Face weights. Otherwise
    Visual ChangeNet loads an explicit converted checkpoint after construction.
    """
    dino_model = create_dinov3_model(
        timm_name,
        pretrained=pretrained,
        img_size=resolution,
    )
    return DINOV3Adapter(
        dino_model,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=interaction_indexes,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


# Interaction ranges: the ViT blocks are split into 4 equal slices (ViT-Adapter convention).
_IDX_12 = [[0, 2], [3, 5], [6, 8], [9, 11]]
_IDX_24 = [[0, 5], [6, 11], [12, 17], [18, 23]]
_IDX_32 = [[0, 7], [8, 15], [16, 23], [24, 31]]
_IDX_40 = [[0, 9], [10, 19], [20, 29], [30, 39]]


def vit_small_dinov3(**kwargs):
    """DINOv3 ViT-Small/16 ViT-Adapter (embed_dim 384, depth 12)."""
    return _make_dinov3_adapter("vit_small_patch16_dinov3.lvd1689m", _IDX_12, **kwargs)


def vit_small_plus_dinov3(**kwargs):
    """DINOv3 ViT-Small+/16 ViT-Adapter (embed_dim 384, depth 12, SwiGLU)."""
    return _make_dinov3_adapter("vit_small_plus_patch16_dinov3.lvd1689m", _IDX_12, **kwargs)


def vit_base_dinov3(**kwargs):
    """DINOv3 ViT-Base/16 ViT-Adapter (embed_dim 768, depth 12)."""
    return _make_dinov3_adapter("vit_base_patch16_dinov3.lvd1689m", _IDX_12, **kwargs)


def vit_large_dinov3(**kwargs):
    """DINOv3 ViT-Large/16 ViT-Adapter (embed_dim 1024, depth 24)."""
    return _make_dinov3_adapter("vit_large_patch16_dinov3.lvd1689m", _IDX_24, **kwargs)


def vit_huge_plus_dinov3(**kwargs):
    """DINOv3 ViT-Huge+/16 ViT-Adapter (embed_dim 1280, depth 32, SwiGLU)."""
    return _make_dinov3_adapter("vit_huge_plus_patch16_dinov3.lvd1689m", _IDX_32, **kwargs)


def vit_7b_dinov3(**kwargs):
    """DINOv3 ViT-7B/16 ViT-Adapter (embed_dim 4096, depth 40, SwiGLU)."""
    return _make_dinov3_adapter("vit_7b_patch16_dinov3.lvd1689m", _IDX_40, **kwargs)


# Arch 1 (euclidean difference): the DINOV3Wrapper cls-token backbone from backbone_v2,
# mirroring vit_model_dict in dino_v2.py. With num_classes=0, forward(x) returns the
# (B, embed_dim) cls token, matching ChangeNetClassify's fc_ip_dim = embed_dims[-1].
dinov3_model_dict = {
    "vit_small_dinov3": partial(dinov3_vits16, num_classes=0),
    "vit_small_plus_dinov3": partial(dinov3_vits16plus, num_classes=0),
    "vit_base_dinov3": partial(dinov3_vitb16, num_classes=0),
    "vit_large_dinov3": partial(dinov3_vitl16, num_classes=0),
    "vit_huge_plus_dinov3": partial(dinov3_vith16plus, num_classes=0),
    "vit_7b_dinov3": partial(dinov3_vit7b16, num_classes=0),
}
