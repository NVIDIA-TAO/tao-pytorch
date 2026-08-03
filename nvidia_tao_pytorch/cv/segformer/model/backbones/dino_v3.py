# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ViT-Adapter backbone for Segformer.

DINOv3 timm models are ``timm.models.eva.Eva`` instances that use rotary position
embeddings (RoPE) instead of a learned ``pos_embed`` and carry ``num_prefix_tokens``
(1 cls + 4 register) prefix tokens. The RoPE tensor is computed once per forward from
the patch grid and applied inside every block to the patch tokens (positions after the
prefix). Because of this, the DINOv2 ViT-Adapter (which drops the prefix tokens and
relies on ``pos_embed`` interpolation) cannot be reused directly.

This module reuses ``DINOV2Adapter`` for the shared ViT-Adapter machinery
(``_init_weights`` / ``_init_deform_weights`` / ``freeze_backbone`` / ``_add_level_embed``)
but, unlike ``DINOV2Adapter`` which *is* a TAO ``DINOV2`` ViT, DINOv3 wraps the external
timm Eva model in ``self.vit``. It therefore uses a fresh ``__init__`` (that does not build
a TAO ViT) and overrides the patch-embed / block-iteration / weight-load paths. The prefix
tokens are kept through the transformer blocks so RoPE application is numerically identical
to the vanilla ``forward_features`` path (verified via block-iteration parity).
"""

from functools import partial

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.segformer.model.backbones.dino_v2 import DINOV2Adapter
from nvidia_tao_pytorch.cv.segformer.model.backbones.adapter_modules import (
    Extractor,
    Injector,
    SpatialPriorModule,
    deform_inputs,
)


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
        """Initialize DINOv3 interaction block.

        Args:
            dim (int): The feature dimension.
            num_heads (int): Parallel attention heads of MultiScaleDeformableAttention.
            n_points (int): The number of sampling points for each query in each head.
            norm_layer (nn.Module): norm layer.
            drop (float): Dropout probability after the feed forward layer.
            drop_path (float): stochastic depth rate.
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
        """Forward function.

        Args:
            x (torch.Tensor): tokens of shape [bs, num_summary + H*W, dim] (prefix + patches).
            c (torch.Tensor): adapter (SPM) features for injector.
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


class DINOV3Adapter(DINOV2Adapter):
    """ViT-Adapter (https://arxiv.org/abs/2205.08534) wrapping a timm DINOv3 (Eva) model.

    Subclasses ``DINOV2Adapter`` purely to reuse its adapter helpers (``_init_weights``,
    ``_init_deform_weights``, ``freeze_backbone``, ``_add_level_embed``). It does NOT build a
    TAO ViT (``DINOV2Adapter.__init__`` is intentionally bypassed); DINOv3 wraps the timm Eva
    model in ``self.vit`` and overrides the patch-embed / block-iteration / weight-load paths.
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        return_idx=[0, 1, 2, 3],
        patch_size=16,
        img_size=224,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the DINOv3 ViT-Adapter.

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
            norm_layer (nn.Module): norm layer used by the adapter modules.
            return_idx (list): List of SPM output indices to return as features.
            patch_size (int): ViT patch size (16 for DINOv3).
            img_size (int): Input image size (must be divisible by 32).
            activation_checkpoint (bool): Whether to use activation checkpointing.
            freeze_at (list or str): Stages to freeze (see ``BackboneBase``). ``"all"``
                freezes the whole backbone; the adapter modules are unfrozen afterwards.
            freeze_norm (bool): Whether to freeze normalization layers.
            **kwargs: Unused, kept for factory-signature compatibility.
        """
        assert img_size % 32 == 0, f"Input img_size ({img_size}) should be divisible by 32."
        # Intentionally bypass DINOV2Adapter.__init__ (which builds a TAO DINOV2 ViT). DINOv3
        # wraps an external timm Eva model instead, so initialize BackboneBase directly and then
        # build the (shared) adapter modules below.
        BackboneBase.__init__(  # pylint: disable=non-parent-init-called
            self,
            in_chans=3,
            num_classes=0,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        self.vit = dino_model
        self.embed_dim = dino_model.embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        # cls + register tokens kept through the transformer blocks (RoPE alignment).
        self.num_summary_tokens = dino_model.num_prefix_tokens
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        normal_(self.level_embed)
        self.spm = SpatialPriorModule(
            in_channel=3,
            patch_size=self.patch_size,
            inplanes=conv_inplane,
            embed_dim=self.embed_dim,
            out_indices=return_idx,
        )
        self.spm.apply(self._init_weights)
        self.interactions = nn.Sequential(
            *[
                DINOV3InteractionBlock(
                    dim=self.embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                    with_cp=self.activation_checkpoint,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.interactions.apply(self._init_weights)
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.up.apply(self._init_weights)
        self.norm1 = nn.BatchNorm2d(self.embed_dim)
        self.norm2 = nn.BatchNorm2d(self.embed_dim)
        self.norm3 = nn.BatchNorm2d(self.embed_dim)
        self.norm4 = nn.BatchNorm2d(self.embed_dim)
        self.apply(self._init_deform_weights)

    # _init_weights, _init_deform_weights, freeze_backbone and _add_level_embed are
    # inherited unchanged from DINOV2Adapter.

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Record the activation-checkpointing flag (matches ``BackboneBase``).

        Overrides the timm-ViT ``set_grad_checkpointing`` inherited via ``DINOV2Adapter``,
        which assumes ``self.patch_embed`` -- DINOv3 wraps the ViT in ``self.vit`` and drives
        the blocks manually, so we only need to store the flag (as ``BackboneBase`` does).
        """
        self.activation_checkpoint = enable

    def get_stage_dict(self):
        """Map stage keys to modules for optional partial freezing."""
        stage_dict = {0: self.vit.patch_embed}
        for i, block in enumerate(self.vit.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Return the wrapped model's classifier."""
        return self.vit.get_classifier()

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the wrapped model's classifier head."""
        self.vit.reset_classifier(num_classes, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        """Route a bare timm DINOv3 state dict into the wrapped ``self.vit`` model.

        Pretrained / SSL->timm-converted DINOv3 checkpoints use bare timm keys
        (``blocks.*``, ``cls_token``, ``reg_token``, ``patch_embed.*``, ``norm.*``).
        The wrapped model lives under ``self.vit``, so prefix those keys with ``vit.``.
        The prefixing is all-or-nothing: it only triggers when no key is already an
        adapter/``vit.`` key.
        """
        adapter_prefixes = (
            "vit.",
            "spm.",
            "interactions.",
            "level_embed",
            "up.",
            "norm1.",
            "norm2.",
            "norm3.",
            "norm4.",
        )
        if state_dict and not any(k.startswith(adapter_prefixes) for k in state_dict):
            state_dict = {f"vit.{k}": v for k, v in state_dict.items()}
        return nn.Module.load_state_dict(self, state_dict, **kwargs)

    def get_patch_embed(self, x):
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

    def forward_feature_pyramid(self, x, indices=None, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps."""
        deform_inputs1, deform_inputs2 = deform_inputs(x, patch_size=self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        _, _, H, W = x.shape
        # Downsampling in the SPM ResNet stem: c3 corresponds to stride 16.
        H, W = H // 16, W // 16

        # Patch embedding + RoPE (prefix tokens kept for RoPE alignment).
        x_tokens, rope, patch_h, patch_w = self.get_patch_embed(x)
        bs, _, dim = x_tokens.shape
        num_summary = self.num_summary_tokens

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blocks = self.vit.blocks[indexes[0]: indexes[-1] + 1]
            x_tokens, c = layer(
                x_tokens,
                c,
                blocks,
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                rope=rope,
                num_summary=num_summary,
            )
            patch_tokens = x_tokens[:, num_summary:]
            outs.append(patch_tokens.transpose(1, 2).view(bs, dim, patch_h, patch_w).contiguous())

        # Split & Reshape
        c2 = c[:, 0: c2.size(1), :]
        c3 = c[:, c2.size(1): c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=c4.shape[-2:], mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def forward_pre_logits(self, x):
        """Return the last feature map (segmentation backbones do not use logits)."""
        return self.forward_feature_pyramid(x)[-1]

    def forward(self, x):
        """Return the multi-scale feature pyramid."""
        return self.forward_feature_pyramid(x)


def _create_dinov3(timm_name):
    """Create a randomly-initialized timm DINOv3 model (weights loaded later)."""
    # pretrained=False: Segformer loads the backbone weights from
    # ``pretrained_backbone_path`` after construction (see SegFormer.__init__).
    return timm.create_model(timm_name, pretrained=False)


def vit_small_dinov3(return_idx=[0, 1, 2, 3], resolution=224, freeze_at=None, activation_checkpoint=False, **kwargs):
    """DINOv3 ViT-Small/16 ViT-Adapter (embed_dim 384, depth 12)."""
    dino_model = _create_dinov3("vit_small_patch16_dinov3.lvd1689m")
    return DINOV3Adapter(
        dino_model,
        img_size=resolution,
        patch_size=16,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


def vit_small_plus_dinov3(return_idx=[0, 1, 2, 3], resolution=224, freeze_at=None, activation_checkpoint=False, **kwargs):
    """DINOv3 ViT-Small+/16 ViT-Adapter (embed_dim 384, depth 12)."""
    dino_model = _create_dinov3("vit_small_plus_patch16_dinov3.lvd1689m")
    return DINOV3Adapter(
        dino_model,
        img_size=resolution,
        patch_size=16,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


def vit_base_dinov3(return_idx=[0, 1, 2, 3], resolution=224, freeze_at=None, activation_checkpoint=False, **kwargs):
    """DINOv3 ViT-Base/16 ViT-Adapter (embed_dim 768, depth 12)."""
    dino_model = _create_dinov3("vit_base_patch16_dinov3.lvd1689m")
    return DINOV3Adapter(
        dino_model,
        img_size=resolution,
        patch_size=16,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


def vit_large_dinov3(return_idx=[0, 1, 2, 3], resolution=224, freeze_at=None, activation_checkpoint=False, **kwargs):
    """DINOv3 ViT-Large/16 ViT-Adapter (embed_dim 1024, depth 24)."""
    dino_model = _create_dinov3("vit_large_patch16_dinov3.lvd1689m")
    return DINOV3Adapter(
        dino_model,
        img_size=resolution,
        patch_size=16,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


def vit_huge_plus_dinov3(return_idx=[0, 1, 2, 3], resolution=224, freeze_at=None, activation_checkpoint=False, **kwargs):
    """DINOv3 ViT-Huge+/16 ViT-Adapter (embed_dim 1280, depth 32)."""
    dino_model = _create_dinov3("vit_huge_plus_patch16_dinov3.lvd1689m")
    return DINOV3Adapter(
        dino_model,
        img_size=resolution,
        patch_size=16,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        drop_path_rate=0.4,
        return_idx=return_idx,
        freeze_at=freeze_at,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )
