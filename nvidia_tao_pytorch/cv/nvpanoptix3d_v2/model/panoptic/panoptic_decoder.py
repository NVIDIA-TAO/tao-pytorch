# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 panoptic segmentation head over VGGT features.

Components:
  - DINO + VGGT concatenation and a three-block per-view spatial mixer
  - one native stride-14 cross-attention memory
  - LoftUp upscaler for stride-2 mask features
  - mask transformer with normalized 2D sine positional encoding
  - text encoder for open-vocabulary classification
"""

from typing import Dict, List, Optional

import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.feature_fusion import (
    ConcatenatedFeatureFusion,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.mask_transformer import (
    NVPanoptix3Dv2MaskTransformer,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.text_encoder import (
    TextEncoder,
)


class NVPanoptix3Dv2PanopticDecoder(nn.Module):
    """
    Panoptic segmentation decoder for NVPanoptix3Dv2.

    Every decoder layer attends to the same native stride-14 fused token map.
    High-resolution spatial detail enters mask prediction through LoftUp's
    RGB-guided stride-2 mask features, not a synthetic cross-attention FPN.
    """

    def __init__(
        self,
        feature_fusion: ConcatenatedFeatureFusion,
        upscaler: nn.Module,
        hidden_dim: int = 768,
        mask_dim: int = 384,
        ff_dim: int = 2048,
        num_queries: int = 200,
        num_heads: int = 8,
        dec_layers: int = 6,
        fixed_vocab: bool = True,
        label_mode: str = "sigmoid",
        deep_supervision: bool = True,
        patch_size: int = 14,
        enable_objectness: bool = False,
    ):
        super().__init__()

        self.feature_fusion = feature_fusion
        self.upscaler = upscaler
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision
        self.label_mode = label_mode
        self.text_encoder = TextEncoder(fixed_vocab=fixed_vocab)

        if self.label_mode == "softmax":
            self.nocls_token = nn.Parameter(torch.randn(self.text_encoder.embed_dim))

        self.mask_transformer = NVPanoptix3Dv2MaskTransformer(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            mask_dim=mask_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            dec_layers=dec_layers,
            lang_dim=self.text_encoder.embed_dim,
            num_feature_levels=1,
            enable_objectness=enable_objectness,
        )

    def forward(
        self,
        dino_feats: torch.Tensor,
        vggt_feats: torch.Tensor,
        images: torch.Tensor,
        classes: List[str],
        outdevice: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            dino_feats:        [B, S, P, C_dino]      DINOv2 patch features
            vggt_feats:        [B, S, P, C_vggt]      VGGT final AA features
            images:            [B, S, 3, H, W]        input images
            classes:           list of class names

        Returns:
            dict with pred_logits, pred_masks, aux_outputs, out_queries
        """
        B, S, P, _ = dino_feats.shape
        H, W = images.shape[-2:]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image shape {(H, W)} must be divisible by patch_size="
                f"{self.patch_size}"
            )
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        expected_patches = patch_h * patch_w
        if P != expected_patches or vggt_feats.shape[2] != expected_patches:
            raise ValueError(
                "Backbone token counts do not match the input orientation: "
                f"DINO P={P}, VGGT P={vggt_feats.shape[2]}, expected "
                f"{patch_h}*{patch_w}={expected_patches} for {(H, W)}"
            )

        # Vocabulary-independent visual feature fusion
        dino_flat = dino_feats.reshape(B * S, P, -1)
        vggt_flat = vggt_feats.reshape(B * S, P, -1)
        fused = self.feature_fusion(
            dino_flat,
            vggt_flat,
            spatial_shape=(patch_h, patch_w),
        )
        fused = fused.reshape(B, S, P, -1)

        # --- Text embeddings (classification head only; visual features are
        # never conditioned on the vocabulary)
        cls_embeddings = self.text_encoder(classes)
        cls_embeddings = cls_embeddings.to(fused.device)
        if self.label_mode == "softmax":
            cls_embeddings = torch.cat([cls_embeddings, self.nocls_token[None]], dim=0)

        # One native stride-14 cross-attention level
        C = fused.shape[-1]
        native = fused.reshape(B, S, patch_h, patch_w, C).permute(0, 1, 4, 2, 3)
        fpn = [native]

        # Upscaler: fused tokens + images -> high-res mask features
        fused_per_view = fused.reshape(B * S, P, -1)
        imgs_per_view = images.reshape(B * S, *images.shape[2:])
        _, mask_feats = self.upscaler(
            (fused_per_view, imgs_per_view),
            (H, W),
        )
        mC, mH, mW = mask_feats.shape[1:]
        mask_feats = mask_feats.reshape(B, S, mC, mH, mW)

        # Multi-view mask transformer with normalized 2D sine PE only.
        pan_out = self.mask_transformer(
            fpn,
            mask_feats,
            cls_embeddings,
            deep_supervision=self.deep_supervision,
            outdevice=outdevice,
        )

        return pan_out
