# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""ReLA module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from torch import Tensor

from nvidia_tao_pytorch.cv.mask_grounding_dino.model.modules import (
    CrossAttentionLayer,
    SelfAttentionLayer,
    FFNLayer,
)
from nvidia_tao_pytorch.cv.dino.model.position_encoding import (
    PositionEmbeddingSineHW,
    PositionEmbeddingSineHWExport,
)
from nvidia_tao_pytorch.cv.dino.model.model_utils import MLP


class ReLA(nn.Module):
    """ReLA module."""

    def __init__(
        self,
        num_queries: int,
        *,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        rla_weight: float = 0.1,
        num_feature_levels: int = 3,
        mask_dim: int = 256,
        export: bool = False,
    ) -> None:
        """
        Args:
            num_queries (int): Number of learnable query embeddings.
            hidden_dim (int): Dimension of hidden embeddings.
            nheads (int): Number of attention heads.
            dim_feedforward (int): Hidden dimension of FFN layers.
            dec_layers (int): Number of decoder layers.
            pre_norm (bool): Apply LayerNorm before attention/FFN if True.
            rla_weight (float, optional): Weight for language attention contribution. Defaults to 0.1.
            num_feature_levels (int, optional): Number of feature levels. Defaults to 3.
            mask_dim (int, optional): Hidden dimension for mask prediction MLP. Defaults to 256.
            out_dim (int, optional): Output dimension for minimap embeddings. Defaults to 2.
        """
        super().__init__()

        # Positional encoding
        N_steps = hidden_dim // 2
        if export:
            self.pe_layer = PositionEmbeddingSineHWExport(N_steps, normalize=True)
        else:
            self.pe_layer = PositionEmbeddingSineHW(N_steps, normalize=True)

        # Transformer decoder layers
        self.num_layers = dec_layers
        self.num_heads = nheads
        self.rla_weight = rla_weight

        self.RLA_vision: nn.ModuleList = nn.ModuleList()
        self.RIA_layers: nn.ModuleList = nn.ModuleList()
        self.transformer_ffn_layers: nn.ModuleList = nn.ModuleList()

        for _ in range(self.num_layers):
            self.RLA_vision.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.RIA_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Learnable queries
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Language cross-attention
        self.lang_weight = nn.parameter.Parameter(data=torch.as_tensor(0.0))
        self.RLA_lang_att = CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)

        # Multi-level feature embeddings
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Output FFNs
        self.minimap_embed = nn.Linear(hidden_dim, 2)
        self.nt_embed = MLP(hidden_dim, hidden_dim, 2, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self,
        srcs: List[Tensor],
        pos_embeds: List[Tensor],
        mask_features: Tensor,
        lang_feat: Tensor,
        spatial_shapes: List[Tensor],
    ) -> Dict[str, List[Tensor]]:
        """Forward pass for ReLA.

        Args:
            srcs (List[Tensor]): Multi-level source features (B, HW, C).
            pos_embeds (List[Tensor]): Positional embeddings for source features.
            mask_features (Tensor): Mask feature map (B, C, H, W).
            lang_feat (Tensor): Language feature embeddings (B, L, C).
            spatial_shapes (List[Tensor]): Spatial dimensions per feature level.

        Returns:
            Dict[str, List[Tensor]]: Dictionary containing 'minimaps', 'no_targets', 'union_mask_logits'.
        """
        srcs = srcs[::-1][-self.num_feature_levels:]
        pos_embeds = pos_embeds[::-1][-self.num_feature_levels:]
        spatial_shapes = spatial_shapes[::-1][-self.num_feature_levels:]

        srcs_tokens: List[Tensor] = []
        pos_embed_tokens: List[Tensor] = []

        use_level_embed = self.num_feature_levels > 1 and self.level_embed is not None
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            if use_level_embed:
                src = src + self.level_embed.weight[lvl][None, None, :]
            srcs_tokens.append(src.permute(1, 0, 2))  # (HW, B, C)
            pos_embed_tokens.append(pos_embed.permute(2, 0, 1))  # (HW, B, C)

        _, bs, _ = srcs_tokens[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        nt_labels, tgt_masks, minimaps = [], [], []

        # First forward through prediction heads
        outputs_minimap, nt_label, tgt_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=spatial_shapes[0]
        )
        nt_labels.append(nt_label)
        minimaps.append(outputs_minimap)
        tgt_masks.append(tgt_mask)

        # ReLA layers
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Cross-attention: query -> vision features
            output = self.RIA_layers[i](
                output,
                srcs_tokens[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos_embed_tokens[level_index],
                query_pos=query_embed,
            )

            # Language attention only in first layer
            if i == 0:
                lang_feat_att = self.RLA_lang_att(output, lang_feat.permute(1, 0, 2))
                lang_feat_att = lang_feat_att * F.sigmoid(self.lang_weight)
                output = output + lang_feat_att * self.rla_weight

            # Self-attention and FFN
            output = self.RLA_vision[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)

            outputs_minimap, nt_label, tgt_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=spatial_shapes[(i + 1) % self.num_feature_levels]
            )
            minimaps.append(outputs_minimap)
            nt_labels.append(nt_label)
            tgt_masks.append(tgt_mask)

        return {
            'minimaps': minimaps,
            'no_targets': nt_labels,
            'union_mask_logits': tgt_masks,
        }

    def forward_prediction_heads(
        self,
        output: Tensor,
        mask_features: Tensor,
        attn_mask_target_size: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute outputs for minimap, no-target labels, and attention masks.

        Args:
            output (Tensor): Query features (num_queries, B, C).
            mask_features (Tensor): Mask feature map (B, C, H, W).
            attn_mask_target_size (Tensor): Target size for attention masks.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: minimaps, NT labels, union mask logits, attention mask.
        """
        region_features = self.decoder_norm(output).transpose(0, 1)

        # Mask embeddings
        region_embed = self.mask_embed(region_features)
        all_mask = torch.einsum("bqc,bchw->bqhw", region_embed, mask_features)

        # Minimap and target mask
        outputs_minimap = self.minimap_embed(region_features)
        tgt_embed = torch.einsum("bqa,bqc->bac", outputs_minimap, region_embed)
        tgt_mask = torch.einsum("bac,bchw->bahw", tgt_embed, mask_features)

        # No-target label
        nt_label = self.nt_embed(region_features).mean(dim=1)

        # Attention mask
        attn_mask = F.interpolate(
            all_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_minimap, nt_label, tgt_mask, attn_mask
