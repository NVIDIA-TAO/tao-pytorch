# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 multi-view mask transformer.

The decoder uses learned object queries and normalized 2D sine positional
encoding only. The same spatial positional encoding is repeated for every
view, so there is no learned ordinal frame table and no view-count-dependent
3D coordinate system.
"""

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from nvidia_tao_pytorch.cv.mask2former.model.transformer_decoder.position_encoding import (
    PositionEmbeddingSine,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.transformer_decoder.joint_depth_transformer_decoder import (
    FFNLayer,
    MLP,
    SelfAttentionLayer,
)


class CrossAttentionLayer(nn.Module):
    """Visual cross-attention with normalized 2D positional encoding only.

    Separate projections keep visual attention explicit and avoid
    geometry-specific parameters. The residual is post-normalized.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize matrix parameters with Xavier-uniform weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt:       [Q, B, C]   query tokens
            memory:    [L, B, C]   key/value tokens (frame features)
            memory_mask: optional attention mask
            pos:       [L, B, C]   2D sinusoidal PE for keys
            query_pos: [Q, B, C]   PE for queries
        """
        residual = tgt

        q = self.q_proj(tgt + query_pos if query_pos is not None else tgt)
        k = self.k_proj(memory + pos if pos is not None else memory)
        v = self.v_proj(memory)

        Q, B, _ = q.shape
        L = k.shape[0]

        q = q.permute(1, 0, 2).reshape(B, Q, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = k.permute(1, 0, 2).reshape(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(1, 0, 2).reshape(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if memory_mask is not None:
            if memory_mask.dim() == 2:
                attn_weights = attn_weights.masked_fill(memory_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            elif memory_mask.dim() == 3:
                attn_weights = attn_weights.masked_fill(memory_mask.unsqueeze(1), float("-inf"))
            elif memory_mask.dim() == 4:
                attn_weights = attn_weights.masked_fill(memory_mask, float("-inf"))
            else:
                attn_weights = attn_weights + memory_mask

        if memory_key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                memory_key_padding_mask[:, None, None, :], float("-inf")
            )

        # FP32 normalization is more stable for 20/50-view inference, where
        # the attention denominator spans tens of thousands of tokens.
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(v.dtype)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).reshape(B, Q, self.d_model).permute(1, 0, 2)

        out = self.out_proj(out)
        return self.norm(residual + self.dropout(out))


# Visual-only multi-view decoder

class NVPanoptix3Dv2MaskTransformer(nn.Module):
    """Multi-view Mask2Former decoder with learned object queries.

    Cross-attention uses normalized 2D sine PE only. Spatial PE is repeated
    across views; there is no frame table, 3D PE, presence token, or
    geometry-based query initialization. A vocabulary-independent objectness
    head optionally separates matched/unmatched confidence from class scores.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim: int = 768,
        ff_dim: int = 2048,
        mask_dim: int = 384,
        num_queries: int = 200,
        num_heads: int = 8,
        dec_layers: int = 6,
        lang_dim: int = 768,
        num_feature_levels: int = 1,
        enable_objectness: bool = False,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        if isinstance(in_dim, int):
            in_dim = [in_dim] * num_feature_levels

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = num_heads
        self.num_layers = dec_layers
        self.enable_objectness = enable_objectness

        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.self_attn_layers.append(SelfAttentionLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.0))
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.0,
                )
            )
            self.ffn_layers.append(FFNLayer(d_model=hidden_dim, dim_feedforward=ff_dim, dropout=0.0))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Learnable Mask2Former queries.
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)

        self.input_proj = nn.ModuleList()
        for d in in_dim:
            if d != hidden_dim:
                proj = nn.Conv2d(d, hidden_dim, kernel_size=1)
                nn.init.xavier_normal_(proj.weight)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        self.lang_embed = nn.Linear(hidden_dim, lang_dim)
        self.cls_logit_scale = nn.Parameter(torch.ones([]))
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # Vocabulary-independent matched/unmatched confidence. Semantic
        # cosine similarity alone is not objectness: an unmatched query can
        # still be close to some text embedding. Keep the two decisions
        # separate so inference can rank by p(object) * p(class | object).
        if self.enable_objectness:
            self.objectness_embed = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.objectness_embed.weight)
            # Start from a conservative 10% object prior for 200 queries.
            nn.init.constant_(self.objectness_embed.bias, -math.log(9.0))

    def build_view_position_encoding(
        self,
        spatial_pos: torch.Tensor,
        n_views: int,
    ) -> torch.Tensor:
        """Tile the same spatial positional encoding across views."""
        return spatial_pos.repeat(n_views, 1, 1)

    def build_spatial_position_encoding(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Encode the feature map in its actual ``(H, W)`` orientation.

        VGGT patch tokens are already laid out in the input tensor's native
        orientation. Transposing only the positional grid for portrait inputs
        changes its flattened token order and assigns positions to the wrong
        visual tokens.
        """
        return self.pe_layer(feature_map, None).flatten(2).permute(2, 0, 1)

    def forward(
        self,
        fpn_f: List[torch.Tensor],
        mask_feats: torch.Tensor,
        cls_embeddings: torch.Tensor,
        deep_supervision: bool = True,
        outdevice: Optional[torch.device] = None,
    ):
        """Decode object queries against the fused multi-view memory."""
        if len(fpn_f) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature levels, got {len(fpn_f)}"
            )
        if outdevice is None:
            outdevice = mask_feats.device

        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            B, N, _, _, _ = fpn_f[i].shape
            size_list.append(fpn_f[i].shape[-2:])
            pos_i = self.build_spatial_position_encoding(fpn_f[i][:, 0])
            pos.append(self.build_view_position_encoding(pos_i, N))

            src_i = self.input_proj[i](fpn_f[i][:, :N].permute(0, 2, 1, 3, 4).flatten(-3))
            src.append(src_i.permute(2, 0, 1) + self.level_embed.weight[i][None, None])

        output = self.query_feat.weight.unsqueeze(1).expand(-1, B, -1)
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)

        outputs_class, outputs_masks, outputs_objectness, attn_mask = self.forward_prediction_heads(
            output, mask_feats, cls_embeddings, attn_mask_target_size=size_list[0], outdevice=outdevice
        )

        predictions_class = []
        predictions_masks = []
        predictions_objectness = []
        if deep_supervision:
            predictions_class.append(outputs_class)
            predictions_masks.append(outputs_masks)
            predictions_objectness.append(outputs_objectness)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.cross_attn_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.self_attn_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed,
            )

            output = self.ffn_layers[i](output)

            outputs_class, outputs_masks, outputs_objectness, attn_mask = self.forward_prediction_heads(
                output, mask_feats, cls_embeddings,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                outdevice=outdevice,
            )

            if deep_supervision or i >= self.num_layers - 1:
                predictions_class.append(outputs_class)
                predictions_masks.append(outputs_masks)
                predictions_objectness.append(outputs_objectness)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_masks[-1],
            "pred_objectness": predictions_objectness[-1],
            "aux_outputs": [
                {
                    "pred_logits": c,
                    "pred_masks": m,
                    "pred_objectness": o,
                }
                for c, m, o in zip(
                    predictions_class[:-1],
                    predictions_masks[:-1],
                    predictions_objectness[:-1],
                )
            ],
            "out_queries": output.detach(),
        }
        return out

    def forward_prediction_heads(
        self,
        output: Tensor,
        mask_feats: Tensor,
        cls_embeddings: Tensor,
        attn_mask_target_size=None,
        outdevice=None,
    ):
        """Produce the per-layer class logits, mask logits and attention mask."""
        if outdevice is None:
            outdevice = mask_feats.device

        decoder_output = self.decoder_norm(output).transpose(0, 1)

        # This normalized projection is an internal classifier feature. It is
        # not returned because the current objective has no contrastive
        # query/text loss; class logits are its only consumer.
        class_query_embed = self.lang_embed(decoder_output)
        class_query_embed = class_query_embed / (
            class_query_embed.norm(dim=-1, keepdim=True) + 1e-7
        )
        outputs_class = (
            self.cls_logit_scale.exp() *
            class_query_embed
            @ cls_embeddings.unsqueeze(0).transpose(1, 2)
        ).to(outdevice)

        outputs_objectness = None
        if self.enable_objectness:
            outputs_objectness = self.objectness_embed(decoder_output).squeeze(-1)
            outputs_objectness = outputs_objectness.to(outdevice)

        mask_embed = self.mask_embed(decoder_output)
        mask_embed = mask_embed.unsqueeze(1).expand(-1, mask_feats.shape[1], -1, -1)

        outputs_mask = torch.einsum("bnqc,bnchw->bnqhw", mask_embed, mask_feats)

        attn_mask = None
        if attn_mask_target_size is not None:
            B, N_v, Q, _, _ = outputs_mask.shape
            attn_mask = outputs_mask.flatten(0, 1)
            attn_mask = F.interpolate(
                attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
            )
            attn_mask = attn_mask.view(B, N_v, Q, *attn_mask_target_size)
            attn_mask = attn_mask.permute(0, 2, 1, 3, 4).flatten(-3)
            attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1) < 0.5).bool()

        return (
            outputs_class,
            outputs_mask.to(outdevice),
            outputs_objectness,
            attn_mask,
        )
