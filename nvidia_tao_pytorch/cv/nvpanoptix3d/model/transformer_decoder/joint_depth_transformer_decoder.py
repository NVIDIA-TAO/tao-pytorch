# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py

"""NVPanoptix3D joint depth transformer decoder."""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from fvcore.nn import weight_init

from nvidia_tao_pytorch.core.modules.activation.activation import MultiheadAttention
from nvidia_tao_pytorch.cv.mask2former.model.transformer_decoder.position_encoding import (
    PositionEmbeddingSine
)


class SelfAttentionLayer(nn.Module):
    """A single self-attention block for query-to-query interactions.

    Inputs are expected in sequence-first format:
    - `tgt`: `[Q, B, C]` where Q is number of queries.
    - `query_pos`: optional positional embeddings of shape `[Q, B, C]`.
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, export=False):
        """Initialize a self-attention layer.

        Args:
            d_model: Embedding dimension C.
            nhead: Number of attention heads.
            dropout: Dropout probability applied on attention output residual.
            activation: Unused here (kept for interface parity).
            normalize_before: If True, apply LayerNorm before attention (pre-norm).
            export: If True, use TAO export-friendly `MultiheadAttention`.
        """
        super().__init__()
        self.export = export
        if export:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Optionally add positional embeddings.

        Args:
            tensor: Base tensor, typically `[Q, B, C]`.
            pos: Positional embedding broadcastable to `tensor`.

        Returns:
            Tensor with positional embedding added if `pos` is not None.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass with **post-norm** (attention -> residual -> norm).

        Args:
            tgt: Query features `[Q, B, C]`.
            tgt_mask: Optional attention mask (passed to MultiheadAttention).
            tgt_key_padding_mask: Optional padding mask of shape `[B, Q]`.
            query_pos: Optional query positional embedding `[Q, B, C]`.

        Returns:
            Updated query features `[Q, B, C]`.
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass with **pre-norm** (norm -> attention -> residual).

        Args:
            tgt: Query features `[Q, B, C]`.
            tgt_mask: Optional attention mask (passed to MultiheadAttention).
            tgt_key_padding_mask: Optional padding mask of shape `[B, Q]`.
            query_pos: Optional query positional embedding `[Q, B, C]`.

        Returns:
            Updated query features `[Q, B, C]`.
        """
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        """Dispatch to pre-norm or post-norm implementation."""
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """A single cross-attention block from queries to encoder/pixel features."""

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, export=False):
        """Initialize a cross-attention layer.

        Args:
            d_model: Embedding dimension C.
            nhead: Number of attention heads.
            dropout: Dropout probability applied on attention output residual.
            activation: Unused here (kept for interface parity).
            normalize_before: If True, apply LayerNorm before attention (pre-norm).
            export: If True, use TAO export-friendly `MultiheadAttention`.
        """
        super().__init__()
        self.export = export
        if export:
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Optionally add positional embeddings (kept as a small helper)."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """Forward pass with **post-norm** (attention -> residual -> norm).

        Args:
            tgt: Query features `[Q, B, C]`.
            memory: Source features `[HW, B, C]`.
            memory_mask: Optional attention mask (commonly `[B*h, Q, HW]`).
            memory_key_padding_mask: Optional padding mask for `memory`.
            pos: Optional positional embedding for `memory`, `[HW, B, C]`.
            query_pos: Optional positional embedding for `tgt`, `[Q, B, C]`.

        Returns:
            Updated query features `[Q, B, C]`.
        """
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """Forward pass with **pre-norm** (norm -> attention -> residual).

        Args:
            tgt: Query features `[Q, B, C]`.
            memory: Source features `[HW, B, C]`.
            memory_mask: Optional attention mask (commonly `[B*h, Q, HW]`).
            memory_key_padding_mask: Optional padding mask for `memory`.
            pos: Optional positional embedding for `memory`, `[HW, B, C]`.
            query_pos: Optional positional embedding for `tgt`, `[Q, B, C]`.

        Returns:
            Updated query features `[Q, B, C]`.
        """
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """Dispatch to pre-norm or post-norm implementation."""
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Position-wise feed-forward network (FFN) used inside the decoder."""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        """Initialize the FFN block.

        Args:
            d_model: Embedding dimension C.
            dim_feedforward: Hidden dimension of the FFN.
            dropout: Dropout probability used on hidden and residual paths.
            activation: Activation function name ("relu", "gelu", "glu").
            normalize_before: If True, apply LayerNorm before FFN (pre-norm).
        """
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Helper kept for interface parity with attention layers."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        """Forward pass with **post-norm** (ffn -> residual -> norm)."""
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        """Forward pass with **pre-norm** (norm -> ffn -> residual)."""
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        """Dispatch to pre-norm or post-norm implementation."""
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Map a string name to a PyTorch activation function.

    Args:
        activation: One of `"relu"`, `"gelu"`, `"glu"`.

    Returns:
        Callable activation function from `torch.nn.functional`.

    Raises:
        NotImplementedError: If `activation` is not one of the supported names.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise NotImplementedError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """A simple multi-layer perceptron used for mask/depth embedding heads."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DepthAwareMultiScaleMaskedTransformerDecoder(nn.Module):
    """Multi-scale masked transformer decoder with depth-aware cross-attention.

    This decoder follows the Mask2Former-style iterative decoding:
    - Maintain a set of learnable queries (content + positional embeddings).
    - Repeatedly apply cross-attention to image features and self-attention among
      queries, then update predictions after each layer.

    Depth-awareness is introduced by a second cross-attention per layer that
    attends to **depth features** (`multi_scale_depth_features`) in addition to
    the standard image features (`x`).
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        export=False,
    ):
        """Initialize the multi-scale masked transformer decoder.

        Args:
            in_channels: Channel dimension of incoming multi-scale features.
            num_classes: Number of foreground classes (background handled via +1).
            mask_classification: Must be True (this implementation assumes it).
            hidden_dim: Transformer embedding dimension C.
            num_queries: Number of learnable queries Q.
            nheads: Number of attention heads.
            dim_feedforward: Hidden dimension of FFN blocks.
            dec_layers: Number of decoder layers.
            pre_norm: If True, use pre-norm in attention/ffn blocks.
            mask_dim: Output dimension of `mask_embed` MLP head.
            enforce_input_project: If True, always project features to `hidden_dim`.
            export: If True, use export-friendly multi-head attention modules.
        """
        super().__init__()
        self.export = export
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.depth_transformer_self_attention_layers = nn.ModuleList()
        self.depth_transformer_cross_attention_layers = nn.ModuleList()
        self.depth_transformer_ffn_layers = nn.ModuleList()

        self.cross_transformer_attention_layers = nn.ModuleList()
        self.depth_output_proj = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    export=export,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    export=export,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.depth_transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    export=export,
                )
            )

            self.cross_transformer_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    export=export,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.depth_decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.depth_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.depth_input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                self.depth_input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
                weight_init.c2_xavier_fill(self.depth_input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
                self.depth_input_proj.append(nn.Sequential())

        self.hidden_dim = hidden_dim

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.depth_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, multi_scale_depth_features, mask_features, depth_features=None, mask=None):
        """Run decoder and produce class/mask predictions.

        Args:
            x: List of multi-scale image features, length `num_feature_levels`.
                Each element is `[B, C_in, H_l, W_l]`.
            multi_scale_depth_features: List of multi-scale depth features aligned
                with `x`, same length and spatial sizes, each `[B, C_in, H_l, W_l]`.
            mask_features: High-resolution mask features from pixel decoder,
                `[B, C_mask, H_mask, W_mask]` (used to generate per-query masks).
            depth_features: Optional extra depth features carried through the API
                and returned in `out` (not used to predict depth here).
            mask: Unused (kept for interface compatibility; explicitly deleted).

        Returns:
            A dict with keys:
            - `pred_logits`, `pred_masks`, `pred_depths`, `aux_outputs`
            - `mask_features`, `depth_features`, `enc_features`, `segm_decoder_out`
        """
        assert len(x) == self.num_feature_levels, "Input does not match the number of feature levels."
        src = []
        depth_src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            depth_src.append(
                self.depth_input_proj[i](
                    multi_scale_depth_features[i]
                ).flatten(2) + self.depth_level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
            depth_src[-1] = depth_src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_depth = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, outputs_depth, attn_mask, \
            decoder_out = self.forward_prediction_heads(
                output=output,
                mask_features=mask_features,
                depth_output=None,
                depth_features=depth_features,
                attn_mask_target_size=size_list[0]
            )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_depth.append(outputs_depth)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention with general features
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            # attention: cross-attention with depth features
            output = self.depth_transformer_cross_attention_layers[i](
                output, depth_src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            # inter-query attention

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, outputs_depth, attn_mask, \
                decoder_out = self.forward_prediction_heads(
                    output=output,
                    mask_features=mask_features,
                    depth_output=None,
                    depth_features=depth_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
                )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_depth.append(outputs_depth)

        assert len(predictions_class) == self.num_layers + 1, \
            f"Number of predictions ({len(predictions_class)}) doesn't match number of layers ({self.num_layers}) + 1."

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_depths": predictions_depth[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
                predictions_depth,
            ),
            "mask_features": mask_features,
            "depth_features": depth_features,
            "enc_features": x,
            "segm_decoder_out": decoder_out
        }
        return out

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        depth_output,
        depth_features,
        attn_mask_target_size
    ):
        """Compute per-layer predictions and an attention mask for the next layer.

        Args:
            output: Query features `[Q, B, C]`.
            mask_features: Pixel-decoder mask features `[B, C_mask, H_mask, W_mask]`.
            depth_output: Unused placeholder for depth branch (kept for API parity).
            depth_features: Optional depth features (unused for prediction here).
            attn_mask_target_size: Spatial size `(H_l, W_l)` of the feature level
                that the next cross-attention will attend to.

        Returns:
            outputs_class: `[B, Q, num_classes + 1]`
            outputs_mask: `[B, Q, H_mask, W_mask]`
            outputs_depth: Always `None` (depth is produced elsewhere)
            attn_mask: Boolean mask typically shaped `[B*h, Q, H_l*W_l]`
            decoder_output: The normalized query features `[B, Q, C]`
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        ).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        # get depth prediction from VGGT, not from transformer decoder
        outputs_depth = None

        return outputs_class, outputs_mask, outputs_depth, attn_mask, decoder_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_depth):
        """Format auxiliary outputs for deep supervision.

        Args:
            outputs_class: List of class logits per layer (including last).
            outputs_seg_masks: List of mask logits per layer (including last).
            outputs_depth: List of depth outputs per layer (including last).

        Returns:
            A list of dicts, one per intermediate layer (excluding final),
            matching the keys expected by training losses.
        """
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_depths": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_depth[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_depths": c}
                for b, c in zip(outputs_seg_masks[:-1], outputs_depth[:-1])
            ]
