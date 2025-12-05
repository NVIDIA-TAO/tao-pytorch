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

"""Submodules for ReLA layers (cross-attention, self-attention, FFN)."""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _get_activation_fn(activation: str):
    """Return a PyTorch activation function given a string.

    Args:
        activation (str): Name of activation function. Options: "relu", "gelu", "glu".

    Returns:
        Callable: Corresponding activation function.

    Raises:
        RuntimeError: If the activation is not supported.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class CrossAttentionLayer(nn.Module):
    """Single layer of cross-attention with optional pre/post normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        normalize_before: bool = False,
    ):
        """
        Initialize a cross-attention layer.

        Args:
            d_model (int): Dimension of input features.
            nhead (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            normalize_before (bool, optional): Whether to apply LayerNorm before attention. Defaults to False.
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add positional embedding if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass when using post-normalization."""
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass when using pre-normalization."""
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass using pre or post normalization based on config."""
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class SelfAttentionLayer(nn.Module):
    """Single layer of self-attention with optional pre/post normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        normalize_before: bool = False,
    ):
        """
        Initialize a self-attention layer.

        Args:
            d_model (int): Dimension of input features.
            nhead (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            normalize_before (bool, optional): Whether to apply LayerNorm before attention. Defaults to False.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add positional embedding if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass when using post-normalization."""
        query = key = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            query, key, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass when using pre-normalization."""
        tgt2 = self.norm(tgt)
        query = key = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            query, key, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass using pre or post normalization based on config."""
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):
    """Feedforward layer with optional pre/post normalization."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        """
        Initialize a feedforward layer.

        Args:
            d_model (int): Input/output feature dimension.
            dim_feedforward (int, optional): Hidden dimension. Defaults to 2048.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            activation (str, optional): Activation function. Defaults to "relu".
            normalize_before (bool, optional): Whether to apply LayerNorm before FFN. Defaults to False.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform for weight matrices."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward_post(self, tgt: Tensor) -> Tensor:
        """Forward pass when using post-normalization."""
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor) -> Tensor:
        """Forward pass when using pre-normalization."""
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt: Tensor) -> Tensor:
        """Forward pass using pre or post normalization based on config."""
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
