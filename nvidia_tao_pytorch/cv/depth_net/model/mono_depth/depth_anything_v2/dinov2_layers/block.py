# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block"""

from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from timm.models.vision_transformer import DropPath, LayerScale, Attention
from .attention import MemEffAttention
from timm.layers import Mlp


try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class Block(nn.Module):
    """Vision Transformer Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        """Vision Transformer Block

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of the hidden dimension in the MLP.. Defaults to 4..
            qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
            qk_norm (bool, optional): Whether to apply normalization to QK. Defaults to False.
            proj_drop (float, optional): Dropout rate for the projection. Defaults to 0..
            attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0..
            init_values (float, optional): Initialization values for layer scaling. Defaults to None.
            drop_path (float, optional): Drop path probability. Defaults to 0..
            act_layer (Callable, optional): Activation function. Defaults to nn.GELU.
            norm_layer (Callable, optional): Normalization layer. Defaults to nn.LayerNorm.
            mlp_layer (Callable, optional): MLP layer class. Defaults to Mlp.
            attn_class (Callable, optional): Attention class to use. Defaults to Attention.
        """
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the block

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after block
        """

        def attn_residual_func(x: Tensor) -> Tensor:
            """Calculates the attention residual.

            Args:
                x (Tensor): The input tensor for which to compute the residual.

            Returns:
                Tensor: The computed attention residual.
            """
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            """Calculates the feedforward residual.

            Args:
                x (Tensor): The input tensor for which to compute the residual.

            Returns:
                Tensor: The computed feedforward residual.
            """
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    """
    This function implements stochastic depth regularization by randomly selecting
    a subset of the batch, applying a residual function to that subset, and then
    adding the scaled residual back to the original tensor. This helps with
    regularization and training stability in deep networks.

    Args:
        x (Tensor): Input tensor of shape (B, ...) where B is batch size.
        residual_func (Callable[[Tensor], Tensor]): Function that computes the
            residual to be added. Should take a tensor and return a tensor of
            the same shape.
        sample_drop_ratio (float, optional): Probability of dropping elements
            from the batch. Must be in [0, 1). Defaults to 0.0.

    Returns:
        Tensor: Input tensor with stochastic depth residual applied.
            Same shape as input tensor.

    Note:
        - The function ensures at least one element is always selected (sample_subset_size >= 1)
        - The residual is scaled by b / sample_subset_size to maintain expected magnitude
        - Uses torch.randperm for random selection without replacement
    """
    # 1) extract subset using permutation
    b = x.shape[0]
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    """
    This utility function generates random batch indices and computes the
    corresponding scaling factor for stochastic depth regularization.

    Args:
        x (torch.Tensor): Input tensor of shape (B, ...) where B is batch size.
        sample_drop_ratio (float, optional): Probability of dropping elements
            from the batch. Must be in [0, 1). Defaults to 0.0.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Selected batch indices for stochastic depth
            - float: Scaling factor to maintain expected magnitude

    Note:
        - Ensures at least one element is always selected
        - Uses torch.randperm for random selection without replacement
        - Scaling factor is b / sample_subset_size
    """
    b = x.shape[0]
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    """
    This function adds a scaled residual to specific indices of the input tensor.
    It supports both standard addition and element-wise scaled addition using
    a scaling vector.

    Args:
        x (Tensor): Input tensor to which residual will be added.
        brange (Tensor): Indices specifying where to add the residual.
        residual (Tensor): Residual tensor to be added.
        residual_scale_factor (float): Global scaling factor for the residual.
        scaling_vector (Tensor, optional): Element-wise scaling vector for the residual.
            If provided, each element in residual is scaled by corresponding element
            in scaling_vector. Defaults to None.

    Returns:
        Tensor: Input tensor with residual added at specified indices.

    Note:
        - Uses torch.index_add for efficient addition at specific indices
        - If scaling_vector is provided, uses scaled_index_add function
        - Residual is converted to input tensor dtype before addition
    """
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    This function generates attention bias masks for nested tensor processing
    and concatenates selected tensors from a list. It's used in the context
    of processing variable-length sequences or nested tensor structures.

    Args:
        x_list (List[Tensor]): List of tensors to process and concatenate.
            Each tensor should represent a sequence of embeddings.
        branges (Optional[List[Tensor]], optional): List of index tensors specifying
            which elements to select from each corresponding tensor in x_list.
            If None, all elements are selected. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - Tensor: Concatenated selected tensors
            - Tensor: Attention bias mask for the concatenated tensor
            - List[Tensor]: List of attention bias masks for individual tensors

    Note:
        - Uses a global cache (attn_bias_cache) to avoid recomputing attention biases
        - The attention bias mask helps handle variable-length sequences in attention
        - If branges is provided, only selected elements are concatenated
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    """
    This function applies stochastic depth regularization to a list of tensors by
    randomly selecting subsets of each tensor, applying a residual function to the
    concatenated subset, and then adding the scaled residuals back to the original
    tensors. This is particularly useful for processing nested tensor structures
    or variable-length sequences.

    Args:
        x_list (List[Tensor]): List of input tensors to process. Each tensor
            should have the same number of features but can have different
            batch sizes or sequence lengths.
        residual_func (Callable[[Tensor, Any], Tensor]): Function that computes
            the residual. Takes a concatenated tensor and optional attention bias,
            returns a residual tensor of the same shape.
        sample_drop_ratio (float, optional): Probability of dropping elements
            from each tensor in the batch. Must be in [0, 1). Defaults to 0.0.
        scaling_vector (Optional[Tensor], optional): Element-wise scaling vector
            for the residual. If provided, each element in the residual is scaled
            by the corresponding element in scaling_vector. Defaults to None.

    Returns:
        List[Tensor]: List of tensors with stochastic depth residuals applied.
            Each tensor has the same shape as the corresponding input tensor.

    Note:
        - Uses get_branges_scales to generate random indices for each tensor
        - Concatenates selected elements using get_attn_bias_and_cat
        - Applies residual_func to the concatenated tensor
        - Splits the result and adds residuals back to individual tensors
        - Supports attention bias for efficient processing of variable-length sequences
    """
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    """
    This block utilizes the MemEffAttention class for attention operations and
    can handle both single tensors and lists of tensors. It implements residual connections
    with optional stochastic depth for training.
    """

    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        """Performs the forward pass for the NestedTensorBlock."""
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
