# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block"""

from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, LayerScale, Attention
from torch import Tensor
from xformers.ops import fmha
from timm.layers import Mlp

from .attention import MemoryEfficientAttention


class Block(nn.Module):
    """Vision Transformer Block"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            attn_class=Attention,
    ):
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
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for the block

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after block
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


def get_subset_index_and_scale_factor(x, drop_ratio=0.0):
    """random selection of the subset of the batch

    Args:
        x (torch.Tensor): Input batch
        drop_ratio (float, optional): The probability of dropping elements from the batch. Defaults to 0.0.

    Returns:
        torch.Tensor: Selected indices
        float: Ratio of selected to original batch size
    """
    B, _, _ = x.shape
    selected_subset_size = max(int(B * (1 - drop_ratio)), 1)
    selected_indices = (torch.randperm(B, device=x.device))[:selected_subset_size]

    return selected_indices, B / selected_subset_size


def apply_residual(
    x, selected_indices, residual, residual_scale_factor, scaling_vector=None
):
    """Applies a scaled residual to selected indices of the input tensor `x`.

    Args:
        x (Tensor): The input tensor to which the residual will be applied.
        selected_indices(Tensor): Indices indicating the subset of `x` where the residual should be applied.
        residual (Tensor): The residual tensor to be added to `x` at `selected_indices`.
        residual_scale_factor (float): A scaling factor applied to the residual before addition.
        scaling_vector (Tensor, optional): A vector for element-wise scaling of the residual. If provided,
            each element in the residual will be scaled by the corresponding element in `scaling_vector`.
            Defaults to None.

    Returns:
        Tensor: The tensor `x` with the residual applied at the specified indices.
    """
    residual = residual.to(dtype=x.dtype)
    if scaling_vector is None:
        x_flat, residual_flat = x.flatten(1), residual.flatten(1)

        return torch.index_add(
            x_flat, 0, selected_indices, residual_flat, alpha=residual_scale_factor
        ).view_as(x)

    x_plus_residual = torch.index_add(
        x,
        dim=0,
        index=selected_indices,
        source=residual.to(dtype=x.dtype) * scaling_vector,
        alpha=residual_scale_factor,
    )

    return x_plus_residual.view_as(x)


@lru_cache(maxsize=None)
def get_attn_bias(batch_sizes: Tuple[int], seq_lens: Tuple[int]):
    """Generates an attention bias mask based on batch sizes and sequence lengths.

    Args:
        batch_sizes (Tuple[int]): A tuple representing the sizes of each batch.
        seq_lens (Tuple[int]): A tuple representing the sequence lengths for each batch.

    Returns:
        fmha.BlockDiagonalMask: An attention bias mask that is block-diagonal,
        constructed from the provided sequence lengths and batch sizes.
    """
    seqlens = []

    for b, x in zip(batch_sizes, seq_lens):
        seqlens.extend([x] * b)

    attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
    attn_bias._batch_sizes = batch_sizes

    return attn_bias


def get_attn_bias_and_cat(
    x_list: List[Tensor], selected_indices: Optional[List[Tensor]] = None
):
    """Generates an attention bias mask and concatenates selected tensors.

    Args:
        x_list (List[Tensor]): A list of tensors from which to generate the attention bias
            and perform concatenation. Each tensor should represent a sequence of embeddings.
        selected_indices(Optional[List[Tensor]], optional): A list of tensors containing indices
            that specify which elements to select from each corresponding tensor in `x_list`.
            Defaults to None, which means all elements will be used.

    Returns:
        Tuple[fmha.BlockDiagonalMask, Tensor]: A tuple containing:
            - fmha.BlockDiagonalMask: An attention bias mask constructed from the batch sizes and sequence lengths of the input tensors.
            - Tensor: A concatenated tensor that contains the selected elements based on the provided `selected_indices` or all elements if `selected_indices` is None.
    """
    if selected_indices is not None:
        batch_sizes = tuple(b.shape[0] for b in selected_indices)
    else:
        batch_sizes = tuple(x.shape[0] for x in x_list)

    seq_lens = tuple(x.shape[1] for x in x_list)

    if selected_indices is not None:
        cat_tensors = torch.cat(
            [
                x.flatten(1)[i.long()].flatten()
                for x, i in zip(x_list, selected_indices)
            ],
            dim=0,
        ).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = [x.reshape([1, -1, *x.shape[2:]]) for x in x_list]
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return get_attn_bias(batch_sizes, seq_lens), cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    """Applies stochastic depth by randomly dropping elements from the input tensors and adding a residual.

    Args:
        x_list (List[Tensor]): A list of input tensors from which elements will be randomly selected
            and potentially modified with a residual.
        residual_func (Callable[[Tensor, Any], Tensor]): A function that takes a tensor and an attention
            bias (or other context) as input, and returns a tensor representing the residual to be added.
        drop_ratio (float, optional): The probability of dropping elements from the batch. Defaults to 0.0,
        scaling_vector (Optional[Tensor], optional): A tensor for scaling the residual values. If provided,
            it scales the residual before applying it to the input tensors. Defaults to None.

    Returns:
        List[Tensor]: A list of tensors with the residuals applied based on stochastic depth and the selected indices.
    """
    selected_indices, residual_scale_factors = zip(
        *[get_subset_index_and_scale_factor(x, drop_ratio=drop_ratio) for x in x_list]
    )  # generate random set of indices for dropping samples in the batch

    # get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, selected_indices)

    # apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = [
        apply_residual(x, brange, residual, residual_scale_factor, scaling_vector)
        for x, brange, residual, residual_scale_factor in zip(
            x_list, selected_indices, residual_list, residual_scale_factors
        )
    ]

    return outputs


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable, drop_ratio: float = 0.0
) -> Tensor:
    """Applies stochastic depth by randomly dropping elements from the input tensor and adding a residual.

    Args:
        x (Tensor): The input tensor from which elements will be randomly selected and potentially modified
            with a residual. It is expected to be a batch of data.
        residual_func (Callable): A function that takes a tensor as input and computes the residual
            to be added back to the input tensor. This function should output a tensor of the same shape
            as the input tensor.
        drop_ratio (float, optional): The probability of dropping elements from the batch. Defaults to 0.0,

    Returns:
        Tensor: The output tensor after applying the residual, with certain elements potentially dropped
            based on the specified drop ratio.
    """
    selected_indices, residual_scale_factor = get_subset_index_and_scale_factor(
        x, drop_ratio=drop_ratio
    )  # extract subset of the batch

    # apply residual
    residual = residual_func(x[selected_indices])

    return apply_residual(x, selected_indices, residual, residual_scale_factor)


class NestedTensorBlock(Block):
    """A block that supports nested tensors with efficient attention mechanisms.

    This block utilizes the MemoryEfficientAttention class for attention operations and
    can handle both single tensors and lists of tensors. It implements residual connections
    with optional stochastic depth for training.
    """

    def __init__(
        self,
        *args,
        attn_class=MemoryEfficientAttention,
        use_custom_attention=True,
        **kwargs,
    ):
        """Initializes the NestedTensorBlock.

        Args:
            attn_class (Callable, optional): The class to use for attention operations. Defaults to MemoryEfficientAttention.
            use_custom_attention (bool): Whether to use memory_efficient_attention.
        """
        if attn_class != MemoryEfficientAttention:
            raise NotImplementedError("Only MemoryEfficentAttention is supported today.")
        super().__init__(*args, attn_class=attn_class, **kwargs)
        self.use_custom_attention = use_custom_attention

    def forward(
        self, x_or_x_list: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """Performs the forward pass for the NestedTensorBlock.

        This method processes either a single tensor or a list of tensors, applying
        attention and feedforward residuals with stochastic depth during training.

        Args:
            x_or_x_list (Union[Tensor, List[Tensor]]): The input tensor or list of tensors to process.

        Returns:
            Union[Tensor, List[Tensor]]: The processed output tensor or list of tensors.
        """
        if isinstance(x_or_x_list, Tensor):
            return self.forward([x_or_x_list])[0]

        x_list = x_or_x_list
        drop_ratio = (
            self.drop_path1.drop_prob if isinstance(self.drop_path1, DropPath) else 0.0
        )

        if self.training and drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                """Calculates the attention residual.

                Args:
                    x (Tensor): The input tensor for which to compute the residual.
                    attn_bias (Optional[Tensor]): Optional attention bias.

                Returns:
                    Tensor: The computed attention residual.
                """
                return self.attn(self.norm1(x), attn_bias=attn_bias, use_custom_attention=self.use_custom_attention)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                """Calculates the feedforward residual.

                Args:
                    x (Tensor): The input tensor for which to compute the residual.
                    attn_bias (Optional[Tensor]): Optional attention bias.

                Returns:
                    Tensor: The computed feedforward residual.
                """
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                drop_ratio=drop_ratio,
                scaling_vector=self.ls1.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                drop_ratio=drop_ratio,
                scaling_vector=self.ls2.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )
            return x_list

        def attn_residual_func_ls(x: Tensor, attn_bias=None) -> Tensor:
            """Computes the layer-scaled attention residual.

            Args:
                x (Tensor): The input tensor for which to compute the residual.
                attn_bias (Optional[Tensor]): Optional attention bias.

            Returns:
                Tensor: The computed layer-scaled attention residual.
            """
            return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias, use_custom_attention=self.use_custom_attention))

        def ffn_residual_func_ls(x: Tensor, attn_bias=None) -> Tensor:
            """Computes the layer-scaled feedforward residual.

            Args:
                x (Tensor): The input tensor for which to compute the residual.
                attn_bias (Optional[Tensor]): Optional attention bias.

            Returns:
                Tensor: The computed layer-scaled feedforward residual.
            """
            return self.ls2(self.mlp(self.norm2(x)))

        attn_bias, x = get_attn_bias_and_cat(x_list)
        x = x + attn_residual_func_ls(x, attn_bias=attn_bias)
        x = x + ffn_residual_func_ls(x)

        return attn_bias.split(x)


class DropPathBlock(Block):
    """A block that implements drop path regularization."""

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass for the DropPathBlock.

        Args:
            x (Tensor): The input tensor to be processed.

        Returns:
            Tensor: The output tensor after applying attention and feedforward layers with residuals.
        """
        def attn_residual_func(x: Tensor) -> Tensor:
            """Calculates the layer-scaled attention residual.

            Args:
                x (Tensor): The input tensor for which to compute the attention residual.

            Returns:
                Tensor: The computed layer-scaled attention residual.
            """
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            """Calculates the layer-scaled feedforward residual.

            Args:
                x (Tensor): The input tensor for which to compute the feedforward residual.

            Returns:
                Tensor: The computed layer-scaled feedforward residual.
            """
            return self.ls2(self.mlp(self.norm2(x)))

        drop_ratio = (
            self.drop_path1.drop_prob if isinstance(self.drop_path1, DropPath) else 0.0
        )

        if self.training and drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x, attn_residual_func, drop_ratio=drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, ffn_residual_func, drop_ratio=drop_ratio
            )
            return x

        x = x + self.drop_path1(attn_residual_func(x))
        x = x + self.drop_path2(ffn_residual_func(x))

        return x
