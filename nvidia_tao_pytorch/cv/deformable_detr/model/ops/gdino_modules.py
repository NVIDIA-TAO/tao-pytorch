# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Grounding DINO MSDeformAttn modules. """

import math
import warnings
from typing import Optional
import os
import sys

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from nvidia_tao_pytorch.cv.deformable_detr.model.ops.functions import MSDeformAttnFunction, load_ops
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import multi_scale_deformable_attn_pytorch, precise_msda_enabled


# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class GDINOMultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Grounding DINO."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        """
        Args:
            embed_dim (int): The embedding dimension of Attention. Default: 256.
            num_heads (int): The number of attention heads. Default: 8.
            num_levels (int): The number of feature map used in Attention. Default: 4.
            num_points (int): The number of sampling points for each query
                in each head. Default: 4.
            img2col_steps (int): The step used in image_to_column. Defualt: 64.
                dropout (float): Dropout layer used in output. Default: 0.1.
            batch_first (bool): if ``True``, then the input and output tensor will be
                provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        head_dim = embed_dim // num_heads

        self.batch_first = batch_first

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()
        # load custom ops
        ops_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = f"MultiScaleDeformableAttention.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
        load_ops(ops_dir, lib_name)

    def _reset_parameters(self):
        return self.init_weights()

    def init_weights(self):
        """
        Default initialization for Parameters of Module.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward Function of MultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """
        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        spatial_shapes = spatial_shapes.long()
        level_start_index = level_start_index.long()

        if torch.onnx.is_in_onnx_export():
            if torch.cuda.is_available() and value.is_cuda:
                output = torch.ops.nvidia.MultiscaleDeformableAttnPlugin_TRT(
                    value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights)

            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights
                )
        else:
            # precise_msda routes CUDA tensors through the deterministic pure-PyTorch
            # path below (the custom CUDA op's atomicAdd backward is non-deterministic).
            if torch.cuda.is_available() and value.is_cuda and not precise_msda_enabled():
                half_float = False
                if value.dtype in [torch.float16, torch.bfloat16]:
                    half_float = value.dtype
                    value = value.float()
                    sampling_locations = sampling_locations.float()
                    attention_weights = attention_weights.float()

                output = MSDeformAttnFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step
                )
                if half_float:
                    output = output.to(half_float)
            else:
                half_float = False
                if value.dtype in [torch.float16, torch.bfloat16]:
                    half_float = value.dtype
                    value = value.float()
                    sampling_locations = sampling_locations.float()
                    attention_weights = attention_weights.float()

                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights,
                    deterministic=precise_msda_enabled()
                )
                if half_float:
                    output = output.to(half_float)

        output = output.view(bs, num_query, self.embed_dim)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
