# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`.
"""

import torch
from pkg_resources import parse_version


# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

enabled = False  # Enable the custom op by setting this to true.

# Fix backwards-incompatible change in PyTorch 1.11.0
# https://github.com/autonomousvision/stylegan-xl/pull/117
# https://github.com/pytorch/pytorch/issues/74437
# https://github.com/NVlabs/stylegan2-ada-pytorch/pull/299
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a')  # Allow prerelease builds of 1.11
_use_pytorch_1_12_api = parse_version(torch.__version__) >= parse_version('1.12.0a')  # Allow prerelease builds of 1.12


def grid_sample(input, grid):
    """Applies a 2D grid sampling operation on the input tensor.

    Args:
        input (torch.Tensor): The input tensor of shape (N, C, H, W).
        grid (torch.Tensor): The grid tensor of shape (N, H_out, W_out, 2).

    Returns:
        torch.Tensor: The sampled output tensor.
    """
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _should_use_custom_op():
    """Checks whether the custom operation should be used.

    Returns:
        bool: True if the custom operation is enabled, False otherwise.
    """
    return enabled


class _GridSample2dForward(torch.autograd.Function):
    """Custom autograd function for the forward pass of 2D grid sampling."""

    @staticmethod
    def forward(ctx, input, grid):
        """Performs the forward pass for the 2D grid sampling operation.

        Args:
            ctx: The context object to save tensors for the backward pass.
            input (torch.Tensor): The input tensor.
            grid (torch.Tensor): The grid tensor for sampling.

        Returns:
            torch.Tensor: The output tensor after grid sampling.
        """
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Performs the backward pass for the 2D grid sampling operation.

        Args:
            ctx: The context object containing saved tensors.
            grad_output (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: Gradients with respect to inputs (grad_input, grad_grid).
        """
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid


class _GridSample2dBackward(torch.autograd.Function):
    """Custom autograd function for the backward pass of 2D grid sampling."""

    @staticmethod
    def forward(ctx, grad_output, input, grid):
        """Computes the gradients during the backward pass.

        Args:
            ctx: The context object to save information for the backward pass.
            grad_output (torch.Tensor): The gradient of the output.
            input (torch.Tensor): The input tensor.
            grid (torch.Tensor): The grid tensor for sampling.

        Returns:
            tuple: Gradients with respect to input and grid.
        """
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        if _use_pytorch_1_12_api:
            op = op[0]
        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        """Computes the second backward pass gradients.

        Args:
            ctx: The context object containing saved tensors.
            grad2_grad_input (torch.Tensor): Gradient from the first backward pass with respect to input.
            grad2_grad_grid (torch.Tensor): Gradient from the first backward pass with respect to grid.

        Returns:
            tuple: Gradients with respect to output and input.
        """
        _ = grad2_grad_grid  # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)

        assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid
