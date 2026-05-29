# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused multiply-add, with slightly faster gradients than `torch.addcmul()`."""

import torch


def fma(a, b, c):  # => a * b + c
    """Computes the fused multiply-add operation: a * b + c.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.
        c (torch.Tensor): The tensor to add after the multiplication.

    Returns:
        torch.Tensor: The result of the operation a * b + c.
    """
    return _FusedMultiplyAdd.apply(a, b, c)


class _FusedMultiplyAdd(torch.autograd.Function):  # a * b + c
    """Custom autograd function for performing the fused multiply-add operation: a * b + c."""

    @staticmethod
    def forward(ctx, a, b, c):  # pylint: disable=arguments-differ
        """Forward pass for the fused multiply-add operation.

        Args:
            ctx: The context object to save information for the backward pass.
            a (torch.Tensor): The first input tensor.
            b (torch.Tensor): The second input tensor.
            c (torch.Tensor): The tensor to add after the multiplication.

        Returns:
            torch.Tensor: The output tensor resulting from the operation.
        """
        out = torch.addcmul(c, a, b)
        ctx.save_for_backward(a, b)
        ctx.c_shape = c.shape
        return out

    @staticmethod
    def backward(ctx, dout):  # pylint: disable=arguments-differ
        """Backward pass for the fused multiply-add operation.

        Computes gradients of the inputs a, b, and c with respect to the output.

        Args:
            ctx: The context object containing saved tensors.
            dout (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: Gradients with respect to inputs (da, db, dc).
        """
        a, b = ctx.saved_tensors
        c_shape = ctx.c_shape
        da = None
        db = None
        dc = None

        if ctx.needs_input_grad[0]:
            da = _unbroadcast(dout * b, a.shape)

        if ctx.needs_input_grad[1]:
            db = _unbroadcast(dout * a, b.shape)

        if ctx.needs_input_grad[2]:
            dc = _unbroadcast(dout, c_shape)

        return da, db, dc


def _unbroadcast(x, shape):
    """Unbroadcasts the input tensor x to match the given shape.

    Args:
        x (torch.Tensor): The input tensor to unbroadcast.
        shape (tuple): The target shape to match.

    Returns:
        torch.Tensor: The unbroadcasted tensor with the specified shape.
    """
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims + 1:])
    assert x.shape == shape
    return x
