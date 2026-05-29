# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom replacement for `torch.nn.functional.conv2d` that supports
arbitrarily high order gradients with zero performance penalty.
"""

import contextlib
import torch
from pkg_resources import parse_version


# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

enabled = False                     # Enable the custom op by setting this to true.
weight_gradients_disabled = False   # Forcefully disable computation of gradients with respect to the weights.

# Fix backwards-incompatible change in PyTorch 1.11.0
# https://github.com/autonomousvision/stylegan-xl/pull/117
# https://github.com/pytorch/pytorch/issues/74437
# https://github.com/NVlabs/stylegan2-ada-pytorch/pull/299
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a')  # Allow prerelease builds of 1.11


@contextlib.contextmanager
def no_weight_gradients(disable=True):
    """Context manager to disable weight gradients temporarily.

    Args:
        disable (bool, optional): If True, disables computation of weight gradients. Defaults to True.
    """
    global weight_gradients_disabled  # pylint: disable=global-statement
    old = weight_gradients_disabled
    if disable:
        weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Custom 2D convolution function with optional gradient fix.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding added to the input. Defaults to 0.
        dilation (int or tuple, optional): Dilation of the convolution. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.

    Returns:
        torch.Tensor: The output tensor after convolution.
    """
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=False, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Custom 2D transposed convolution function with optional gradient fix.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding added to the input. Defaults to 0.
        output_padding (int or tuple, optional): Extra padding added to the output. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        dilation (int or tuple, optional): Dilation of the convolution. Defaults to 1.

    Returns:
        torch.Tensor: The output tensor after transposed convolution.
    """
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=True, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation).apply(input, weight, bias)
    return torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)


def _should_use_custom_op(input):
    """Determines whether the custom convolution operation should be used.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        bool: True if the custom operation should be used, False otherwise.
    """
    assert isinstance(input, torch.Tensor)
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False
    elif _use_pytorch_1_11_api:
        # The work-around code doesn't work on PyTorch 1.11.0 onwards
        return False
    elif input.device.type != 'cuda':
        return False
    return True


def _tuple_of_ints(xs, ndim):
    """Ensures the input is a tuple of integers with a specified dimension.

    Args:
        xs (int, tuple, or list): Input integer, tuple, or list to be converted.
        ndim (int): Number of dimensions.

    Returns:
        tuple: Tuple of integers with the specified number of dimensions.
    """
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


_conv2d_gradfix_cache = dict()
_null_tensor = torch.empty([0])


def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    """Custom function to handle gradient fixes for 2D convolution and transposed convolution operations.

    Args:
        transpose (bool): Whether to use transposed convolution.
        weight_shape (tuple): Shape of the weight tensor.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding added to the input.
        output_padding (tuple): Extra padding added to the output for transposed convolution.
        dilation (tuple): Dilation of the convolution.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        Conv2d: Custom autograd function for 2D convolution.
    """
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else:  # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))

    # Helpers.
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        """calculates the output padding for a convolution transpose operation
        based on the input and output shapes, stride, padding, and dilation.

        Args:
            input_shape (tuple): The shape of the input tensor.
            output_shape (tuple): The shape of the output tensor.

        Returns:
            list: A list of padding values for each dimension.
        """
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2] -
            (output_shape[i + 2] - 1) * stride[i] -
            (1 - 2 * padding[i]) -
            dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):
        """
        Custom autograd function for forward and backward pass of 2D convolutions.
        It includes optimizations for specific cases like 1x1 convolutions using cuBLAS.
        """

        @staticmethod
        def forward(ctx, input, weight, bias):
            """Forward pass of the convolution.

            Args:
                ctx: Context object for saving information for backward pass.
                input (torch.Tensor): Input tensor.
                weight (torch.Tensor): Weight tensor (filters).
                bias (torch.Tensor): Optional bias tensor.

            Returns:
                torch.Tensor: Output tensor after applying convolution.
            """
            assert weight.shape == weight_shape
            ctx.save_for_backward(
                input if weight.requires_grad else _null_tensor,
                weight if input.requires_grad else _null_tensor,
            )
            ctx.input_shape = input.shape

            # Simple 1x1 convolution => cuBLAS (only on Volta, not on Ampere).
            if weight_shape[2:] == stride == dilation == (1, 1) and padding == (0, 0) and torch.cuda.get_device_capability(input.device) < (8, 0):
                a = weight.reshape(groups, weight_shape[0] // groups, weight_shape[1])
                b = input.reshape(input.shape[0], groups, input.shape[1] // groups, -1)
                c = (a.transpose(1, 2) if transpose else a) @ b.permute(1, 2, 0, 3).flatten(2)
                c = c.reshape(-1, input.shape[0], *input.shape[2:]).transpose(0, 1)
                c = c if bias is None else c + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                return c.contiguous(memory_format=(torch.channels_last if input.stride(1) == 1 else torch.contiguous_format))

            # General case => cuDNN.
            if transpose:
                return torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs)
            return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass of the convolution.

            Args:
                ctx: Context object holding saved tensors.
                grad_output (torch.Tensor): Gradient of the loss w.r.t the output.

            Returns:
                tuple: Gradients w.r.t input, weight, and bias.
            """
            input, weight = ctx.saved_tensors
            input_shape = ctx.input_shape
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input_shape, output_shape=grad_output.shape)
                op = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs)
                grad_input = op.apply(grad_output, weight, None)
                assert grad_input.shape == input_shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):
        """
        Custom autograd function to compute the gradient of the convolution with respect to the weights.
        It includes optimizations for specific cases like 1x1 convolutions.
        """

        @staticmethod
        def forward(ctx, grad_output, input):
            """Forward pass for computing weight gradient.

            Args:
                ctx: Context object for saving information for backward pass.
                grad_output (torch.Tensor): Gradient of the loss w.r.t the output.
                input (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Gradient of the loss w.r.t the weights.
            """
            ctx.save_for_backward(
                grad_output if input.requires_grad else _null_tensor,
                input if grad_output.requires_grad else _null_tensor,
            )
            ctx.grad_output_shape = grad_output.shape
            ctx.input_shape = input.shape

            # Simple 1x1 convolution => cuBLAS (on both Volta and Ampere).
            if weight_shape[2:] == stride == dilation == (1, 1) and padding == (0, 0):
                a = grad_output.reshape(grad_output.shape[0], groups, grad_output.shape[1] // groups, -1).permute(1, 2, 0, 3).flatten(2)
                b = input.reshape(input.shape[0], groups, input.shape[1] // groups, -1).permute(1, 2, 0, 3).flatten(2)
                c = (b @ a.transpose(1, 2) if transpose else a @ b.transpose(1, 2)).reshape(weight_shape)
                return c.contiguous(memory_format=(torch.channels_last if input.stride(1) == 1 else torch.contiguous_format))

            # General case => cuDNN.
            name = 'aten::cudnn_convolution_transpose_backward_weight' if transpose else 'aten::cudnn_convolution_backward_weight'
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32]

            return torch._C._jit_get_operation(name)(weight_shape, grad_output, input, padding, stride, dilation, groups, *flags)

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            """Backward pass to compute gradients of the gradients.

            Args:
                ctx: Context object holding saved tensors.
                grad2_grad_weight (torch.Tensor): Gradient of the loss w.r.t the weight gradient.

            Returns:
                tuple: Gradients w.r.t the input and output.
            """
            grad_output, input = ctx.saved_tensors
            grad_output_shape = ctx.grad_output_shape
            input_shape = ctx.input_shape
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output_shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input_shape, output_shape=grad_output_shape)
                op = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs)
                grad2_input = op.apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input_shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d
