# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtered Leaky-ReLU"""

import os
import numpy as np
import sys
import torch
import warnings

from .. import misc
from . import upfirdn2d
from . import bias_act


_plugin = None


def _init():
    """Initialize the custom filtered Leaky ReLU plugin.

    This function initializes the custom plugin for the `filtered_lrelu` operation
    by compiling and loading the necessary CUDA and C++ source files, along with
    header files. The plugin is loaded only once and reused in subsequent operations.

    Returns:
        bool: A boolean indicating the success of the plugin initialization.
    """
    global _plugin  # pylint: disable=global-statement
    if _plugin is None:
        # # Method 1: Prebuilt .so with 'importlib.import_module', This method should use PYBIND11_MODULE in C++, and call the plugin in python like: _plugin.filtered_lrelu(input_tensor)
        # import importlib
        # _plugin = importlib.import_module("nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops.filtered_lrelu_plugin")  # This method should use PYBIND11_MODULE in C++, and call the plugin in python like: _plugin.filtered_lrelu(input_tensor)

        # # Method 2: Prebuilt .so with 'torch.ops.load_library', This method should use torch::RegisterOperators in C++, and call the plugin in python like: _plugin.filtered_lrelu(input_tensor)
        ops_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = f"filtered_lrelu_plugin.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
        torch.ops.load_library(os.path.join(ops_dir, lib_name))
        _plugin = torch.ops.nvidia

    return True


def _get_filter_size(f):
    """Retrieve the dimensions of the provided filter.

    This function returns the width and height of the filter. If no filter is provided, it returns (1, 1).
    The function supports both 1D (separable) and 2D filters.

    Args:
        f (torch.Tensor or None): A filter tensor with one or two dimensions. If `None`, returns (1, 1).

    Returns:
        tuple: A tuple containing two integers, (filter_width, filter_height).

    Raises:
        AssertionError: If the input filter is not a tensor or has more than two dimensions.
    """
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0]  # width, height


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1


def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda', export=False):
    r"""Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if export:
        impl = 'ref'
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    # The export flag is passed into _filtered_lrelu_ref also. Though it is ref implementation, it may still use custom ops like bias_act and upfirdn2d
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter, export=export)


def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, export=False):
    """Slow and memory-inefficient reference implementation of `filtered_lrelu()` using
    existing `upfirdn2d()` and `bias_act()` ops.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        fu (torch.Tensor, optional): Filter for upsampling. Defaults to None.
        fd (torch.Tensor, optional): Filter for downsampling. Defaults to None.
        b (torch.Tensor, optional): Bias tensor. Defaults to None.
        up (int, optional): Upsampling factor. Defaults to 1.
        down (int, optional): Downsampling factor. Defaults to 1.
        padding (int, optional): Amount of padding. Defaults to 0.
        gain (float, optional): Gain for the activation. Defaults to sqrt(2).
        slope (float, optional): Negative slope for Leaky ReLU. Defaults to 0.2.
        clamp (float, optional): Clamping value. Defaults to None.
        flip_filter (bool, optional): Whether to flip the filter. Defaults to False.
        export (bool): The export flag is passed into _filtered_lrelu_ref. Though it is ref implementation, it may still use custom ops like bias_act and upfirdn2d

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, channels, out_height, out_width).
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

    # Compute using existing ops.
    x = bias_act.bias_act(x=x, b=b, export=export)  # Apply bias.
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter, export=export)  # Upsample.
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp, export=export)  # Bias, leaky ReLU, clamp.
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter, export=export)  # Downsample.

    # Check output shape & dtype.
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x


_filtered_lrelu_cuda_cache = dict()


def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.

    Args:
        up (int): Upsampling factor. Defaults to 1.
        down (int): Downsampling factor. Defaults to 1.
        padding (int): Amount of padding. Defaults to 0.
        gain (float): Gain for the activation. Defaults to sqrt(2).
        slope (float): Negative slope for Leaky ReLU. Defaults to 0.2.
        clamp (float, optional): Clamping value. Defaults to None.
        flip_filter (bool, optional): Whether to flip the filter. Defaults to False.

    Returns:
        function: A callable for the filtered Leaky ReLU operation.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    # Forward op.
    class FilteredLReluCuda(torch.autograd.Function):
        """Custom autograd function for the CUDA implementation of filtered_lrelu."""

        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):  # pylint: disable=arguments-differ
            """Forward pass for filtered Leaky ReLU operation.

            Args:
                ctx: The context object to save information for backward pass.
                x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
                fu (torch.Tensor): Filter for upsampling.
                fd (torch.Tensor): Filter for downsampling.
                b (torch.Tensor): Bias tensor.
                si (torch.Tensor): Sign input tensor.
                sx (int): Horizontal stride.
                sy (int): Vertical stride.

            Returns:
                torch.Tensor: Output tensor after applying filtered Leaky ReLU.
            """
            assert isinstance(x, torch.Tensor) and x.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = torch.empty([0])

            # Missing bias tensor.
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            x = x.contiguous()
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)

            # Call C++/Cuda plugin if datatype is supported.
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                    warnings.warn("filtered_lrelu called with non-default cuda stream but concurrent execution is not supported", RuntimeWarning)
                y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1

            # No Cuda kernel found? Fall back to generic implementation. Still more memory efficient than the reference implementation because
            # only the bit-packed sign tensor is retained for gradient computation.
            if return_code < 0:
                warnings.warn("filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback", RuntimeWarning)

                y = x.add(b.unsqueeze(-1).unsqueeze(-1))  # Add bias.
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)  # Upsample.
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs)  # Activation function and sign handling. Modifies y in-place.
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter)  # Downsample.

            # Prepare for gradient computation.
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            """Backward pass for filtered Leaky ReLU operation.

            Args:
                ctx: The context object with saved tensors.
                grad_output (torch.Tensor): Gradient of the loss with respect to the output.

            Returns:
                tuple: Gradients with respect to inputs (x, fu, fd, b, si).
            """
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = None  # 0
            dfu = None; assert not ctx.needs_input_grad[1]  # noqa: E702
            dfd = None; assert not ctx.needs_input_grad[2]  # noqa: E702
            db = None  # 3
            dsi = None; assert not ctx.needs_input_grad[4]  # noqa: E702
            dsx = None; assert not ctx.needs_input_grad[5]  # noqa: E702
            dsy = None; assert not ctx.needs_input_grad[6]  # noqa: E702

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda
