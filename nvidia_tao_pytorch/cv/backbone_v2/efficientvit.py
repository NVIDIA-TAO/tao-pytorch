# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

"""EfficientViT backbone module.

This module provides EfficientViT implementations for the TAO PyTorch framework.
EfficientViT is a vision transformer architecture designed for high-resolution
dense prediction tasks with efficient multi-scale linear attention mechanisms.

The EfficientViT architecture was introduced in "EfficientViT: Multi-Scale Linear
Attention for High-Resolution Dense Prediction" by Cai et al. This implementation
extends the original EfficientViT to provide additional functionality for backbone
integration and feature extraction.

Key Features:
- Multi-scale linear attention for efficient computation
- High-resolution dense prediction capabilities
- Support for multiple model sizes (B0-B3, L0-L3)
- LiteMLA (Lightweight Multi-scale Linear Attention) mechanism
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design with good accuracy/speed balance

Classes:
    ConvLayer: Basic convolutional layer with normalization and activation
    UpSampleLayer: Upsampling layer for feature resolution increase
    DSConv: Depthwise separable convolution
    MBConv: Mobile inverted bottleneck convolution
    FusedMBConv: Fused mobile inverted bottleneck convolution
    ResBlock: Residual block for feature processing
    LiteMLA: Lightweight Multi-scale Linear Attention module
    EfficientViTBlock: EfficientViT transformer block
    ResidualBlock: Residual block with main and shortcut paths
    OpSequential: Sequential operation container
    EfficientViT: EfficientViT model with enhanced TAO integration
    EfficientViTLarge: Large EfficientViT model variant

Functions:
    efficientvit_b0: EfficientViT B0 model
    efficientvit_b1: EfficientViT B1 model
    efficientvit_b2: EfficientViT B2 model
    efficientvit_b3: EfficientViT B3 model
    efficientvit_l0: EfficientViT L0 model
    efficientvit_l1: EfficientViT L1 model
    efficientvit_l2: EfficientViT L2 model
    efficientvit_l3: EfficientViT L3 model

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import efficientvit_b0
    >>> model = efficientvit_b0(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

Note:
    The TIMM implementation doesn't match
    the performance of the original implementation. Therefore, we stick to the
    original implementation available at:
    https://github.com/mit-han-lab/efficientvit
"""

from functools import partial
from inspect import signature
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


################################################################################
#                             Utility                                          #
################################################################################


def val2list(x, repeat_time=1):
    """Convert value to list.

    This utility function converts various input types to a list format.
    If the input is already a list or tuple, it returns it as a list.
    Otherwise, it creates a list with the input value repeated.

    Args:
        x: Input value to convert to list
        repeat_time (int, optional): Number of times to repeat the value if it's not already a list.
            Defaults to 1.

    Returns:
        list: List containing the input value(s)

    Example:
        >>> val2list(3, repeat_time=2)
        [3, 3]
        >>> val2list([1, 2, 3])
        [1, 2, 3]
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x, min_len=1, idx_repeat=-1):
    """Convert value to tuple.

    This utility function converts various input types to a tuple format.
    It ensures the tuple has at least the specified minimum length by repeating
    elements if necessary.

    Args:
        x: Input value to convert to tuple
        min_len (int, optional): Minimum length of the output tuple. Defaults to 1.
        idx_repeat (int, optional): Index of element to repeat if length is insufficient.
            Defaults to -1 (last element).

    Returns:
        tuple: Tuple containing the input value(s)

    Example:
        >>> val2tuple(3, min_len=3)
        (3, 3, 3)
        >>> val2tuple([1, 2], min_len=4)
        (1, 2, 2, 2)
    """
    x = val2list(x)
    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    """Build keyword arguments from config dictionary.

    This utility function filters a configuration dictionary to only include
    parameters that are valid for the target function.

    Args:
        config (dict): Configuration dictionary containing parameter names and values
        target_func (callable): Target function to check valid parameters against

    Returns:
        dict: Filtered dictionary containing only valid parameters for the target function

    Example:
        >>> config = {'a': 1, 'b': 2, 'invalid': 3}
        >>> def target_func(a, b):
        ...     pass
        >>> build_kwargs_from_config(config, target_func)
        {'a': 1, 'b': 2}
    """
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_act(name: str, **kwargs):
    """Build activation layer from name.

    This utility function creates activation layers based on the provided name.
    It supports various activation functions commonly used in neural networks.

    Args:
        name (str): Name of the activation function
        **kwargs: Additional arguments passed to the activation function

    Returns:
        nn.Module or None: Activation layer or None if name is not recognized

    Supported activations:
        - "relu": ReLU activation
        - "relu6": ReLU6 activation (clamped at 6)
        - "hswish": Hard Swish activation
        - "silu": SiLU/Swish activation
        - "gelu": GELU activation with tanh approximation

    Example:
        >>> act = build_act("relu")
        >>> act = build_act("gelu")
    """
    REGISTERED_ACT_DICT: dict[str, type] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": partial(nn.GELU, approximate="tanh"),
    }
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]):
    """Calculate same padding for convolution layers.

    This utility function calculates the padding required to maintain the same
    output size as input size for convolution operations.

    Args:
        kernel_size (Union[int, Tuple[int, ...]]): Kernel size(s) for convolution

    Returns:
        Union[int, Tuple[int, ...]]: Padding value(s) for same output size

    Example:
        >>> get_same_padding(3)
        1
        >>> get_same_padding((3, 5))
        (1, 2)
    """
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(x, size=None, scale_factor=None, mode="bicubic", align_corners=False):
    """Resize tensor with various interpolation modes.

    This utility function resizes input tensors using different interpolation
    methods. It supports both bilinear and bicubic interpolation modes.

    Args:
        x (torch.Tensor): Input tensor to resize
        size (tuple, optional): Target size (H, W). Defaults to None.
        scale_factor (float, optional): Scale factor for resizing. Defaults to None.
        mode (str, optional): Interpolation mode. Defaults to "bicubic".
        align_corners (bool, optional): Whether to align corners. Defaults to False.

    Returns:
        torch.Tensor: Resized tensor

    Example:
        >>> x = torch.randn(1, 3, 224, 224)
        >>> resized = resize(x, size=(112, 112), mode="bilinear")
    """
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


################################################################################
#                             Basic Layers                                     #
################################################################################


class ConvLayer(nn.Module):
    """Basic convolutional layer with normalization and activation.

    This class defines a standard convolutional layer that includes dropout,
    convolution, normalization, and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the convolution. Defaults to 3.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        dilation (int, optional): Dilation for the convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        use_bias (bool, optional): Whether to use bias in the convolution. Defaults to False.
        dropout (float, optional): Dropout rate for the layer. Defaults to 0.
        norm (str, optional): Normalization layer type. Defaults to "bn2d".
        act_func (str, optional): Activation function type. Defaults to "relu".

    Attributes:
        dropout (nn.Dropout2d or None): Dropout layer if dropout > 0.
        conv (nn.Conv2d): Convolutional layer.
        norm (nn.BatchNorm2d or None): Normalization layer.
        act (nn.Module or None): Activation layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        if norm == "bn2d":
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm {norm} not implemented.")
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor):
        """Forward pass of the ConvLayer.

        This method performs the forward computation of the convolutional layer.
        It includes dropout, convolution, normalization, and activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution and all operations.
        """
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    """Upsampling layer for feature resolution increase.

    This class provides an upsampling layer that can be used to increase the
    resolution of feature maps. It supports various interpolation modes.

    Args:
        mode (str, optional): Interpolation mode. Defaults to `bicubic`.
        size (tuple, optional): Target size (H, W). Defaults to `None`.
        factor (int, optional): Factor for upsampling. Defaults to `2`.
        align_corners (bool, optional): Whether to align corners. Defaults to `False`.

    Attributes:
        mode (str): Interpolation mode.
        size (list or None): Target size as a list.
        factor (int or None): Upsampling factor.
        align_corners (bool): Whether to align corners.
    """

    def __init__(
        self,
        mode="bicubic",
        size=None,
        factor=2,
        align_corners=False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor):
        """Forward pass of the UpSampleLayer.

        This method performs the upsampling operation. It checks if the input
        tensor already matches the target size or if the factor is 1.
        It also handles potential dtype conversions.

        Args:
            x (torch.Tensor): Input tensor to upsample.

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


################################################################################
#                             Basic Blocks                                     #
################################################################################


class DSConv(nn.Module):
    """Depthwise separable convolution.

    This class implements a depthwise separable convolution (DSConv) which
    consists of a depthwise convolution followed by a pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 3.
        stride (int, optional): Stride for the depthwise convolution. Defaults to 1.
        use_bias (bool, optional): Whether to use bias in the depthwise convolution. Defaults to False.
        norm (tuple, optional): Normalization layer types for depthwise and pointwise convolutions.
            Defaults to ("bn2d", "bn2d").
        act_func (tuple, optional): Activation function types for depthwise and pointwise convolutions.
            Defaults to ("relu6", None).

    Attributes:
        depth_conv (ConvLayer): Depthwise convolutional layer.
        point_conv (ConvLayer): Pointwise convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor):
        """Forward pass of the `DSConv`.

        This method performs the depthwise separable convolution.
        It first applies the depthwise convolution and then the pointwise convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after `DSConv`.
        """
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck convolution.

    This class implements a Mobile inverted bottleneck convolution (MBConv)
    which is a building block for EfficientViT. It consists of an inverted
    bottleneck, a depthwise convolution, and a pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 3.
        stride (int, optional): Stride for the MBConv. Defaults to 1.
        mid_channels (int, optional): Number of channels in the middle layer.
            If None, it is calculated as in_channels * expand_ratio. Defaults to None.
        expand_ratio (float, optional): Expansion ratio for the inverted bottleneck. Defaults to 6.
        use_bias (tuple, optional): Whether to use bias in the inverted bottleneck,
            depthwise convolution, and pointwise convolution. Defaults to (True, True, False).
        norm (tuple, optional): Normalization layer types for inverted bottleneck,
            depthwise convolution, and pointwise convolution. Defaults to ("bn2d", "bn2d", "bn2d").
        act_func (tuple, optional): Activation function types for inverted bottleneck,
            depthwise convolution, and pointwise convolution. Defaults to ("relu6", "relu6", None).

    Attributes:
        inverted_conv (ConvLayer): Inverted bottleneck convolutional layer.
        depth_conv (ConvLayer): Depthwise convolutional layer.
        point_conv (ConvLayer): Pointwise convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MBConv.

        This method performs the MBConv operation.
        It first applies the inverted bottleneck, then the depthwise convolution,
        and finally the pointwise convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MBConv.
        """
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    """Fused Mobile inverted bottleneck convolution layer.

    This class implements a fused Mobile inverted bottleneck convolution
    (FusedMBConv) which combines spatial convolution and pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the spatial convolution. Defaults to 3.
        stride (int, optional): Stride for the spatial convolution. Defaults to 1.
        mid_channels (int, optional): Number of channels in the middle layer.
            If None, it is calculated as in_channels * expand_ratio. Defaults to None.
        expand_ratio (float, optional): Expansion ratio for the spatial convolution. Defaults to 6.
        groups (int, optional): Number of groups for the spatial convolution. Defaults to 1.
        use_bias (tuple, optional): Whether to use bias in the spatial convolution and pointwise convolution.
            Defaults to (True, False).
        norm (tuple, optional): Normalization layer types for spatial convolution and pointwise convolution.
            Defaults to ("bn2d", "bn2d").
        act_func (tuple, optional): Activation function types for spatial convolution and pointwise convolution.
            Defaults to ("relu6", None).

    Attributes:
        spatial_conv (ConvLayer): Spatial convolutional layer.
        point_conv (ConvLayer): Pointwise convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FusedMBConv.

        This method performs the fused MBConv operation.
        It first applies the spatial convolution and then the pointwise convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after FusedMBConv.
        """
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    """Residual block for feature processing.

    This class implements a standard residual block with two convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 3.
        stride (int, optional): Stride for the convolutional layers. Defaults to 1.
        mid_channels (int, optional): Number of channels in the middle layer.
            If None, it is calculated as in_channels * expand_ratio. Defaults to None.
        expand_ratio (float, optional): Expansion ratio for the middle layer. Defaults to 1.
        use_bias (bool, optional): Whether to use bias in the convolutional layers. Defaults to False.
        norm (tuple, optional): Normalization layer types for the convolutional layers.
            Defaults to ("bn2d", "bn2d").
        act_func (tuple, optional): Activation function types for the convolutional layers.
            Defaults to ("relu6", None).

    Attributes:
        conv1 (ConvLayer): First convolutional layer.
        conv2 (ConvLayer): Second convolutional layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the `ResBlock`.

        This method performs the residual block operation.
        It first applies the first convolutional layer, then the second convolutional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after `ResBlock`.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=None,
        heads_ratio=1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales=(5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)), out_channels, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor):
        """ReLU Linear attention."""
        B, _, H, W = list(qkv.size())
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = torch.reshape(
            qkv,
            (B, -1, 3 * self.dim, H * W),
        )
        q, k, v = (
            qkv[:, :, 0: self.dim],
            qkv[:, :, self.dim: 2 * self.dim],
            qkv[:, :, 2 * self.dim:],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor):
        """ReLU Quadratic attention."""
        B, _, H, W = list(qkv.size())
        qkv = torch.reshape(
            qkv,
            (B, -1, 3 * self.dim, H * W),
        )
        q, k, v = (
            qkv[:, :, 0: self.dim],
            qkv[:, :, self.dim: 2 * self.dim],
            qkv[:, :, 2 * self.dim:],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor):
        """Forward pass of the LiteMLA.

        This method performs the LiteMLA operation.
        It generates multi-scale q, k, v, applies aggregation, and then
        applies the ReLU linear or quadratic attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after LiteMLA.
        """
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)
        return out


class EfficientViTBlock(nn.Module):
    """EfficientViT transformer block.

    This class implements a single EfficientViT transformer block which
    consists of a LiteMLA module and a local MBConv module.

    Args:
        in_channels (int): Number of input channels.
        heads_ratio (float, optional): Ratio of heads for the LiteMLA module. Defaults to 1.0.
        dim (int, optional): Dimension of the head for LiteMLA. Defaults to 32.
        expand_ratio (float, optional): Expansion ratio for the MBConv block. Defaults to 4.
        scales (tuple, optional): Scales for LiteMLA aggregation. Defaults to (5,).
        norm (str, optional): Normalization layer type. Defaults to "bn2d".
        act_func (str, optional): Activation function type. Defaults to "hswish".

    Attributes:
        context_module (ResidualBlock): LiteMLA module.
        local_module (ResidualBlock): MBConv module.
    """

    def __init__(
        self, in_channels, heads_ratio=1.0, dim=32, expand_ratio=4, scales=(5,), norm="bn2d", act_func="hswish"
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            nn.Identity(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, nn.Identity())

    def forward(self, x: torch.Tensor):
        """Forward pass of the EfficientViTBlock.

        This method performs the EfficientViTBlock operation.
        It first applies the LiteMLA module and then the local MBConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after EfficientViTBlock.
        """
        x = self.context_module(x)
        x = self.local_module(x)
        return x


################################################################################
#                             Functional Blocks                                #
################################################################################


class ResidualBlock(nn.Module):
    """Residual block with main and shortcut paths.

    This class implements a standard residual block with a main path and
    an optional shortcut path.

    Args:
        main (nn.Module): Main module (e.g., ConvLayer, MBConv).
        shortcut (nn.Module or None): Shortcut module (e.g., Identity, ConvLayer).
        post_act (str, optional): Activation function type for the output. Defaults to None.
        pre_norm (str, optional): Normalization layer type for the main path. Defaults to None.

    Attributes:
        pre_norm (nn.Module or None): Pre-normalization layer.
        main (nn.Module): Main module.
        shortcut (nn.Module or None): Shortcut module.
        post_act (nn.Module or None): Post-activation layer.
    """

    def __init__(self, main, shortcut, post_act=None, pre_norm=None):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor):
        """Forward main path of the ResidualBlock.

        This method performs the forward computation of the main path.
        It includes pre-normalization if applicable.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after main path.
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        return self.main(x)

    def forward(self, x: torch.Tensor):
        """Forward pass of the ResidualBlock.

        This method performs the residual block operation.
        It adds the output of the main path and the shortcut path,
        and applies post-activation if applicable.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after ResidualBlock.
        """
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    """Sequential operation container.

    This class provides a container for a sequence of operations.
    It filters out None operations and creates a ModuleList.

    Args:
        op_list (list): List of operations to be sequenced.

    Attributes:
        op_list (nn.ModuleList): List of valid operations.
    """

    def __init__(self, op_list):
        super().__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor):
        """Forward pass of the OpSequential.

        This method performs the sequential operation.
        It applies each operation in the op_list to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after sequential operations.
        """
        for op in self.op_list:
            x = op(x)
        return x


################################################################################
#                             Backbones                                        #
################################################################################


class EfficientViT(BackboneBase):
    """EfficientViT model.

    EfficientViT is a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior
    high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel
    convolution, or complicated topology structure to obtain good performances, EfficientViT uses multi-scale linear
    attention achieves the global receptive field and multi-scale learning with only lightweight and hardware-efficient
    operations.

    References:
    - [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](
      https://arxiv.org/abs/2205.14756)
    - [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
    """

    def __init__(
        self,
        in_chans=3,
        width_list=(8, 16, 32, 64, 128),
        depth_list=(1, 2, 2, 2, 2),
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        num_classes=1000,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        export=False,
        **kwargs,
    ):
        """Initialize the EfficientViT model.

        Args:
            in_chans: Number of input image channels.
            width_list: Feature dimension at each stage.
            depth_list: Number of blocks at each stage.
            dim: Dimension of the head.
            expand_ratio: Expand ratio for the MBConv block.
            norm: Normalization layer type.
            act_func: Activation layer type.
            num_classes: Number of classes for classification head.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_chans,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, nn.Identity()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, nn.Identity() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        self.num_features = in_channels
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """Build local block.

        This method builds a single local block (DSConv, MBConv, FusedMBConv, or ResBlock)
        based on the provided parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the block.
            expand_ratio (float): Expansion ratio for the block.
            norm (str): Normalization layer type.
            act_func (str): Activation function type.
            fewer_norm (bool, optional): Whether to use fewer normalization layers. Defaults to False.

        Returns:
            nn.Module: Built block module.
        """
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def get_stage_dict(self):
        """Get the stage dictionary.

        This method returns a dictionary mapping stage indices to their modules.

        Returns:
            dict: Dictionary of stages.
        """
        stage_dict = {0: self.input_stem}
        for i, stage in enumerate(self.stages, start=1):
            stage_dict[i] = stage
        return stage_dict

    def get_classifier(self):
        """Get the classifier module.

        This method returns the classifier module (head) of the EfficientViT model.

        Returns:
            nn.Module: Classifier module.
        """
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the classifier head.

        This method resets the classifier head to a new number of classes.

        Args:
            num_classes (int): New number of classes.
            **kwargs: Additional arguments.
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head.

        This method performs the forward pass through the backbone up to the
        pre-logits stage, excluding the final classification head.
        It applies activation checkpointing if enabled.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after pre-logits stage.
        """
        x = self.input_stem(x)
        for stage in self.stages:
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
        return x.mean(dim=[2, 3])

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        x = self.input_stem(x)
        for stage in self.stages:
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
            outs.append(x)
        return outs

    def forward(self, x):
        """Forward pass of the EfficientViT model.

        This method performs the final forward pass of the EfficientViT model,
        including the pre-logits stage and the classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the EfficientViT model.
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


class EfficientViTLarge(BackboneBase):
    """EfficientViT Large model.

    EfficientViT is a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior
    high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel
    convolution, or complicated topology structure to obtain good performances, EfficientViT uses multi-scale linear
    attention achieves the global receptive field and multi-scale learning with only lightweight and hardware-efficient
    operations.

    References:
    - [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](
      https://arxiv.org/abs/2205.14756)
    - [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
    """

    def __init__(
        self,
        in_chans=3,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
        num_classes=1000,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        export=False,
        **kwargs,
    ):
        """Initialize the EfficientViTLarge class.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            width_list: Feature dimension at each stage.
            depth_list: Number of blocks at each stage.
            qkv_dim: Dimension of the head.
            norm: Normalization layer type.
            act_func: Activation layer type.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [ConvLayer(in_channels=in_chans, out_channels=width_list[0], stride=2, norm=norm, act_func=act_func)]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                stage_id=0,
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            stage0.append(ResidualBlock(block, nn.Identity()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:4], depth_list[1:4]), start=1):
            stage = []
            for i in range(d + 1):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    stage_id=stage_id,
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4 if stride == 1 else 16,
                    norm=norm,
                    act_func=act_func,
                    fewer_norm=stage_id > 2,
                )
                block = ResidualBlock(block, nn.Identity() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[4:], depth_list[4:]), start=4):
            stage = []
            block = self.build_local_block(
                stage_id=stage_id,
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=24,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=qkv_dim,
                        expand_ratio=6,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        self.num_features = in_channels
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def build_local_block(
        stage_id: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        """Build local block.

        This method builds a single local block (ResBlock, FusedMBConv, or MBConv)
        based on the provided parameters and stage ID.

        Args:
            stage_id (int): Stage ID.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the block.
            expand_ratio (float): Expansion ratio for the block.
            norm (str): Normalization layer type.
            act_func (str): Activation function type.
            fewer_norm (bool, optional): Whether to use fewer normalization layers. Defaults to False.

        Returns:
            nn.Module: Built block module.
        """
        if expand_ratio == 1:
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif stage_id <= 2:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def get_stage_dict(self):
        """Get the stage dictionary.

        This method returns a dictionary mapping stage indices to their modules.

        Returns:
            dict: Dictionary of stages.
        """
        stage_dict = {}
        for i, stage in enumerate(self.stages):
            stage_dict[i] = stage
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module.

        This method returns the classifier module (head) of the EfficientViTLarge model.

        Returns:
            nn.Module: Classifier module.
        """
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the classifier head.

        This method resets the classifier head to a new number of classes.

        Args:
            num_classes (int): New number of classes.
            **kwargs: Additional arguments.
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head.

        This method performs the forward pass through the backbone up to the
        pre-logits stage, excluding the final classification head.
        It applies activation checkpointing if enabled.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after pre-logits stage.
        """
        for stage in self.stages:
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        x = self.input_stem(x)
        for stage in self.stages:
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
            outs.append(x)
        return outs

    def forward(self, x):
        """Forward pass of the EfficientViTLarge model.

        This method performs the final forward pass of the EfficientViTLarge model,
        including the pre-logits stage and the classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the EfficientViTLarge model.
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def efficientvit_b0(**kwargs):
    """Constructs a EfficientViT-B0 model.

    EfficientViT-B0 is a lightweight EfficientViT model with a smaller number of parameters.
    It is designed for efficient inference on edge devices.

    Args:
        **kwargs: Additional arguments for EfficientViT-B0.

    Returns:
        EfficientViT: EfficientViT-B0 model.
    """
    return EfficientViT(width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b1(**kwargs):
    """Constructs a EfficientViT-B1 model.

    EfficientViT-B1 is a medium-sized EfficientViT model with a moderate number of parameters.
    It provides a good balance between accuracy and speed.

    Args:
        **kwargs: Additional arguments for EfficientViT-B1.

    Returns:
        EfficientViT: EfficientViT-B1 model.
    """
    return EfficientViT(width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b2(**kwargs):
    """Constructs a EfficientViT-B2 model.

    EfficientViT-B2 is a larger EfficientViT model with a larger number of parameters.
    It provides better accuracy but slightly slower inference.

    Args:
        **kwargs: Additional arguments for EfficientViT-B2.

    Returns:
        EfficientViT: EfficientViT-B2 model.
    """
    return EfficientViT(width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_b3(**kwargs):
    """Constructs a EfficientViT-B3 model.

    EfficientViT-B3 is the largest EfficientViT model with the highest number of parameters.
    It provides the best accuracy but the slowest inference.

    Args:
        **kwargs: Additional arguments for EfficientViT-B3.

    Returns:
        EfficientViT: EfficientViT-B3 model.
    """
    return EfficientViT(width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32, **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l0(**kwargs):
    """Constructs a EfficientViT-L0 model.

    EfficientViT-L0 is a lightweight EfficientViT model variant.
    It is designed for efficient inference on edge devices.

    Args:
        **kwargs: Additional arguments for EfficientViT-L0.

    Returns:
        EfficientViTLarge: EfficientViT-L0 model.
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 1, 1, 4, 4], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l1(**kwargs):
    """Constructs a EfficientViT-L1 model.

    EfficientViT-L1 is a medium-sized EfficientViT model variant.
    It provides a good balance between accuracy and speed.

    Args:
        **kwargs: Additional arguments for EfficientViT-L1.

    Returns:
        EfficientViTLarge: EfficientViT-L1 model.
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 1, 1, 6, 6], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l2(**kwargs):
    """Constructs a EfficientViT-L2 model.

    EfficientViT-L2 is a larger EfficientViT model variant.
    It provides better accuracy but slightly slower inference.

    Args:
        **kwargs: Additional arguments for EfficientViT-L2.

    Returns:
        EfficientViTLarge: EfficientViT-L2 model.
    """
    return EfficientViTLarge(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 2, 8, 8], **kwargs)


@BACKBONE_REGISTRY.register()
def efficientvit_l3(**kwargs):
    """Constructs a EfficientViT-L3 model.

    EfficientViT-L3 is the largest EfficientViT model variant.
    It provides the best accuracy but the slowest inference.

    Args:
        **kwargs: Additional arguments for EfficientViT-L3.

    Returns:
        EfficientViTLarge: EfficientViT-L3 model.
    """
    return EfficientViTLarge(width_list=[64, 128, 256, 512, 1024], depth_list=[1, 2, 2, 8, 8], **kwargs)
