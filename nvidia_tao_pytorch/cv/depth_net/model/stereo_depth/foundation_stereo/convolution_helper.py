# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Convolution helper files for Stereo"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _is_contiguous(tensor: torch.Tensor) -> bool:
    """ Checks whether tensor is contiguous
    """
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


def check_valid_options(item: str, instance_dictionary: dict):
    """Function to check if an item is in the dictionary.


    Args:
        item (str): key to search for.
        instance_dictionary (dict): Dictionary to search for.
    """
    if item not in instance_dictionary.keys():
        raise NotImplementedError(f"Invalid option {item} requested. Choose among {instance_dictionary.keys()}")
    return instance_dictionary[item]


def get_norm_type(norm_type):
    """Returns the normalization layer class from a string key.

    This function acts as a simple factory to get a PyTorch normalization
    layer class (e.g., BatchNorm2d, InstanceNorm3d) based on a string
    identifier. It is useful for dynamically selecting normalization layers
    in neural network architectures.

    Args:
        norm_type (str): The keyword for the desired normalization type.
            Valid options are:
            - 'instance3d': Returns `torch.nn.InstanceNorm3d`.
            - 'batch3d': Returns `torch.nn.BatchNorm3d`.
            - 'instance2d': Returns `torch.nn.InstanceNorm2d`.
            - 'batch2d': Returns `torch.nn.BatchNorm2d`.

    Returns:
        torch.nn.Module: The normalization layer class (not an instance).

    Raises:
        NotImplementedError: If `norm_type` is not one of the valid keys.

    """
    norm = {'instance3d': nn.InstanceNorm3d, 'batch3d': nn.BatchNorm3d,
            'instance2d': nn.InstanceNorm2d, 'batch2d': nn.BatchNorm2d}

    check_valid_options(norm_type, norm)

    return norm[norm_type]


def get_conv_type(conv_type):
    """Returns a convolution or deconvolution layer class based on a string key.

    This function serves as a factory to retrieve a PyTorch convolution
    layer class (e.g., `Conv2d`, `Conv3d`, `ConvTranspose2d`) from a
    given string identifier. This is a common pattern for creating modular
    neural network architectures where the type of layer can be easily
    switched.

    Args:
        conv_type (str): The keyword for the desired convolution type.
            Valid options are:
            - 'conv2d': Returns `torch.nn.Conv2d`.
            - 'conv3d': Returns `torch.nn.Conv3d`.
            - 'deconv2d': Returns `torch.nn.ConvTranspose2d` (2D deconvolution).
            - 'deconv3d': Returns `torch.nn.ConvTranspose3d` (3D deconvolution).

    Returns:
        torch.nn.Module: The convolution or deconvolution layer class (not an instance).

    Raises:
        KeyError: If `conv_type` is not one of the valid keys.
    """
    conv = {'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d,
            'deconv2d': nn.ConvTranspose2d, 'deconv3d': nn.ConvTranspose3d}

    check_valid_options(conv_type, conv)

    return conv[conv_type]


class ResnetBasicBlock(nn.Module):
    """
    Implements a Basic Resnet Block as described in the paper "Deep Residual Learning for Image Recognition."

    A Basic ResNet block consists of two convolutional layers with a ReLU activation function
    after the first convolution and after the final element-wise addition.
    It includes a residual connection (or skip connection) that adds the input of the block
    to the output of the convolutional layers. This helps mitigate the vanishing gradient
    problem in deep neural networks.

    Args:
        inplanes (int): The number of input channels for the first convolutional layer.
        planes (int): The number of output channels for both convolutional layers.
                      This is also the number of input channels for the second convolution.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
        stride (int, optional): The stride of the first convolutional layer. The second
                                convolutional layer has a stride of 1. Defaults to 1.
        padding (int, optional): The amount of zero-padding to add to both sides of the
                                 input. Defaults to 1.
        downsample (nn.Module, optional): A downsampling module (e.g., a 1x1 convolution)
                                          to match the dimensions of the input and output
                                          when the stride is greater than 1. Defaults to None.
        groups (int, optional): The number of blocked connections from input channels to
                                output channels. Not supported for the BasicBlock. Defaults to 1.
        base_width (int, optional): The base number of channels in each convolutional group.
                                    Not supported for the BasicBlock. Defaults to 64.
        dilation (int, optional): The spacing between kernel elements. Not supported for
                                  the BasicBlock. Defaults to 1.
        norm_layer (nn.Module, optional): The normalization layer to use (e.g., nn.BatchNorm2d).
                                          If None, no normalization is applied. Defaults to nn.BatchNorm2d.
        bias (bool, optional): Whether to include a learnable bias in the convolutional layers.
                               Defaults to False.

    Raises:
        ValueError: If `groups` is not 1 or `base_width` is not 64.
        NotImplementedError: If `dilation` is greater than 1.
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1,
                 padding=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                               stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Defines the computation performed at every call.

        The forward pass of the BasicBlock computes a residual connection.
        The input `x` is passed through a sequence of two convolutional layers,
        batch normalization (if enabled), and a ReLU activation. The original
        input `x` is then added to the output of this sequence.
        If a `downsample` module is provided (e.g., when the stride is > 1),
        it is applied to the original input `x` to match its dimensions
        with the output of the convolutional layers before the element-wise addition.

        Args:
            x (torch.Tensor): The input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the block, after the residual
                          connection and final ReLU activation.
        """
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResnetBasicBlock3D(nn.Module):
    """A 3D residual block module based on the ResNet architecture.

    This class implements a standard residual block for 3D convolutional
    neural networks. It consists of two 3D convolutional layers with an
    activation function (ReLU) and an optional normalization layer
    (e.g., BatchNorm3d) applied after each convolution. The core feature
    is the 'residual connection' where the input tensor is added to the
    output of the two convolutional layers, helping to mitigate the
    vanishing gradient problem in deep networks.

    This implementation is designed for simplicity and currently only
    supports a specific configuration (groups=1, base_width=64,
    dilation=1), but it serves as a foundational building block for
    larger 3D ResNet models.

    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
        stride (int, optional): The stride of the first convolution. Defaults to 1.
        padding (int, optional): The padding for the convolutions. Defaults to 1.
        downsample (nn.Module, optional): A downsampling module to match
            dimensions for the residual connection. This is typically a
            convolutional layer or a pooling layer used when the stride
            is greater than 1 or `inplanes` does not equal `planes`.
            Defaults to None.
        groups (int, optional): The number of blocked connections from
            input channels to output channels. Currently, only `groups=1`
            is supported. Defaults to 1.
        base_width (int, optional): The number of channels per group.
            Currently, only `base_width=64` is supported. Defaults to 64.
        dilation (int, optional): The dilation rate for the convolutions.
            Currently, only `dilation=1` is supported. Defaults to 1.
        norm_layer (nn.Module, optional): The normalization layer class to use,
            e.g., `nn.BatchNorm3d` or `nn.InstanceNorm3d`. Defaults to
            `nn.BatchNorm3d`.
        bias (bool, optional): If `True`, adds a learnable bias to the
            convolutions. Defaults to `False`.

    Raises:
        ValueError: If `groups` is not 1 or `base_width` is not 64.
        NotImplementedError: If `dilation` is greater than 1.

    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):

        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Performs the forward pass of the ResnetBasicBlock3D.

        The forward pass computes the residual block's output. It applies the
        first convolution, followed by an optional normalization and a ReLU.
        The output is then passed through the second convolution and another
        optional normalization. Finally, the residual connection is performed
        by adding the identity mapping (potentially downsampled) to the
        output, and a final ReLU activation is applied.

        Args:
            x (torch.Tensor): The input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the block.
        """
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Conv(nn.Module):
    """A general-purpose convolutional module for PyTorch.

    This class provides a flexible, single-block convolutional layer that
    can be configured at initialization. It supports various types of
    convolutions (e.g., 2D, 3D, and their transpose variants), optional
    normalization, and an optional ReLU activation function. The internal
    layer types are determined by string keys passed to the constructor,
    leveraging helper functions like `get_conv_type` and `get_norm_type`.

    The module's forward pass applies the operations in the standard order:
    convolution, followed by normalization (if specified), and finally a
    Leaky ReLU activation (if `relu` is True).

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        relu (bool): If True, a Leaky ReLU activation is applied after
            the normalization layer.
        norm_type (Optional[str]): The type of normalization layer to use.
            Should be a key recognized by `get_norm_type` (e.g.,
            'batch2d', 'instance3d'). If None, no normalization is
            applied.
        conv_type (str): The type of convolution layer to use. Must be a
            key recognized by `get_conv_type` (e.g., 'conv2d', 'deconv3d').
        **kwargs: Additional keyword arguments to be passed directly to the
            convolutional layer's constructor (e.g., `kernel_size`, `stride`,
            `padding`).

    Attributes:
        conv (nn.Module): The convolutional layer.
        norm (Optional[nn.Module]): The normalization layer, or None if not used.
        relu (bool): A flag indicating whether to apply ReLU.

    Example:
        >>> # Example of a 2D convolutional block with BatchNorm and LeakyReLU
        >>> conv_block_2d = Conv(in_channels=3, out_channels=64, relu=True,
        ...                      norm_type='batch2d', conv_type='conv2d',
        ...                      kernel_size=3, stride=1, padding=1)
        >>> print(conv_block_2d)
        Conv(
          (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    """

    def __init__(self, in_channels, out_channels, relu, norm_type, conv_type, **kwargs):
        super(Conv, self).__init__()
        self.relu = relu
        self.norm = None
        self.conv = get_conv_type(conv_type)(in_channels, out_channels, bias=False, **kwargs)
        if norm_type is not None:
            self.norm = get_norm_type(norm_type)(out_channels)
            # External ckpts may store this norm under ``bn`` or ``IN``; remap at load.
            self._register_load_state_dict_pre_hook(self._remap_norm_keys)

    def _remap_norm_keys(self, state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
        """Rewrite ``<prefix>bn.*`` and ``<prefix>IN.*`` → ``<prefix>norm.*``."""
        for alias in ('bn', 'IN'):
            old_pre = prefix + alias + '.'
            new_pre = prefix + 'norm.'
            for k in list(state_dict.keys()):
                if k.startswith(old_pre):
                    state_dict[new_pre + k[len(old_pre):]] = state_dict.pop(k)

    def forward(self, x):
        """Performs the forward pass of the Conv module.

        The input tensor `x` is passed sequentially through the convolution,
        an optional normalization layer, and an optional Leaky ReLU activation.

        Args:
            x (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output tensor after applying the operations.
        """
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class Conv2xDownScale(nn.Module):
    """A module for downscaling feature maps by a factor of 2, often used in U-Net architectures.

    This class provides a flexible building block for decoder paths in networks like U-Net. It performs a
    convolutional operation to downsample the input tensor by a factor of 2. It then combines this downscaled
    tensor with a 'remaining' tensor (typically a skip connection from the encoder path) via concatenation or
    addition, and finally passes the combined tensor through a second convolutional block.

    Args:
        in_channels (int): Number of input channels for the first convolution.
        out_channels (int): Number of output channels for both convolutions.
        norm_type (str): Type of normalization layer to use (e.g., 'batch2d').
        conv_type (str): Type of convolutional layer to use (e.g., 'conv2d').
        concat (bool, optional): If `True`, combines the downscaled tensor and the 'remaining' tensor
            by concatenation. If `False`, they are combined via addition. Defaults to `True`.
        keep_concat (bool, optional): Relevant only if `concat` is `True`. If `True`, the number of
            output channels for the second convolution is double `out_channels`. This is
            typically used to maintain a larger number of features after concatenation. Defaults to `True`.
        relu (bool, optional): If `True`, a Leaky ReLU activation is applied after the first
            convolution. Defaults to `True`.
        keep_dispc (bool, optional): Special handling for 3D deconvolution with a specific
            kernel and stride for a particular application (e.g., depth estimation). This
            is only relevant when `conv_type` is 'deconv3d'. Defaults to `False`.
        use_resnet_layer (bool, optional): If `True`, the second convolutional block is
            a `ResnetBasicBlock`. If `False`, it's a standard `Conv` block. Defaults to `False`.
        conv2_type (str, optional): Convolution kind for the second block. ``'deconv'``
            (default) reuses ``conv_type`` so conv2 mirrors conv1. ``'conv'`` maps
            ``deconv2d``/``deconv3d`` to ``conv2d``/``conv3d`` for conv2 only.
        rem_channels (Optional[int], optional): Channel count of the ``rem``
            tensor passed to ``forward(x, rem)``. None (default) keeps the
            legacy assumption ``rem.channels == out_channels`` and ``conv2``
            input becomes ``out_channels * 2``. Set explicitly when ``rem``
            has a different channel count than the conv1 output.
        conv2_out_channels (Optional[int], optional): Output channel count
            for ``conv2``. None (default) keeps the legacy formula
            ``out_channels * mul`` (mul = 2 if keep_concat else 1). Set when
            the second block needs a non-formulaic output width.
        **kwargs: Additional keyword arguments to be passed to the first `Conv` block's
            constructor (e.g., `padding`, `kernel_size`).

    Attributes:
        conv1 (Conv): The first convolutional block, responsible for downscaling.
        conv2 (nn.Module): The second convolutional block, which processes the combined tensor.
        concat (bool): Stores the concatenation flag.
    """

    def __init__(self, in_channels, out_channels, norm_type, conv_type, concat=True,
                 keep_concat=True, relu=True, keep_dispc=False, use_resnet_layer=False,
                 conv2_type: str = 'deconv',
                 rem_channels: Optional[int] = None,
                 conv2_out_channels: Optional[int] = None,
                 conv2_relu: bool = False,
                 **kwargs):

        super(Conv2xDownScale, self).__init__()
        self.concat = concat
        norm_type = norm_type
        conv_type = conv_type
        if conv_type == 'deconv3d':
            kernel = (4, 4, 4)
            if keep_dispc:
                kernel = (1, 4, 4)
                stride = (1, 2, 2)
                padding = (0, 1, 1)
                self.conv1 = Conv(in_channels, out_channels, relu,
                                  norm_type, conv_type, kernel_size=kernel, stride=stride, padding=padding, **kwargs)
        else:
            if conv_type == 'deconv2d':
                kernel = 4
            else:
                kernel = 3
            self.conv1 = Conv(in_channels, out_channels, relu, norm_type,
                              conv_type, kernel_size=kernel, stride=2, padding=1, **kwargs)

        # Explicit pairing; safe if conv-type registry grows.
        _conv2_pair = {
            'conv2d': 'conv2d',
            'conv3d': 'conv3d',
            'deconv2d': 'conv2d',
            'deconv3d': 'conv3d',
        }
        if conv2_type == 'deconv':
            conv2_full_type = conv_type
        elif conv2_type == 'conv':
            if conv_type not in _conv2_pair:
                raise ValueError(
                    f"conv_type {conv_type!r} not in supported pairing {sorted(_conv2_pair)}")
            conv2_full_type = _conv2_pair[conv_type]
        else:
            raise ValueError(f"conv2_type must be 'conv' or 'deconv', got {conv2_type!r}")

        if self.concat:
            mul = 2 if keep_concat else 1
            conv2_in = (out_channels + rem_channels
                        if rem_channels is not None
                        else out_channels * 2)
            conv2_out = (conv2_out_channels
                         if conv2_out_channels is not None
                         else out_channels * mul)
            if use_resnet_layer:
                self.conv2 = ResnetBasicBlock(
                    conv2_in, conv2_out, kernel_size=3,
                    stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
            else:
                self.conv2 = Conv(
                    conv2_in, conv2_out, relu=conv2_relu, norm_type=norm_type, conv_type=conv2_full_type,
                    kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, relu=conv2_relu, norm_type=norm_type,
                              conv_type=conv2_full_type, kernel_size=3, stride=1, padding=1, **kwargs)

    def forward(self, x, rem):
        """Performs the forward pass of the Conv2xDownScale module.

        The forward pass consists of a sequence of operations:
        1.  The input tensor `x` is passed through the first convolution (`self.conv1`) to
            perform downscaling.
        2.  The resulting tensor's spatial dimensions are resized to match the dimensions of the
            'remaining' tensor `rem` using bilinear interpolation if they do not match.
        3.  The downscaled tensor and `rem` are combined, either by concatenation along the
            channel dimension or by element-wise addition, based on the `self.concat` flag.
        4.  The combined tensor is passed through the second convolutional block (`self.conv2`)
            for further processing.

        Args:
            x (torch.Tensor): The main input tensor from the previous layer.
            rem (torch.Tensor): The 'remaining' tensor, typically a skip connection from a
                corresponding encoder layer.

        Returns:
            torch.Tensor: The final output tensor of the module.
        """
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class Conv3dNormActReduced(nn.Module):
    """A 3D convolutional module with reduced spatial and depth kernels, followed by normalization and activation.

    This module is designed to perform 3D convolutions more efficiently by splitting the operation into two
    sequential steps: first, a 2D convolution that operates on the height and width dimensions independently for
    each depth slice, and second, a 1D convolution that operates only along the depth dimension. This factorization
    can reduce the number of parameters and computation compared to a single, full 3D convolution, making it
    suitable for processing volumetric data. The module applies normalization and a ReLU activation after each
    convolutional layer.

    Args:
        C_in (int): The number of input channels.
        C_out (int): The number of output channels.
        hidden (Optional[int], optional): The number of channels in the intermediate hidden layer. If `None`, it defaults to `C_out`.
        kernel_size (int, optional): The kernel size for the spatial (height and width) dimensions of the first convolution. Defaults to 3.
        kernel_disp (Optional[int], optional): The kernel size for the depth dimension of the second convolution. If `None`, it defaults to `kernel_size`.
        stride (int, optional): The stride for the convolution in the spatial dimensions. Defaults to 1.
        norm (nn.Module, optional): The normalization layer class to use (e.g., `nn.BatchNorm3d`). Defaults to `nn.BatchNorm3d`.

    Attributes:
        conv1 (nn.Sequential): The first block, consisting of a 3D convolution with a (1, H, W) kernel,
            followed by normalization and ReLU.
        conv2 (nn.Sequential): The second block, consisting of a 3D convolution with a (D, 1, 1) kernel,
            followed by normalization and ReLU.
    """

    def __init__(self, C_in, C_out, hidden: Optional[int] = None,
                 kernel_size: int = 3, kernel_disp: Optional[int] = None,
                 stride: int = 1, norm: nn.Module = nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
            kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1, kernel_size, kernel_size),
                      padding=(0, kernel_size // 2, kernel_size // 2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1),
                      padding=(kernel_disp // 2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the Conv3dNormActReduced module.

        The input tensor is passed through the `conv1` block, which applies a 2D-like convolution
        across the spatial dimensions. The output of `conv1` is then fed into the `conv2` block, which
        applies a 1D-like convolution across the depth dimension. The final result is the output of `conv2`.

        Args:
            x (torch.Tensor): The input tensor with a shape of (B, C, D, H, W), where:
                              - B is the batch size.
                              - C is the number of channels.
                              - D is the depth.
                              - H is the height.
                              - W is the width.

        Returns:
            torch.Tensor: The output tensor after the two-step convolution, normalization, and activation.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x
