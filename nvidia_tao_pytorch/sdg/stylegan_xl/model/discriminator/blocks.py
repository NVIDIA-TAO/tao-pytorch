# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" StyleGAN-XL discriminator blocks. """

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from nvidia_tao_pytorch.sdg.stylegan_xl.model.generator.networks_stylegan2 import Conv2dLayer, Conv2dLayerDepthwise


# single layers

def conv2d(*args, **kwargs):
    """Create a 2D convolutional layer with spectral normalization.

    Args:
        *args: Positional arguments passed to nn.Conv2d.
        **kwargs: Keyword arguments passed to nn.Conv2d.

    Returns:
        nn.Conv2d: A 2D convolutional layer with spectral normalization applied.
    """
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def conv_transpose2d(*args, **kwargs):
    """Create a 2D transposed convolutional layer with spectral normalization.

    Args:
        *args: Positional arguments passed to nn.ConvTranspose2d.
        **kwargs: Keyword arguments passed to nn.ConvTranspose2d.

    Returns:
        nn.ConvTranspose2d: A 2D transposed convolutional layer with spectral normalization applied.
    """
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def embedding(*args, **kwargs):
    """Create an embedding layer with spectral normalization.

    Args:
        *args: Positional arguments passed to nn.Embedding.
        **kwargs: Keyword arguments passed to nn.Embedding.

    Returns:
        nn.Embedding: An embedding layer with spectral normalization applied.
    """
    return spectral_norm(nn.Embedding(*args, **kwargs))


def linear(*args, **kwargs):
    """Create a linear (fully connected) layer with spectral normalization.

    Args:
        *args: Positional arguments passed to nn.Linear.
        **kwargs: Keyword arguments passed to nn.Linear.

    Returns:
        nn.Linear: A linear layer with spectral normalization applied.
    """
    return spectral_norm(nn.Linear(*args, **kwargs))


def norm_layer(c, mode='batch'):
    """Create a normalization layer based on the specified mode.

    Args:
        c (int): Number of channels.
        mode (str): Normalization mode. Can be 'batch' for BatchNorm or 'group' for GroupNorm.

    Returns:
        nn.Module: A normalization layer (nn.BatchNorm2d or nn.GroupNorm) based on the mode.
    """
    if mode == 'group':
        return nn.GroupNorm(c // 2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


# Activations
class GLU(nn.Module):
    """Gated Linear Unit (GLU) Activation Function."""

    def forward(self, x):
        """Forward function.

        Args:
            feat (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Swish function.
        """
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):
    """Swish Activation Function."""

    def forward(self, feat):
        """Forward function.

        Args:
            feat (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Swish function.
        """
        return feat * torch.sigmoid(feat)


# Upblocks

class InitLayer(nn.Module):
    """Initialization Layer for the Generator."""

    def __init__(self, nz, channel, sz=4):
        """Initializes the InitLayer.

        Args:
            nz (int): Size of the input noise vector.
            channel (int): Number of output channels.
            sz (int, optional): Spatial size of the output tensor. Default is 4.
        """
        super().__init__()

        self.init = nn.Sequential(
            conv_transpose2d(nz, channel * 2, sz, 1, 0, bias=False),
            norm_layer(channel * 2),
            GLU(),
        )

    def forward(self, noise):
        """Forward function.

        Args:
            noise (torch.Tensor): Input noise tensor with shape (batch_size, nz).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, channel, sz, sz).
        """
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


class UpBlockSmallCond(nn.Module):
    """Conditional Small Up-sampling Block."""

    def __init__(self, in_planes, out_planes, z_dim):
        """Initializes the UpBlockSmallCond.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            z_dim (int): Dimensionality of the conditional vector.
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn = which_bn(2 * out_planes)
        self.act = GLU()

    def forward(self, x, c):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map.
            c (torch.Tensor): Conditional vector.

        Returns:
            torch.Tensor: Output feature map after processing.
        """
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c)
        x = self.act(x)
        return x


class UpBlockBigCond(nn.Module):
    """Conditional Big Up-sampling Block."""

    def __init__(self, in_planes, out_planes, z_dim):
        """Initializes the UpBlockBigCond.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            z_dim (int): Dimensionality of the conditional vector.
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2 * out_planes)
        self.bn2 = which_bn(2 * out_planes)
        self.act = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map.
            c (torch.Tensor): Conditional vector.

        Returns:
            torch.Tensor: Output feature map after processing.
        """
        # block 1
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)
        x = self.act(x)

        # block 2
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)
        x = self.act(x)

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block."""

    def __init__(self, ch_in, ch_out):
        """Initializes the SEBlock.

        Args:
            ch_in (int): Number of input channels for the small feature map.
            ch_out (int): Number of output channels.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        """Forward function.

        Args:
            feat_small (torch.Tensor): Small feature map.
            feat_big (torch.Tensor): Large feature map.

        Returns:
            torch.Tensor: Scaled large feature map.
        """
        return feat_big * self.main(feat_small)


# Downblocks

class DownBlock(nn.Module):
    """Basic Down-sampling Block."""

    def __init__(self, in_planes, out_planes, width=1):
        """Initializes the DownBlock.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            width (int, optional): Multiplier for the number of output channels. Default is 1.
        """
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes * width, 4, 2, 1, bias=True),
            norm_layer(out_planes * width),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        """Forward function.

        Args:
            feat (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Down-sampled feature map.
        """
        return self.main(feat)


class DownBlockSGBlocks(nn.Module):
    """Down-sampling Block with Separable Grouped Convolutions."""

    def __init__(self, in_channels, out_channels):
        """Initializes the DownBlockSGBlocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        conv_depthwise = Conv2dLayerDepthwise(in_channels, in_channels, kernel_size=3, activation='linear')
        conv_pointwise = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation='lrelu', down=2)
        self.main = nn.Sequential(conv_depthwise, conv_pointwise)

    def forward(self, feat):
        """Forward function.

        Args:
            feat (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Processed feature map.
        """
        return self.main(feat)


class SeparableConv2d(nn.Module):
    """Separable Convolutional Layer."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        """Initializes the SeparableConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            bias (bool, optional): Whether to use a bias term. Default is False.
        """
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels,
                                kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after separable convolution.
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DownBlockSep(nn.Module):
    """Down-sampling Block with Separable Convolution."""

    def __init__(self, in_planes, out_planes):
        """Initializes the DownBlockSep.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
        """
        super().__init__()
        self.main = nn.Sequential(
            SeparableConv2d(in_planes, out_planes, 3),
            norm_layer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, feat):
        """Forward function.

        Args:
            feat (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Down-sampled feature map.
        """
        return self.main(feat)


class DownBlockPatch(nn.Module):
    """Patch-based Down-sampling Block."""

    def __init__(self, in_planes, out_planes):
        """Initializes the DownBlockPatch.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
        """
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            norm_layer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        """Forward function.

        Args:
            feat (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Processed feature map.
        """
        return self.main(feat)


# CSM

class ResidualConvUnit(nn.Module):
    """Residual Convolutional Unit (RCU) for feature processing."""

    def __init__(self, cin, activation, bn):
        """Initializes the ResidualConvUnit.

        Args:
            cin (int): Number of input channels.
            activation (nn.Module): Activation function.
            bn (bool): Whether to apply batch normalization.
        """
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the residual connection.
        """
        return self.skip_add.add(self.conv(x), x)


class FeatureFusionBlock(nn.Module):
    """Feature Fusion Block for combining multi-scale features."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        """Initializes the FeatureFusionBlock.

        Args:
            features (int): Number of input channels.
            activation (nn.Module): Activation function.
            deconv (bool, optional): Whether to use deconvolution. Default is False.
            bn (bool, optional): Whether to apply batch normalization. Default is False.
            expand (bool, optional): Whether to expand the number of channels. Default is False.
            align_corners (bool, optional): Align corners when upsampling. Default is True.
            lowest (bool, optional): Whether this is the lowest block in the hierarchy. Default is False.
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward function.

        Args:
            xs (tuple of Tensor): Input tensors to be fused.

        Returns:
            torch.Tensor: Fused and upsampled output tensor.
        """
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


# Misc

class NoiseInjection(nn.Module):
    """Noise Injection Layer for adding random noise to the feature map."""

    def __init__(self):
        """Initializes the NoiseInjection layer.

        Creates a learnable weight parameter for controlling the amount of noise.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        """Forward function.

        Args:
            feat (torch.Tensor): Input feature map.
            noise (Tensor, optional): Noise to be added. If None, random noise is generated. Default is None.

        Returns:
            torch.Tensor: Feature map with added noise.
        """
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class CCBN(nn.Module):
    """Conditional Batchnorm."""

    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        """Initializes the CCBN layer.

        Args:
            output_size (int): Number of output channels.
            input_size (int): Dimensionality of the conditional input.
            which_linear (callable): Function to create linear layers for gain and bias.
            eps (float, optional): Small epsilon value to avoid division by zero. Default is 1e-5.
            momentum (float, optional): Momentum for the running mean and variance. Default is 0.1.
        """
        super().__init__()
        self.output_size, self.input_size = output_size, input_size

        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor to be normalized.
            y (torch.Tensor): Conditional input for calculating gains and biases.

        Returns:
            torch.Tensor: Normalized and modulated output tensor.
        """
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias


class CCBN1D(nn.Module):
    """Conditional Batchnorm."""

    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        """Initializes the CCBN1D layer.

        Args:
            output_size (int): Number of output channels.
            input_size (int): Dimensionality of the conditional input.
            which_linear (callable): Function to create linear layers for gain and bias.
            eps (float, optional): Small epsilon value to avoid division by zero. Default is 1e-5.
            momentum (float, optional): Momentum for the running mean and variance. Default is 0.1.
        """
        super().__init__()
        self.output_size, self.input_size = output_size, input_size

        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor to be normalized.
            y (torch.Tensor): Conditional input for calculating gains and biases.

        Returns:
            torch.Tensor: Normalized and modulated output tensor.
        """
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1)
        bias = self.bias(y).view(y.size(0), -1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size, mode='bilinear', align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): input

        Returns:
            tensor: interpolated data
        """
        x = self.interp(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
