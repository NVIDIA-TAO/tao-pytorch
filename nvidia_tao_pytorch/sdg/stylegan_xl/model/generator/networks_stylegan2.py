# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StyleGAN2's componenets utilized in StlyeGAN-XL. StyleGAN2 architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py
"""

import numpy as np
import torch

from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import conv2d_resample
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import upfirdn2d
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import bias_act


class FullyConnectedLayer(torch.nn.Module):
    """Fully Connected Layer."""

    def __init__(
        self,
        in_features,                    # Number of input features.
        out_features,                   # Number of output features.
        bias=True,                      # Apply additive bias before the activation function?
        activation='linear',            # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,                # Learning rate multiplier.
        bias_init=0,                    # Initial value for the additive bias.
    ):
        """Initializes the FullyConnectedLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Apply additive bias before the activation function? Defaults to True.
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. Defaults to 'linear'.
            lr_multiplier (float, optional): Learning rate multiplier. Defaults to 1.
            bias_init (float, optional): Initial value for the additive bias. Defaults to 0.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        """Forward pass through the FullyConnectedLayer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Output tensor after applying the fully connected layer.
        """
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        """Provides a string representation of the FullyConnectedLayer.

        Returns:
            str: A string representation of the FullyConnectedLayer instance, including
                various attributes such as dimensions and configurations.
        """
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class Conv2dLayer(torch.nn.Module):
    """2D Convolutional Layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation='linear',
        up=1,
        down=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        """Initializes the Conv2dLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Width and height of the convolution kernel.
            bias (bool, optional): Apply additive bias before the activation function? Defaults to True.
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. Defaults to 'linear'.
            up (int, optional): Integer upsampling factor. Defaults to 1.
            down (int, optional): Integer downsampling factor. Defaults to 1.
            resample_filter (list, optional): Low-pass filter to apply when resampling activations. Defaults to [1,3,3,1].
            conv_clamp (float, optional): Clamp the output to +-X, None = disable clamping. Defaults to None.
            channels_last (bool, optional): Expect the input to have memory_format=channels_last? Defaults to False.
            trainable (bool, optional): Update the weights of this layer during training? Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        """Forward pass through the Conv2dLayer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
            gain (float, optional): Gain factor for the activation function. Defaults to 1.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional layer.
        """
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        """Provides a string representation of the Conv2dLayer.

        Returns:
            str: A string representation of the Conv2dLayer instance, including
                various attributes such as dimensions and configurations.
        """
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])


class Conv2dLayerDepthwise(torch.nn.Module):
    """Depthwise 2D Convolutional Layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation='linear',
        up=1,
        down=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        """Initializes the Conv2dLayerDepthwise.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Width and height of the convolution kernel.
            bias (bool, optional): Apply additive bias before the activation function? Defaults to True.
            activation (str, optional): Activation function: 'relu', 'lrelu', etc. Defaults to 'linear'.
            up (int, optional): Integer upsampling factor. Defaults to 1.
            down (int, optional): Integer downsampling factor. Defaults to 1.
            resample_filter (list, optional): Low-pass filter to apply when resampling activations. Defaults to [1,3,3,1].
            conv_clamp (float, optional): Clamp the output to +-X, None = disable clamping. Defaults to None.
            channels_last (bool, optional): Expect the input to have memory_format=channels_last? Defaults to False.
            trainable (bool, optional): Update the weights of this layer during training? Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([in_channels, 1, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([in_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        """Forward pass through the Conv2dLayerDepthwise.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
            gain (float, optional): Gain factor for the activation function. Defaults to 1.

        Returns:
            torch.Tensor: Output tensor after applying the depthwise convolutional layer.
        """
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight, groups=self.in_channels)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        """Provides a string representation of the Conv2dLayerDepthwise.

        Returns:
            str: A string representation of the Conv2dLayerDepthwise instance, including
                various attributes such as dimensions and configurations.
        """
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])
