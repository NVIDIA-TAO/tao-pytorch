# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model components for distilling TAO Toolkit models."""

from typing import List, Type, Union

import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.models import Backbone


class MLP(nn.Module):
    """MLP block definition

    Creates following structure
    if depth > 1
        conv1 (in_ch, hidden_ch) -> conv(hidden_ch, hidden_ch) x depth -> conv2(hidden_ch, out_ch) -> act + (if skip x)
    else
        conv1 (in_ch, hidden_ch) -> conv2(hidden_ch, out_ch) -> act + (if skip x )
    """

    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 act: Type[nn.Module] = nn.GELU,
                 kernel_size: int = 1,
                 depth: int = 1,
                 skip_connect: bool = False
                 ):
        """Initializes the MLP block.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            act (Type[nn.Module], optional): Activation function. Defaults to nn.GELU.
            kernel_size (int, optional): Kernel size. Defaults to 1.
            depth (int, optional): Depth of the MLP block. Defaults to 1.
            skip_connect (bool, optional): Whether to use skip connection. Defaults to False.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        if depth > 1:

            hidden_layers = []

            for _ in range(depth - 1):
                layer = nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
                hidden_layers.append(layer)

            self.hidden_layers = nn.ModuleList(hidden_layers)
        else:
            self.hidden_layers = None

        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        self.act = act()
        self.skip_connect = skip_connect

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.act(x)

        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                if self.skip_connect:
                    x = x + self.act(layer(x))
                else:
                    x = self.act(layer(x))

        x = self.conv2(x)
        return x


class Neck(nn.Module):
    """Base neck module"""

    def in_channels(self) -> List[int]:
        """Returns the input channels of the neck"""
        raise NotImplementedError("in_channels must be implemented")

    def out_channels(self) -> List[int]:
        """Returns the output channels of the neck"""
        raise NotImplementedError("out_channels must be implemented")

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of the neck"""
        raise NotImplementedError("forward must be implemented")


class LayerListNeck(Neck):
    """LayerListNeck class to use a list of layers as neck"""

    def __init__(self, in_channels: List[int],
                 out_channels: List[int],
                 layers: List[nn.Module]
                 ):
        """Initializes the LayerListNeck

        Args:
            in_channels (List[int]): List of input channels.
            out_channels (List[int]): List of output channels.
            layers (List[nn.Module]): List of layers.
        """
        super().__init__()

        if len(in_channels) != len(out_channels):
            raise RuntimeError("Length of input channels must equal length of output channels.")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def in_channels(self) -> List[int]:
        """Returns the input channels of the neck"""
        return self._in_channels

    def out_channels(self) -> List[int]:
        """Returns the output channels of the neck"""
        return self._out_channels

    def forward(self, features: List[torch.Tensor]):
        """Forward pass of the neck

        Args:
            features (List[torch.Tensor]): List of input feature maps.

        Returns:
            List[torch.Tensor]: List of output feature maps.
        """
        if len(features) != len(self.layers):
            raise RuntimeError("Number of feature maps must match number of projection layers.")

        features = [self.layers[i](features[i]) for i in range(len(features))]
        return features


# Modifying LinearNeck definition since Pytorch lightning complains cause of hierarchy
# 'LinearNeck' object has no attribute '_modules'
class LinearNeck(nn.Module):
    """LinearNeck class to use linear projection as neck"""

    def __init__(self, in_channels: List[int],
                 out_channels: List[int],
                 kernel_size: int = 1
                 ):
        """Initializes the LinearNeck

        Args:
            in_channels (List[int]): List of input channels.
            out_channels (List[int]): List of output channels.
            kernel_size (int, optional): Kernel size. Defaults to 1.
        """
        super().__init__()

        if len(in_channels) != len(out_channels):
            raise RuntimeError("Length of input channels must equal length of output channels.")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        self.layers = nn.ModuleList(self.layers)

    def in_channels(self) -> List[int]:
        """Returns the input channels of the neck"""
        return self._in_channels

    def out_channels(self) -> List[int]:
        """Returns the output channels of the neck"""
        return self._out_channels

    def forward(self, features: List[torch.Tensor]):
        """Forward pass of the neck

        Args:
            features (List[torch.Tensor]): List of input feature maps.

        Returns:
            List[torch.Tensor]: List of output feature maps.
        """
        if len(features) != len(self.layers):
            raise RuntimeError("Number of feature maps must match number of projection layers.")

        features = [self.layers[i](features[i]) for i in range(len(features))]
        return features


# Modifying MlpNeck definition since Pytorch lightning complains cause of hierarchy
class MLPNeck(nn.Module):
    """MlpNeck class to use MLP projection as neck"""

    def __init__(self, in_channels: List[int],
                 out_channels: List[int],
                 expansion: Union[float, int] = 1,
                 kernel_size: int = 1,
                 act: Type[nn.Module] = nn.GELU,
                 depth: int = 1,
                 skip_connect: bool = False
                 ):
        """Initializes the MlpNeck

        Args:
            in_channels (List[int]): List of input channels.
            out_channels (List[int]): List of output channels.
            expansion (Union[float, int], optional): Expansion factor. Defaults to 1.
            kernel_size (int, optional): Kernel size. Defaults to 1.
            act (Type[nn.Module], optional): Activation function. Defaults to nn.GELU.
            depth (int, optional): Depth of the MLP block. Defaults to 1.
            skip_connect (bool, optional): Whether to use skip connection. Defaults to False.
        """
        super().__init__()

        if len(in_channels) != len(out_channels):
            raise RuntimeError("Length of input channels must equal length of output channels.")
        layers = []
        for in_ch, out_ch in zip(in_channels, out_channels):
            layer = MLP(
                in_channels=in_ch,
                hidden_channels=int(in_ch * expansion),
                out_channels=out_ch,
                act=act,
                kernel_size=kernel_size,
                depth=depth,
                skip_connect=skip_connect
            )
            layers.append(layer)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def in_channels(self) -> List[int]:
        """Returns the input channels of the neck"""
        return self._in_channels

    def out_channels(self) -> List[int]:
        """Returns the output channels of the neck"""
        return self._out_channels

    def forward(self, features: List[torch.Tensor]):
        """Forward pass of the neck

        Args:
            features (List[torch.Tensor]): List of input feature maps.

        Returns:
            List[torch.Tensor]: List of output feature maps.
        """
        if len(features) != len(self.layers):
            raise RuntimeError("Number of feature maps must match number of projection layers.")

        features = [self.layers[i](features[i]) for i in range(len(features))]

        return features


class BackboneNeck(Backbone):
    """Backbone fitted with neck

    The neck acts as a transition layer to match the channels of the student
    backbone to the teacher.
    """

    def __init__(self, backbone: Backbone,
                 neck: Neck,
                 freeze_backbone: bool = False
                 ):
        """Initializes the BackboneNeck"""
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.freeze_backbone = freeze_backbone

    def out_strides(self) -> List[int]:
        """Returns the output strides of the backbone"""
        return self.backbone.out_strides()

    def out_channels(self) -> List[int]:
        """Returns the output channels of the backbone"""
        return self.neck.out_channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the BackboneNeck

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of output feature maps.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        y = self.neck(x)

        return y
