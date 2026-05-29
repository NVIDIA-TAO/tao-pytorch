# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resnet2D backbones for re-identification."""
import math
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """Creates a 3x3 convolution layer with padding.

    Args:
        in_planes (int): Number of input planes.
        out_planes (int): Number of output planes.
        stride (int, optional): Stride size. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 Convolutional layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Defines a basic block for ResNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initializes the basic block layers.

        Args:
            inplanes (int): Number of input planes.
            planes (int): Number of output planes.
            stride (int, optional): Stride size. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer, if any. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Defines the forward pass for the basic block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the basic block.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Defines a bottleneck block for ResNet."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initializes the bottleneck block layers.

        Args:
            inplanes (int): Number of input planes.
            planes (int): Number of output planes.
            stride (int, optional): Stride size. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer, if any. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Defines the forward pass for the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the bottleneck block.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet2D model."""

    def __init__(self, block, layers, last_stride, feat_dim):
        """Initializes the ResNet model.

        Args:
            block (nn.Module): Type of block to be used in the model, BasicBlock or Bottleneck.
            layers (list): Number of layers in each of the 4 blocks of the network.
            last_stride (int): Stride for the last convolutional layer.
            feat_dim (int): Dimensionality of the output feature embeddings.
        """
        self.inplanes = 64
        self.feat_dim = feat_dim
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        if self.feat_dim != 2048:
            self.feature = nn.Conv2d(2048, feat_dim, kernel_size=1, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Creates a layer of the ResNet model.

        Args:
            block (nn.Module): Type of block to be used in the layer, BasicBlock or Bottleneck.
            planes (int): Number of planes in each block.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride size. Defaults to 1.

        Returns:
            nn.Sequential: The created layer of blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass for the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the ResNet model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.feat_dim != 2048:
            x = self.feature(x)
        return x

    def load_param(self, model_path):
        """Loads parameters for the model from the given path.

        Args:
            model_path (str): Path to the saved model parameters.
        """
        param_dict = torch.load(model_path)
        for i in param_dict:
            j = i.replace("base.", "")
            if 'fc' in i:
                continue
            if j in self.state_dict().keys():  # noqa pylint: disable=missing-kwoa
                self.state_dict()[j].copy_(param_dict[i])  # noqa pylint: disable=missing-kwoa

    def random_init(self):
        """Initializes the model with random weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def cross_modality_pretrain(conv1_weight, orig_channel, target_channel):
    """Computes weights for cross modality.

    Args:
        conv1_weight (torch.Tensor): Weights of the first convolutional layer.
        orig_channel (int): Original number of channels.
        target_channel (int): Target number of channels.

    Returns:
        torch.Tensor: New weights for the first convolutional layer.
    """
    # transform the original channel weight to target channel
    S = 0
    for i in range(orig_channel):
        S += conv1_weight[:, i, :, :]
    avg = S / orig_channel
    new_conv1_weight = torch.FloatTensor(64, target_channel, 7, 7)
    for i in range(target_channel):
        new_conv1_weight[:, i, :, :] = avg.data
    return new_conv1_weight


def weight_transform(model_dict, pretrain_dict, target_channel):
    """Transforms the weights of the first convolutional layer.

    Args:
        model_dict (dict): Dictionary of the model state.
        pretrain_dict (dict): Dictionary of the pretrained model weights.
        target_channel (int): Target number of channels.

    Returns:
        dict: Updated model state dictionary with transformed weights for the first convolutional layer.
    """
    weight_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    wo = pretrain_dict[list(pretrain_dict.keys())[0]]
    orig_channel = wo.shape[1]
    if target_channel == orig_channel:
        wt = wo
    else:
        wt = cross_modality_pretrain(wo, orig_channel, target_channel)

    weight_dict['conv1.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict
