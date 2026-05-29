# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resnet2D backbones for action recognition."""
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet_18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet_34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet_50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet_101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet_152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.

    This function constructs a 3x3 convolutional layer with padding=1 using the specified input planes, output planes,
    and stride. It returns the convolutional layer.

    Args:
        in_planes (int): The number of input planes.
        out_planes (int): The number of output planes.
        stride (int, optional): The stride of the convolution. Defaults to 1.

    Returns:
        torch.nn.Conv2d: The constructed convolutional layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    This class defines a basic block for ResNet that inherits from the `nn.Module` class. It constructs a basic block
    using the specified input planes, output planes, stride, and downsample. It defines a `forward` method that applies
    the basic block to the input tensor and returns the output tensor.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initialize the BasicBlock.

        Args:
            inplanes (int): The number of input planes.
            planes (int): The number of output planes.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            downsample (nn.Module, optional): The downsampling layer. Defaults to None.
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
        """forward"""
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
    """Bottleneck block for ResNet.

    This class defines a bottleneck block for ResNet that inherits from the `nn.Module` class
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initialize Bottleneck.
        Args:
            inplanes (int): The number of input planes.
            planes (int): The number of output planes.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            downsample (nn.Module, optional): The downsampling layer. Defaults to None.
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
        """forward"""
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
    """ResNet2D module.

    This class defines a ResNet2D model that inherits from the `nn.Module` class
    """

    def __init__(self, block, layers, nb_classes=101, channel=20, dropout_ratio=0.0):
        """Initialize the ResNet

        Args:
            block (nn.Module): The block to use in the ResNet2D model.
            layers (list of int): The number of layers in each block.
            nb_classes (int, optional): The number of classes. Defaults to 101.
            channel (int, optional): The number of input channels. Defaults to 20.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.0.
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AvgPool2d(7)
        self.block_expansion = block.expansion
        self.fc_cls = nn.Linear(512 * block.expansion, nb_classes)
        if dropout_ratio > 0.0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        #  Uncomment for already trained models
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """Construct the layers with module template.

        Args:
            block (nn.Module): The block template to use in the layer.
            planes (int): The number of output planes.
            blocks (int): The number of blocks in the layer.
            stride (int, optional): The stride of the convolution. Defaults to 1.

        Returns:
            nn.Sequential: The constructed layer.
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

    def replace_logits(self, nb_classes):
        """Replace final logits.

        This method replaces the final logits layer of the ResNet2D model with a new linear layer that has the specified number
        of output classes.

        Args:
            nb_classes (int): The number of output classes.
        """
        self.fc_cls = nn.Linear(512 * self.block_expansion, nb_classes)

    def forward(self, x):
        """forward"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.fc_cls(x)
        return out


def resnet2d(backbone,
             pretrained_weights=None,
             channel=3,
             nb_classes=5,
             imagenet_pretrained=False,  # @TODO(tylerz)Internal test option
             **kwargs):
    """
    ResNet2D.

    This function constructs a ResNet2D model using the specified backbone, pretrained weights, number of input channels,
    number of classes, and additional keyword arguments. It returns the constructed ResNet2D model.

    Args:
        backbone (str): The backbone to use in the ResNet2D model.
        pretrained_weights (dict, optional): The pretrained weights. Defaults to None.
        channel (int, optional): The number of input channels. Defaults to 3.
        nb_classes (int, optional): The number of classes. Defaults to 5.
        imagenet_pretrained (bool, optional): Whether to use pretrained weights from ImageNet. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.Module: The constructed ResNet2D model.
    """
    arch_settings = {
        'resnet_18': (BasicBlock, [2, 2, 2, 2]),
        'resnet_34': (BasicBlock, [3, 4, 6, 3]),
        'resnet_50': (Bottleneck, [3, 4, 6, 3]),
        'resnet_101': (Bottleneck, [3, 4, 23, 3]),
        'resnet_152': (Bottleneck, [3, 8, 36, 3])
    }
    model = ResNet(arch_settings[backbone][0], arch_settings[backbone][1],
                   nb_classes=nb_classes, channel=channel, **kwargs)
    model_dict = model.state_dict()

    if pretrained_weights:
        pretrain_dict = pretrained_weights
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    else:
        if imagenet_pretrained:  # @TODO(tylerz) Internal test option
            pretrain_dict = model_zoo.load_url(
                model_urls[backbone], 'tmp/')
            model_dict = model.state_dict()
            model_dict = weight_transform(model_dict, pretrain_dict, channel)
            model.load_state_dict(model_dict)

    return model


def cross_modality_pretrain(conv1_weight, orig_channel, target_channel):
    """Compute weights for cross modality.

    This function computes the weights for cross modality by transforming the original channel weight to the target channel.
    It returns the new convolutional weight tensor.

    Args:
        conv1_weight (torch.Tensor): The original convolutional weight tensor.
        orig_channel (int): The number of original channels.
        target_channel (int): The number of target channels.

    Returns:
        torch.Tensor: The new convolutional weight tensor.
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
    """Weight transform.

    This function transforms the weights of the first convolutional layer of a model using the specified pretrained weights
    and target channel. It returns the transformed model weights.

    Args:
        model_dict (dict): The model weights.
        pretrain_dict (dict): The pretrained weights.
        target_channel (int): The number of target channels.

    Returns:
        dict: The transformed model weights.
    """
    weight_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    wo = pretrain_dict[list(pretrain_dict.keys())[0]]
    orig_channel = wo.shape[1]
    if target_channel == orig_channel:
        wt = wo
    else:
        print(list(pretrain_dict.keys())[0])
        print("orig_channel: {} VS target_channel: {}".format(orig_channel, target_channel))
        wt = cross_modality_pretrain(wo, orig_channel, target_channel)

    weight_dict['conv1.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict
