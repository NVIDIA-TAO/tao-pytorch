# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resnet3D backbones for action recognition."""
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet_18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet_34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet_50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet_101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet_152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding.

    This function constructs a 3x3x3 convolutional layer with the specified number of input planes, output planes, stride,
    groups, dilation, and bias. It returns the constructed convolutional layer.

    Args:
        in_planes (int): The number of input planes.
        out_planes (int): The number of output planes.
        stride (int or tuple, optional): The stride of the convolution. Defaults to 1.
        groups (int, optional): The number of groups. Defaults to 1.
        dilation (int, optional): The dilation of the convolution. Defaults to 1.

    Returns:
        nn.Conv3d: The constructed convolutional layer.
    """
    if isinstance(stride, int):
        stride = (1, stride, stride)
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution.

    This function constructs a 1x1x1 convolutional layer with the specified number of input planes, output planes, and stride.
    It returns the constructed convolutional layer.

    Args:
        in_planes (int): The number of input planes.
        out_planes (int): The number of output planes.
        stride (int or tuple, optional): The stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv3d: The constructed convolutional layer.
    """
    if isinstance(stride, int):
        stride = (1, stride, stride)
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3d(nn.Module):
    """Basic block for ResNet3D.

    This class defines a basic block for ResNet3D, which consists of two 3x3x3 convolutional layers with batch normalization
    and ReLU activation, and a residual connection. The block downsamples the input when the stride is not equal to 1.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """Initializes the basic block.

        This method initializes the basic block by defining the two 3x3x3 convolutional layers with batch normalization
        and ReLU activation, and the residual connection. The block downsamples the input when the stride is not equal to 1.

        Args:
            inplanes (int): The number of input planes.
            planes (int): The number of output planes.
            stride (int, optional): The stride of the block. Defaults to 1.
            downsample (nn.Module, optional): The downsampling layer. Defaults to None.
            groups (int, optional): The number of groups. Defaults to 1.
            base_width (int, optional): The base width. Defaults to 64.
            dilation (int, optional): The dilation of the convolution. Defaults to 1.
            norm_layer (nn.Module, optional): The normalization layer. Defaults to None.

        Raises:
            ValueError: If groups is not equal to 1 or base_width is not equal to 64.
            NotImplementedError: If dilation is greater than 1.
        """
        super(BasicBlock3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward"""
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


class Bottleneck3d(nn.Module):
    """Bottleneck module for ResNet3D.

    This class defines a bottleneck module for ResNet3D, which consists of three convolutional layers with batch normalization
    and ReLU activation, and a residual connection. The module downsamples the input when the stride is not equal to 1.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """Initializes the bottleneck module.

        This method initializes the bottleneck module by defining the three convolutional layers with batch normalization
        and ReLU activation, and the residual connection. The module downsamples the input when the stride is not equal to 1.

        Args:
            inplanes (int): The number of input planes.
            planes (int): The number of output planes.
            stride (int, optional): The stride of the module. Defaults to 1.
            downsample (nn.Module, optional): The downsampling layer. Defaults to None.
            groups (int, optional): The number of groups. Defaults to 1.
            base_width (int, optional): The base width. Defaults to 64.
            dilation (int, optional): The dilation of the convolution. Defaults to 1.
            norm_layer (nn.Module, optional): The normalization layer. Defaults to None.
        """
        super(Bottleneck3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    """ResNet3D module"""

    def __init__(self, block, layers, nb_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, modality='rgb', dropout_ratio=0.8):
        """Initializes a ResNet3D module.

        Args:
            block (nn.Module): The block type.
            layers (list of int): The number of layers in each block.
            nb_classes (int): The number of output classes.
            zero_init_residual (bool, optional): Whether to zero-initialize the last batch normalization layer in each
                residual branch. Defaults to False.
            groups (int, optional): The number of groups. Defaults to 1.
            width_per_group (int, optional): The base width. Defaults to 64.
            replace_stride_with_dilation (list of bool, optional): Whether to replace the 2x2x2 stride with a dilated
                convolution instead. Defaults to None.
            norm_layer (nn.Module, optional): The normalization layer. Defaults to None.
            modality (str, optional): The modality, either "rgb" or "of". Defaults to 'rgb'.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.8.
        """
        super(ResNet3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.modality = modality
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self._make_stem_layer()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():  # self.modules() --> Depth-First-Search the Net
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)

        if dropout_ratio > 0.0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(512 * block.expansion, nb_classes)
        self.block_expansion = block.expansion

    def replace_logits(self, nb_classes):
        """Replace the final logits with new class.

        Args:
            nb_classes (int): number of new classes.
        """
        self.fc_cls = nn.Linear(512 * self.block_expansion, nb_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Make module layer.

        Args:
            block (nn.Module): The block type.
            planes (int): The number of output planes.
            blocks (int): The number of blocks in the layer.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            dilate (bool, optional): Whether to use dilated convolution. Defaults to False.

        Returns:
            nn.Sequential: The module layer.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer.
        """
        if self.modality == 'rgb':
            inchannels = 3
        elif self.modality == 'of':
            inchannels = 2
        else:
            raise ValueError('Unknown modality: {}'.format(self.modality))
        self.conv1 = nn.Conv3d(inchannels, self.inplanes, kernel_size=(5, 7, 7),
                               stride=2, padding=(2, 3, 3), bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2,
                                    padding=(0, 1, 1))  # kernel_size=(2, 3, 3)

    def _forward_impl(self, x):
        """Forward implementation."""
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

        return x

    def forward(self, x):
        """Forward."""
        return self._forward_impl(x)

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.
        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d models.
            module_name_2d (str): The name of corresponding conv module in the
                2d models.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _inflate_bn_params(self, bn3d, state_dict_2d, module_name_2d,
                           inflated_param_names):
        """Inflate a norm module from 2d to 3d.
        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d models.
            module_name_2d (str): The name of corresponding bn module in the
                2d models.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    def inflate_weights(self, state_dict_r2d):
        """Inflate the resnet2d parameters to resnet3d.
        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d models,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        """
        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.BatchNorm3d):  # pylint:disable=R1701
                if name + '.weight' not in state_dict_r2d:
                    print(f'Module not exist in the state_dict_r2d: {name}')
                else:
                    shape_2d = state_dict_r2d[name + '.weight'].shape
                    shape_3d = module.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        print(f'Weight shape mismatch for: {name}'
                              f'3d weight shape: {shape_3d}; '
                              f'2d weight shape: {shape_2d}. ')
                    else:
                        if isinstance(module, nn.Conv3d):
                            self._inflate_conv_params(module, state_dict_r2d, name, inflated_param_names)
                        else:
                            self._inflate_bn_params(module, state_dict_r2d, name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            print(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')


def resnet3d(arch, nb_classes, progress=True, modality='rgb', pretrained2d=True,
             pretrained_weights=None, **kwargs):
    """
    Args:
        arch (str): The architecture of resnet.
        modality (str): The modality of input, 'RGB' or 'Flow'.
        progress (bool): If True, displays a progress bar of the download to stderr.
        pretrained2d (bool): If True, utilize the pretrained parameters in 2d models.
        pretrained_weights (dict): torch pretrained weights.
    """
    arch_settings = {
        'resnet_18': (BasicBlock3d, (2, 2, 2, 2)),
        'resnet_34': (BasicBlock3d, (3, 4, 6, 3)),
        'resnet_50': (Bottleneck3d, (3, 4, 6, 3)),
        'resnet_101': (Bottleneck3d, (3, 4, 23, 3)),
        'resnet_152': (Bottleneck3d, (3, 8, 36, 3))
    }

    model = ResNet3d(*arch_settings[arch], modality=modality, nb_classes=nb_classes, **kwargs)
    if pretrained2d:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.inflate_weights(state_dict)

    if pretrained_weights:
        pretrain_dict = pretrained_weights
        model.load_state_dict(pretrain_dict)

    return model
