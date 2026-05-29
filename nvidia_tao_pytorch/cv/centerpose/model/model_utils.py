# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Model functions. """

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

import math
import logging
import numpy as np
from os.path import join


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data, name, hash_code):
    """download the pretrained model"""
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash_code))


class BasicBlock(nn.Module):
    """Basic block of the DLA network"""

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        """Initialize the DLA block layers"""
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        """Forward the tensor to the basic block"""
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    """Root function for the recursive neural network module"""

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        """Initialize the root function with residual block"""
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        """Forward the feature to the recursive module"""
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    """Recursive neural network module that constructs a hierarchical tree structure of blocks"""

    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        """Initialize the tree function"""
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        """Forward function of the tree class"""
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    """Defines a Deep Layer Aggregation network architecture with varying levels and channels"""

    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        """Initialize the DLA network"""
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        """Forward an input through multiple hierarchical layers to produce a multi-scale output."""
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data, name, hash_code):
        """Load the ImageNet pretrained weights, default is off"""
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash_code)
            model_weights = model_zoo.load_url(model_url)
        self.load_state_dict(model_weights, strict=False)


def fill_fc_weights(layers):
    """Fill the fc layer with weights"""
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    """Fill up the weights for IDAUp network"""
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    """Convolution layer block, the original DLA function used DCN"""

    def __init__(self, chi, cho):
        """Initialize the convolutional layers"""
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward the image to the conv block"""
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    """IDAUP network used to upscale the feature map, sub function of the DLAUp network"""

    def __init__(self, o, channels, up_f):
        """Initialize the upscale network"""
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        """Forward the feature map to scale up"""
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    """DLAUp network used to upscale the feature map"""

    def __init__(self, startp, channels, scales, in_channels=None):
        """Initialize the DLAUp network with the scale"""
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        """Forward the feature map to scale up"""
        out = [layers[-1]]  # start with the deepest stage
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class ConvGRUCell(nn.Module):
    """ConvGRU network compute each output group from an underlying sequentially-refined hidden state"""

    def __init__(self, input_channels, hidden_channels, kernel_size):
        """Initialize the ConvGRU network"""
        super(ConvGRUCell, self).__init__()

        assert hidden_channels % 2 == 0, 'Hidden channels need to be evenly divisible.'

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wir = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whr = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wiz = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Win = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whn = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.br = None
        self.bz = None
        self.bin = None
        self.bhn = None

    def forward(self, x, h):
        """Forward the features to group"""
        rt = torch.sigmoid(self.Wir(x).to(x.device) + self.Whr(h).to(x.device) + self.br.to(x.device))  # reset
        zt = torch.sigmoid(self.Wiz(x).to(x.device) + self.Whz(h).to(x.device) + self.bz.to(x.device))  # update
        nt = torch.tanh(self.Win(x).to(x.device) + self.bin.to(x.device) + rt * (self.Whn(h).to(x.device) + self.bhn.to(x.device)))

        ht = (1 - zt) * nt + zt * h
        return ht

    def init_hidden(self, batch_size, hidden, shape, device):
        """Initialize the hidden layers"""
        if self.br is None:
            self.br = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bz = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bin = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bhn = torch.zeros(1, hidden, shape[0], shape[1]).to(device)

        else:
            assert shape[0] == self.br.size()[2], f"Input height {shape[0]} mismatched to the hidden layer height {self.br.size()[2]}."
            assert shape[1] == self.br.size()[3], f"Input width {shape[1]} mismatched to the hidden layer width {self.br.size()[3]}."
        return torch.zeros(batch_size, hidden, shape[0], shape[1]).to(device)


class ConvGRU(nn.Module):
    """
    ConvRGU network implements the idea of output grouping and sequential feature association.
    The input_channels corresponds to the first input feature map hidden state is a list of succeeding lstm layers.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        """Initialize the ConvGRU network"""
        super(ConvGRU, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs):
        """Forward the input features to ConvGRU network to perform the feature association"""
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = inputs
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    h = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                        shape=(height, width), device=x.device)
                    internal_state.append(h)

                # do forward
                h = internal_state[i]
                x = getattr(self, name)(x, h)
                internal_state[i] = x
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, x


def group_norm(out_channels):
    """Group normalization"""
    num_groups = 32
    if out_channels % 32 == 0:
        out = nn.GroupNorm(num_groups, out_channels)
    else:
        out = nn.GroupNorm(num_groups // 2, out_channels)
    return out


def get_dla_base(pretrained=True, **kwargs):  # DLA-34
    """Get the DLA base network"""
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash_code='ba72cf86')
        logger.info("Loaded the ImageNet pretrained DLA34 model")
    return model
