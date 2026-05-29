# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ResNet backbone for Sparse4D."""

import torch.nn as nn
import inspect
from typing import Dict, Optional, Tuple, Union, Type
from timm.models.resnet import Bottleneck
from nvidia_tao_pytorch.cv.backbone_v2.resnet import ResNet
from nvidia_tao_pytorch.cv.sparse4d.model.backbone.registry import SPARSE4D_BACKBONE_REGISTRY


_norm_mapping = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}


def _get_norm_abbr(layer_class: Type[nn.Module]) -> str:
    """Infer abbreviation from standard PyTorch norm layer class."""
    if not inspect.isclass(layer_class):
        raise TypeError(
            f'layer_class must be a type, but got {type(layer_class)}')

    abbr = 'norm_layer'  # Default value

    # Check specific PyTorch classes using issubclass
    # Order matters: check InstanceNorm before BatchNorm because IN inherits BN
    if issubclass(layer_class, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        abbr = 'in'
    # Check BatchNorm types (including SyncBN if supported)
    elif issubclass(layer_class, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        abbr = 'bn'
    elif issubclass(layer_class, nn.GroupNorm):
        abbr = 'gn'
    elif issubclass(layer_class, nn.LayerNorm):
        abbr = 'ln'
    else:
        # Fallback based on class name (less robust but mirrors mmcv logic)
        class_name = layer_class.__name__.lower()
        if 'batch' in class_name:
            abbr = 'bn'
        elif 'group' in class_name:
            abbr = 'gn'
        elif 'layer' in class_name:
            abbr = 'ln'
        elif 'instance' in class_name:
            abbr = 'in'
    return abbr


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer without mmcv registry.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type (e.g., 'BN', 'GN').
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    cfg = dict(cfg)
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in _norm_mapping:
        # Try case-insensitive match as a fallback
        layer_type_upper = layer_type.upper()
        if layer_type_upper in _norm_mapping:
            layer_type = layer_type_upper
        else:
            raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer_class = _norm_mapping[layer_type]
    abbr = _get_norm_abbr(norm_layer_class)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    # Common default for eps, remove if causing issues or handled differently
    cfg_.setdefault('eps', 1e-5)

    # Instantiate layer based on type - handle GN args separately
    if layer_type == 'GN':
        if 'num_groups' not in cfg_:
            raise KeyError('cfg must contain num_groups for GroupNorm')
        # GroupNorm expects num_channels keyword argument
        layer = norm_layer_class(num_channels=num_features, **cfg_)
    else:
        # Other standard norm layers take num_features as the first positional arg
        layer = norm_layer_class(num_features, **cfg_)

    # Set requires_grad for layer parameters
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


_conv_mapping = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,  # Alias 'Conv' to Conv2d
}


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in _conv_mapping:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer_class = _conv_mapping[layer_type]

    layer = conv_layer_class(*args, **kwargs, **cfg_)

    return layer


class ResNet_FPN(ResNet):
    """ResNet FPN Module class """

    def __init__(self, out_channels, return_idx=[0, 1, 2, 3], **kwargs):
        """Init"""
        super().__init__(**kwargs)
        self.return_idx = return_idx
        _out_strides = [4, 8, 16, 32]
        self.out_channels = [out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

    def forward_feature_pyramid(self, x):
        """Forward"""
        x = super().forward_intermediates(x, indices=4, intermediates_only=True)
        out = []
        for i in self.return_idx:
            out.append(x[i])
        return out


@SPARSE4D_BACKBONE_REGISTRY.register()
def resnet_50(out_indices=[0, 1, 2, 3], **kwargs):
    """ ResNet-50 model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet_FPN(out_channels=[256, 512, 1024, 2048], block=Bottleneck, layers=[3, 4, 6, 3], return_idx=out_indices, **kwargs)


@SPARSE4D_BACKBONE_REGISTRY.register()
def resnet_101(out_indices=[0, 1, 2, 3], **kwargs):
    """ ResNet-101 model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNet_FPN(out_channels=[256, 512, 1024, 2048], block=Bottleneck, layers=[3, 4, 23, 3], return_idx=out_indices, **kwargs)
