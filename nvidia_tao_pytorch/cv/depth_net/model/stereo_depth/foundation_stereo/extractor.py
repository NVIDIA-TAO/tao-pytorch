# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extractor module consists of various encoder architectures"""

import os
from typing import Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.convolution_helper import Conv, Conv2xDownScale
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo import utils
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.submodule import LayerNorm2d
from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.pl_mono_model import MonoDepthNetPlModel
from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.depth_anything_v2.dpt import RelativeDepthAnythingV2
from nvidia_tao_pytorch.cv.backbone_v2.edgenext import EdgeNeXt


class FeatureAtt(nn.Module):
    """
    Computes multiplicative feature matching between the cost volume and input feature.

    This module implements a feature attention mechanism that uses a 2D feature map
    to "attend" to a 3D cost volume. It learns a per-channel multiplicative
    attention mask for the cost volume based on the input feature.

    Args:
        cv_chan (int): The number of channels in the cost volume.
        feat_chan (int): The number of channels in the input feature map.
    """

    def __init__(self, cv_chan, feat_chan):
        super().__init__()

        # A small sub-network to produce the attention map from the feature map.
        # It consists of a 1x1 convolution to reduce the number of channels,
        # followed by another 1x1 convolution to match the cost volume's channel
        # dimension.
        self.feat_att = nn.Sequential(
            Conv(feat_chan, feat_chan // 2,
                 conv_type='conv2d', norm_type='instance2d',
                 relu=True, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1))

    def forward(self, cv, feat):
        """
        Defines the computation performed at every call.

        The forward pass computes a 2D attention map from the feature map,
        expands it to a 3D tensor to match the dimensions of the cost volume,
        applies a sigmoid activation, and then performs a element-wise
        multiplication with the cost volume.

        Args:
            cv (torch.Tensor): The cost volume tensor of shape
                `(B, C, D, H, W)`, where `B` is the batch size, `C` is the
                number of channels, `D` is the disparity/depth dimension,
                and `H` and `W` are the height and width.
            feat (torch.Tensor): The input feature map tensor of shape
                `(B, C_feat, H, W)`, where `B` is the batch size, `C_feat` is the
                number of channels, and `H` and `W` are the height and width.

        Returns:
            torch.Tensor: The attended cost volume tensor, with the same shape
                as the input cost volume `(B, C, D, H, W)`.
        """
        # Generate the 2D attention map from the feature map
        feat_att = self.feat_att(feat)

        # Unsqueeze the attention map to match the cost volume's dimensions
        # This adds a new dimension for the disparity/depth dimension.
        # Shape changes from (B, cv_chan, H, W) to (B, cv_chan, 1, H, W)
        feat_att = feat_att.unsqueeze(2)

        # Apply a sigmoid activation to the attention map to constrain values
        # between 0 and 1.
        # Perform element-wise multiplication with the cost volume.
        # This acts as a per-channel, per-pixel multiplicative gate for the cost volume.
        cv = torch.sigmoid(feat_att) * cv

        return cv


class EdgeNextConvEncoder(nn.Module):
    """
    An EdgeNextConvEncoder block as proposed in the EdgeNeXt paper.

    This block is a building block for the EdgeNeXt architecture. It
    consists of a depth-wise convolution, followed by a normalization
    layer, and then two point-wise convolutions with a GELU activation
    in between. It also includes a layer scaling mechanism and a residual
    connection.

    Args:
        dim (int): The number of input and output channels.
        layer_scale_init_value (float, optional): Initial value for the
            learnable gamma parameter in layer scaling. If 0 or less,
            layer scaling is disabled. Defaults to 1e-6.
        expan_ratio (int, optional): The expansion ratio for the first
            point-wise convolution. Used only when ``pwconv1_dim`` is
            None; the hidden dimension is then ``dim * expan_ratio``.
            Defaults to 4.
        kernel_size (int, optional): The kernel size for the depth-wise
            convolution. Defaults to 7.
        norm (str, optional): The type of normalization layer to use.
            Can be 'layer', 'batch', or 'none'. Defaults to 'layer'.
        pwconv1_dim (Optional[int], optional): Explicit hidden dimension
            for the first pointwise convolution. When None, derived from
            ``expan_ratio * dim``.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6,
                 expan_ratio=4, kernel_size=7, norm='layer',
                 pwconv1_dim: Optional[int] = None):
        super().__init__()

        # Depth-wise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)

        # Normalization layer
        if norm == 'layer':
            self.norm = LayerNorm2d(dim, eps=1e-6)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = nn.Identity()

        # Point-wise convolutions
        if pwconv1_dim is None:
            pwconv1_dim = expan_ratio * dim
        self.pwconv1 = nn.Linear(dim, pwconv1_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(pwconv1_dim, dim)

        # Layer scaling gamma parameter
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        """
        Defines the computation performed at every call.

        The forward pass of the EdgeNextConvEncoder block applies a
        depth-wise convolution, followed by normalization, and then a
        two-layer MLP with a residual connection. The input tensor is
        expected to be in the format `(N, C, H, W)`.

        Args:
            x (torch.Tensor): The input tensor of shape `(N, C, H, W)`.

        Returns:
            torch.Tensor: The output tensor after applying the block,
                of the same shape `(N, C, H, W)`.
        """
        input_feature = x

        # Apply depth-wise convolution
        x = self.dwconv(x)

        # Apply normalization
        x = self.norm(x)

        # Reshape for point-wise convolutions (Linear layers operate on the last dimension)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # Apply point-wise convolutions with GELU activation
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Apply layer scaling if enabled
        if self.gamma is not None:
            x = self.gamma * x

        # Reshape back to the original format
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Add the residual connection
        x = input_feature + x
        return x


class ResidualBlock(nn.Module):
    """Residual Bottleneck block.

    This class is a bottleneck block used in the Resnet model construciton.
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """Constructor for ResidualBlock class.

        Args:
            in_planes (int): The channel dimension of input features into the block.
            planes (str): Channel dimension of intermediate features in the block.
            norm_fn (str): Feature normalization type.
            stride (int): Size of convolutional strides.
        Returns:
            N/A

        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()
        else:
            valid_norm_function_options = ["group", "batch", "instance", "layer", "none"]
            raise NotImplementedError(
                f"Norm function: ({norm_fn}) requested is not supported. \
                    Please choose one among the following options {valid_norm_function_options}")

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        """Forward function for Residual Block."""
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class MultiBasicEncoder(nn.Module):
    """
    Feature extractor encoder.

    This class uses a multi-block approach to extract 3 spatially sized
    features from an input image at different scales. It is designed to
    produce features that are downsampled by factors of 4, 8, and 16
    relative to the input image size.

    For all three stages of features, the network outputs multiple
    features, making a total of 9 feature tensors for a single input image.
    This is useful for multi-scale applications like stereo matching or
    optical flow.
    """

    def __init__(self, output_dim=([128, 128, 128], [128, 128, 128]),
                 norm_fn='batch', dropout=0.0, downsample=3):
        """
        Constructor for MultiBasicEncoder class.

        Args:
            output_dim (list of list of int): A list of lists, where each inner
                list specifies the channel dimensions for the output features
                at the different downsampling levels (4x, 8x, 16x).
                For example, `[[128, 128, 128], [128, 128, 128]]` would mean
                two sets of outputs, each with 128 channels at 4x, 8x, and 16x
                downsampling.
            norm_fn (str): Specifies the normalization type used in the model
                graph. Supported values are 'batch', 'instance', 'group', 'layer',
                and 'none'. Defaults to 'batch'.
            dropout (float): The dropout probability. If 0, dropout is not used.
                Defaults to 0.0.
            downsample (int): The number of stages to downsample the features.
                This parameter controls the spatial scaling of features at each
                layer compared to the preceding layer. Defaults to 3.
        """
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        # Initial convolution and normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'layer':
            self.norm1 = LayerNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        else:
            raise IndexError("please select norm_fn from the following list \
                             {['group', 'batch', 'instance', 'layer'] or set it to none!}")

        self.relu1 = nn.ReLU(inplace=True)

        # Residual blocks for feature extraction at different scales
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        # Output layers for the 4x downsampled features
        self.outputs04 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, norm_fn=self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            self.outputs04.append(conv_out)

        # Output layers for the 8x downsampled features
        self.outputs08 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, norm_fn=self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            self.outputs08.append(conv_out)

        # Output layers for the 16x downsampled features
        self.outputs16 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            self.outputs16.append(conv_out)

        # Optional dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # Initialization of weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        Helper function to create a layer of two ResidualBlocks.

        Args:
            dim (int): The number of output channels for the layer.
            stride (int, optional): The stride for the first block in the layer.
                Defaults to 1.

        Returns:
            nn.Sequential: A sequential module containing two residual blocks.
        """
        layer1 = ResidualBlock(in_planes=self.in_planes, planes=dim, norm_fn=self.norm_fn, stride=stride)
        layer2 = ResidualBlock(in_planes=dim, planes=dim, norm_fn=self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):
        """
        Forward pass for the MultiBasicEncoder.

        It processes an input image through multiple residual blocks and
        outputs feature maps at different resolutions. It can optionally
        handle a dual input for tasks like stereo or optical flow, and
        allows for specifying the number of output layers.

        Args:
            x (torch.Tensor): The input image tensor of shape `(B, 3, H, W)`.
            dual_inp (bool, optional): If `True`, the input `x` is assumed to
                contain two concatenated images, and the output `v` will
                contain the features for the second image. Defaults to `False`.
            num_layers (int, optional): The number of downsampled feature levels
                to output. Can be 1, 2, or 3. Defaults to 3.

        Returns:
            tuple: A tuple containing the output feature maps.
                - If `dual_inp` is `False`: Returns a tuple of tuples, where each
                    inner tuple contains the feature maps at a specific scale
                    (e.g., `(outputs04, outputs08, outputs16)`).
                - If `dual_inp` is `True`: Returns a tuple containing the
                    feature maps and the features for the second image.
                    (e.g., `(outputs04, outputs08, outputs16, v)`).
        """
        # Initial convolution and normalization
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Residual blocks for initial downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Handle dual input if specified
        if dual_inp:
            v = x
            x = x[:(x.shape[0] // 2)]

        # Get 4x downsampled features
        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        # Get 8x downsampled features
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        # Get 16x downsampled features
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)


class ContextNetwork(MultiBasicEncoder):
    """
    Feature extractor for a context network.

    This class inherits from `MultiBasicEncoder` and extends it to incorporate
    features from an external source, such as a Vision Transformer (ViT),
    for enhanced context understanding. It is designed to extract multi-scale
    features from an input image, fusing them with ViT features at an
    intermediate stage.
    """

    def __init__(self, cfg, output_dim=([128, 128, 128], [128, 128, 128]), norm_fn='batch', downsample=3):
        """
        Constructor for ContextNetwork class.

        Args:
            cfg (object): A configuration object containing model settings. It is
                expected to have an attribute `encoder` which specifies the
                encoder type, used to retrieve the ViT feature dimension.
            output_dim (list of list of int): A list of lists, where each inner
                list specifies the channel dimensions for the output features
                at the different downsampling levels (4x, 8x, 16x).
                Defaults to `([128, 128, 128], [128, 128, 128])`.
            norm_fn (str): Specifies the normalization type used in the model
                graph. Supported values are 'batch' and 'instance'. Defaults
                to 'batch'.
            downsample (int): The number of stages to downsample the features.
                This parameter controls the spatial scaling of features at each
                layer compared to the preceding layer. Defaults to 3.
        """
        super().__init__(output_dim=output_dim, norm_fn=norm_fn, downsample=downsample)

        self.patch_size = 14
        self.image_size = 518
        self.norm_fn = norm_fn

        # Overriding the initial layers from MultiBasicEncoder to handle fusion later
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        # Downsampling layer for the initial features before fusion
        self.down = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0),
                                  nn.BatchNorm2d(128))

        # Get ViT feature dimension from the configuration
        vit_dim = DepthAnythingFeature.model_configs[cfg.encoder]['features'] // 2

        # Fusion layer: concatenates the CNN features with the ViT features
        self.conv2 = Conv(128 + vit_dim, 128, conv_type='conv2d', norm_type='instance2d',
                          relu=True, kernel_size=3, padding=1)

        # A normalization layer for the fused features
        self.norm = nn.BatchNorm2d(256)

        # Output layers for the 4x downsampled features (post-fusion)
        self.outputs04 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, norm_fn=self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            self.outputs04.append(conv_out)

        # Output layers for the 8x downsampled features (post-fusion)
        self.outputs08 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            self.outputs08.append(conv_out)

        # Output layers for the 16x downsampled features (post-fusion)
        self.outputs16 = nn.ModuleList()
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            self.outputs16.append(conv_out)

    def forward(self, x_in, vit_feat, dual_inp=False, num_layers=3):
        """
        Forward pass for the ContextNetwork.

        It processes an input image through initial CNN layers, fuses the
        resulting features with external Vision Transformer (ViT) features,
        and then continues to extract multi-scale features.

        Args:
            x_in (torch.Tensor): The input image tensor of shape `(B, 3, H, W)`.
            vit_feat (torch.Tensor): The features from an external Vision
                Transformer, expected to have a shape compatible for concatenation
                with the CNN features.
            dual_inp (bool, optional): This parameter is inherited from
                `MultiBasicEncoder` but is not used in this specific implementation.
                Defaults to `False`.
            num_layers (int, optional): The number of downsampled feature levels
                to output. This implementation always returns all three levels
                (4x, 8x, 16x) and this parameter is ignored. Defaults to 3.

        Returns:
            tuple: A tuple containing the output feature maps at different scales:
                `(outputs04, outputs08, outputs16)`.
        """
        # Initial CNN feature extraction
        x = self.conv1(x_in)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Feature fusion: concatenate CNN features with ViT features
        x = torch.cat([x, vit_feat], dim=1)
        x = self.conv2(x)

        # Get 4x downsampled features (after fusion)
        outputs04 = [f(x) for f in self.outputs04]

        # Get 8x downsampled features
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        # Get 16x downsampled features
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        # Return a tuple of all three feature scales
        return (outputs04, outputs08, outputs16)


class DepthAnythingFeature(nn.Module):
    """
    Feature extractor for DepthAnythingV2.

    This class encapsulates the feature extraction and depth prediction
    components of the DepthAnythingV2 model. It is designed to load a
    pretrained model, extract intermediate features from its Vision Transformer
    (ViT) encoder, and pass them through a depth prediction head.
    """

    # Configuration for different ViT encoder variants
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    def __init__(self, cfg, export=False):
        """
        Constructor for DepthAnythingFeature class.

        Args:
            cfg (hydra.OmegaConf.DictConfig): A configuration object from Hydra,
                expected to contain `depth_anythingv2_pretrained_path` and `encoder`.
        """
        super().__init__()

        pretrained_path = cfg.stereo_backbone.depth_anything_v2_pretrained_path
        encoder = cfg.encoder

        # Initialize the DepthAnythingV2 model with the specified configuration
        depth_anything_config = {'encoder': encoder}
        depth_anything_config.update({'mono_backbone': {'use_bn': cfg.stereo_backbone.use_bn,
                                                        'use_clstoken': cfg.stereo_backbone.use_clstoken,
                                                        'pretrained_path': None}})

        depth_anything = RelativeDepthAnythingV2(depth_anything_config, max_depth=None, export=export)

        # Load pretrained weights from a checkpoint file
        # when the module is a pytorch lightning module,
        # we load with strict=False otherwise, we set strict to True.
        if pretrained_path:
            checkpoint_weight = load_pretrained_weights(pretrained_path, map_location='cpu')
            if "pytorch-lightning_version" not in checkpoint_weight:
                depth_anything.load_state_dict(checkpoint_weight, strict=False)
            else:
                MonoDepthNetPlModel(cfg).load_from_checkpoint(
                    pretrained_path,
                    map_location="cpu",
                    experiment_spec=cfg,
                    strict=True)

        self.depth_anything = depth_anything

        # Define the indices of the intermediate layers to extract features from
        self.encoder = depth_anything.model_configs[encoder]['intermediate_layer_idx']

    def forward(self, x):
        """
        Defines the computation performed at every call.

        The forward pass extracts features from the Vision Transformer encoder
        at specific intermediate layers, and then passes these features through
        the depth prediction head to generate a disparity map and other
        intermediate outputs.

        Args:
            x (torch.Tensor): The input image tensor of shape `(B, C, H, W)`.

        Returns:
            dict: A dictionary containing various outputs from the depth head.
                - 'out' (torch.Tensor): The final depth prediction output.
                - 'path_1' (torch.Tensor): Intermediate feature from the depth head.
                - 'path_2' (torch.Tensor): Intermediate feature from the depth head.
                - 'path_3' (torch.Tensor): Intermediate feature from the depth head.
                - 'path_4' (torch.Tensor): Intermediate feature from the depth head.
                - 'features' (list of torch.Tensor): The raw intermediate features
                    extracted from the ViT encoder.
                - 'disp' (torch.Tensor): The predicted disparity map.
        """
        h, w = x.shape[-2:]

        # Extract intermediate features from the pretrained ViT encoder
        features = self.depth_anything.pretrained.get_intermediate_layers(
            x, self.encoder, return_class_token=True)

        # Calculate patch dimensions
        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size

        # Pass the features through the depth prediction head
        out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(
            features, patch_h, patch_w, normalize_output=True)

        return {'out': out, 'path_1': path_1, 'path_2': path_2,
                'path_3': path_3, 'path_4': path_4, 'features': features, 'disp': disp}


class Feature(nn.Module):
    """
    Feature extractor or encoder for left and right images.

    This class combines a pretrained EdgeNeXt model with a DepthAnythingV2
    feature extractor to create a powerful encoder. It first uses
    DepthAnythingV2 to extract rich, high-level features and then uses
    the layers of EdgeNeXt as a U-Net style encoder-decoder to refine and
    upsample these features, resulting in multi-scale output feature maps.
    """

    def __init__(self, cfg, export=False, use_vit_backbone: bool = True,
                 phantom_vit_feat_dim: Optional[int] = None):
        """
        Constructor for the Feature class.

        Args:
            cfg (omegaconf.dictconfig.DictConfig): A configuration object
                containing parameters such as `edgenext_pretrained_path` and
                `encoder`.
            export (bool, optional): If True, the underlying DepthAnythingV2
                feature extractor is built in export mode. Defaults to False.
            use_vit_backbone (bool, optional): If True, a frozen DepthAnythingV2
                ViT branch is fused into the 1/4 decoder output. If False, the
                ViT branch is omitted and the 1/4 conv4 input drops by
                ``vit_feat_dim`` channels. Defaults to True.
            phantom_vit_feat_dim (Optional[int], optional): When set (and
                ``use_vit_backbone=False``), allocates this many "phantom"
                channels in ``conv4``'s output and switches ``conv4`` to a
                single 1x1 expansion ``Conv2d``. Matches upstream-ckpt layouts
                that reserve the ViT slot without running a ViT branch.
                Defaults to None (legacy Sequential fusion+refine layout).
        """
        super().__init__()

        if phantom_vit_feat_dim is not None and use_vit_backbone:
            raise ValueError(
                "phantom_vit_feat_dim is only meaningful with "
                "use_vit_backbone=False; the ViT branch already populates "
                "the slot when enabled.")

        # Initialize and load pretrained EdgeNeXt model
        model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         d2_scales=[2, 2, 3, 4],
                         classifier_dropout=0.0)

        if os.path.isfile(cfg.stereo_backbone.edgenext_pretrained_path):
            load_dict = torch.load(cfg.stereo_backbone.edgenext_pretrained_path,
                                   weights_only=False)
            state_dict = load_dict['state_dict']

            # Adjust state_dict keys to match the model's structure
            state_dict = utils.process_edgenext_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=True)

        self.downsample_layers = model.downsample_layers
        self.stages = model.stages

        self.channel = [48, 96, 160, 304]   # channel sizes hardcoded for decoder design

        # Initialize and freeze the DepthAnythingV2 feature extractor (optional)
        self.use_vit_backbone = use_vit_backbone
        if self.use_vit_backbone:
            self.dino = DepthAnythingFeature(cfg, export=export)
            self.dino = utils.freeze_model(self.dino)
            vit_feat_dim = DepthAnythingFeature.model_configs[cfg.encoder]['features'] // 2
        else:
            self.dino = None
            vit_feat_dim = phantom_vit_feat_dim if phantom_vit_feat_dim is not None else 0
        extra_channel_boost = 0

        # Decoder part: Deconvolutional layers for upsampling
        self.deconv32_16 = Conv2xDownScale(
            self.channel[3], self.channel[2], norm_type='instance2d',
            conv_type='deconv2d', concat=True, use_resnet_layer=True)
        self.deconv16_8 = Conv2xDownScale(
            self.channel[2] * 2, self.channel[1], norm_type='instance2d',
            conv_type='deconv2d', concat=True, use_resnet_layer=True)
        self.deconv8_4 = Conv2xDownScale(
            self.channel[1] * 2, self.channel[0], norm_type='instance2d',
            conv_type='deconv2d', concat=True, use_resnet_layer=True)

        if phantom_vit_feat_dim is not None and not self.use_vit_backbone:
            # Single 1x1 expansion that reserves a ViT-sized channel band
            # without a fusion or residual-refine stack.
            self.conv4 = nn.Conv2d(
                self.channel[0] * 2,
                self.channel[0] * 2 + vit_feat_dim,
                kernel_size=1, padding=0)
        else:
            # Default: Sequential(Conv + 2x ResidualBlock) for fusion+refine.
            self.conv4 = nn.Sequential(
                Conv(self.channel[0] * 2 + vit_feat_dim,
                     self.channel[0] * 2 + vit_feat_dim + extra_channel_boost,
                     conv_type='conv2d', norm_type='instance2d', relu=True,
                     kernel_size=3, stride=1, padding=1),
                ResidualBlock(self.channel[0] * 2 + vit_feat_dim + extra_channel_boost,
                              self.channel[0] * 2 + vit_feat_dim + extra_channel_boost,
                              norm_fn='instance'),
                ResidualBlock(self.channel[0] * 2 + vit_feat_dim + extra_channel_boost,
                              self.channel[0] * 2 + vit_feat_dim, norm_fn='instance'))

        self.patch_size = 14
        self.d_out = [self.channel[0] * 2 + vit_feat_dim,
                      self.channel[1] * 2,
                      self.channel[2] * 2, self.channel[3]]

    def forward(self, x):
        """
        Forward pass for the Feature extractor.

        It processes an input image through the DepthAnythingV2 model to
        get a disparity map and features, and also through the EdgeNeXt
        encoder-decoder path to produce multi-scale feature maps. The
        DepthAnythingV2 features are fused into the main network path.

        Args:
            x (torch.Tensor): The input image tensor of shape `(B, 3, H, W)`.

        Returns:
            tuple: A tuple containing two elements:
                - A list of `torch.Tensor`: The multi-scale feature maps
                  `[x4, x8, x16, x32]`, with dimensions corresponding to
                  the `d_out` attribute.
                - `torch.Tensor`: The `vit_feat` output from the DepthAnythingV2
                  model, resized to the 4x downsampling level.
        """
        _, _, height, width = x.shape

        if self.use_vit_backbone:
            # Resize input image for DepthAnythingV2, keeping aspect ratio
            divider = np.lcm(self.patch_size, 16)
            h_resize, w_resize = utils.get_resize_keep_aspect_ratio(
                height, width, divider=divider, max_h=1344, max_w=1344)
            x_in_ = F.interpolate(x, size=(h_resize, w_resize), mode='bicubic', align_corners=False)

            # Extract features from DepthAnythingV2 (in evaluation mode with no_grad)
            self.dino = self.dino.eval()
            with torch.no_grad():
                output = self.dino(x_in_)
            vit_feat = output['out']

            # Resize ViT features to match the 4x downsampling level
            vit_feat = F.interpolate(
                vit_feat, size=(height // 4, width // 4), mode='bilinear', align_corners=True)
        else:
            vit_feat = None

        # EdgeNeXt encoder path
        x = self.downsample_layers[0](x)
        x4 = self.stages[0](x)
        x4_ds = self.downsample_layers[1](x4)
        x8 = self.stages[1](x4_ds)
        x8_ds = self.downsample_layers[2](x8)
        x16 = self.stages[2](x8_ds)
        x16_ds = self.downsample_layers[3](x16)
        x32 = self.stages[3](x16_ds)

        # Decoder path with skip connections and feature fusion
        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)

        # Concatenate upsampled features with ViT features and process with final convolution
        if self.use_vit_backbone:
            x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.conv4(x4)

        return [x4, x8, x16, x32], vit_feat
