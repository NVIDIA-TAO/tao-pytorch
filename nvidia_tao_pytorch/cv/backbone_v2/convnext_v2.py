# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvNeXt V2 backbone module.

This module provides ConvNeXt V2 implementations for the TAO PyTorch framework.
ConvNeXt V2 is an improved version of ConvNeXt that introduces Global Response
Normalization (GRN) to enhance inter-channel feature competition and improve
model performance.

The ConvNeXt V2 architecture was introduced in "ConvNeXt V2: Co-designing and
Scaling ConvNets with Masked Autoencoders" by Liu et al. This implementation
extends the ConvNeXt V2 architecture to provide additional functionality for
backbone integration and feature extraction.

Key Features:
- Modern convolutional architecture with transformer-inspired design
- Global Response Normalization (GRN) for improved feature competition
- Support for multiple model sizes (Atto, Femto, Pico, Nano, Tiny, Base, Large, Huge)
- Layer scale initialization for better convergence
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design with good accuracy/speed balance

Classes:
    Block: ConvNeXt and ConvNeXtV2 building block
    ConvNeXtV2: Modern convolutional neural network with enhanced TAO integration

Functions:
    convnextv2_atto: ConvNeXt V2 Atto model
    convnextv2_femto: ConvNeXt V2 Femto model
    convnextv2_pico: ConvNeXt V2 Pico model
    convnextv2_nano: ConvNeXt V2 Nano model
    convnextv2_tiny: ConvNeXt V2 Tiny model
    convnextv2_base: ConvNeXt V2 Base model
    convnextv2_large: ConvNeXt V2 Large model
    convnextv2_huge: ConvNeXt V2 Huge model

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import convnextv2_base
    >>> model = convnextv2_base(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import GlobalResponseNorm, LayerNorm2d


class Block(nn.Module):
    """ConvNeXt and ConvNeXtV2 building block.

    This block implements the core building unit of ConvNeXt and ConvNeXtV2 architectures.
    It combines depthwise convolution, pointwise convolution, and normalization layers
    to create an efficient and powerful feature extraction module.

    If `use_grn=True`, use `GlobalResponseNorm` instead of layer scale which is the ConvNeXtV2 style.
    If `use_grn=False`, use layer scale which is the ConvNeXt V1 style.

    Architecture:
    1. Depthwise convolution (7x7 kernel)
    2. Layer normalization
    3. Pointwise convolution (1x1) with GELU activation
    4. Global Response Normalization (GRN) or Layer Scale
    5. Pointwise convolution (1x1)
    6. Residual connection with drop path

    Attributes:
        dwconv: Depthwise convolution layer
        norm: Layer normalization
        pwconv1: First pointwise convolution
        act: GELU activation function
        grn: Global Response Normalization (ConvNeXt V2) or Identity (ConvNeXt V1)
        gamma: Layer scale parameter (ConvNeXt V1) or None (ConvNeXt V2)
        pwconv2: Second pointwise convolution
        drop_path: Drop path for regularization

    Example:
        >>> block = Block(dim=96, use_grn=True)
        >>> x = torch.randn(1, 96, 56, 56)
        >>> output = block(x)  # Shape: (1, 96, 56, 56)
    """

    def __init__(self, dim, drop_path=0.0, use_grn=True, layer_scale_init_value=0.0):
        """Initialize the ConvNeXtV2 block.

        Args:
            dim (int): Number of input channels.
            drop_path (float, optional): Stochastic depth rate. Defaults to `0.0`.
            use_grn (bool, optional): Whether to use `GlobalResponseNorm`.
                True for ConvNeXt V2, False for ConvNeXt V1. Defaults to `True`.
            layer_scale_init_value (float, optional): Init value for Layer Scale.
                Used only when `use_grn=False`. Defaults to `0.0`.
        """
        super().__init__()
        self.use_grn = bool(use_grn)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if use_grn:
            self.grn = GlobalResponseNorm(4 * dim)
            self.gamma = None
        else:
            self.grn = nn.Identity()
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ConvNeXt block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_grn:
            x = self.grn(x)
        x = self.pwconv2(x)
        if not self.use_grn and self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_tensor + self.drop_path(x)
        return x


class ConvNeXtV2(BackboneBase):
    """ConvNeXt V1 and V2 model with enhanced TAO integration.

    ConvNeXts is ConvNets that modernizes the classic ResNet design by incorporating
    elements inspired by vision Transformers. This results in a model that achieves
    competitive accuracy and scalability compared to Transformers, while retaining
    the simplicity and efficiency of ConvNets.

    ConvNeXt V2 introduces the Global Response Normalization (GRN) layer to the
    ConvNeXt architecture to enhance inter-channel feature competition and improve
    model performance.

    Architecture Overview:
    1. Stem layer with 4x4 convolution and normalization
    2. Four stages with increasing feature dimensions
    3. Downsampling layers between stages
    4. Multiple ConvNeXt blocks per stage
    5. Global average pooling and classification head

    Key Features:
    - Hierarchical design with multiple stages
    - Global Response Normalization for improved feature competition
    - Depthwise and pointwise convolutions for efficiency
    - Residual connections for gradient flow
    - Integration with TAO backbone framework

    Attributes:
        downsample_layers: List of downsampling layers (stem + 3 intermediate)
        stages: List of ConvNeXt stages, each containing multiple blocks
        norm: Final layer normalization
        head: Classification head
        num_features: Final feature dimension
        num_stages: Number of stages in the model

    Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    - [https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
    - [https://github.com/facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)

    Example:
        >>> model = ConvNeXtV2(
        ...     depths=[3, 3, 27, 3],
        ...     dims=[128, 256, 512, 1024],
        ...     use_grn=True,
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        # `use_grn` and `layer_scale_init_value` are for switching between ConvNeXt and ConvNeXtV2.
        # ConvNeXt uses `use_grn=False` and `layer_scale_init_value=1e-6`.
        # ConvNeXtV2 uses `use_grn=True` and `layer_scale_init_value=0.0`.
        use_grn=True,
        layer_scale_init_value=0.0,
        drop_path_rate=0.0,
        head_init_scale=1.0,
        export_pre_logits=False,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the ConvNextV2 model with enhanced configuration options.

        This constructor initializes a ConvNeXt V2 model with the specified architecture
        and provides additional functionality for backbone integration and feature extraction.

        Args:
            in_chans (int, optional): Number of input image channels. Defaults to `3`.
            num_classes (int, optional): Number of classes for classification head. Defaults to `1000`.
            depths (list, optional): Number of blocks at each stage. Defaults to `[3, 3, 9, 3]`.
            dims (list, optional): Feature dimension at each stage. Defaults to `[96, 192, 384, 768]`.
            use_grn (bool, optional): Whether to use `GlobalResponseNorm`.
                True for ConvNeXt V2, False for ConvNeXt V1. Defaults to `True`.
            layer_scale_init_value (float, optional): Init value for Layer Scale.
                Used only when `use_grn=False`. Defaults to `0.0`.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to `0.0`.
            head_init_scale (float, optional): Init scaling value for classifier weights and biases.
                Defaults to `1.0`.
            export_pre_logits (bool, optional): Whether to export the pre_logits features.
                Defaults to `False`.
            activation_checkpoint (bool, optional): Whether to use activation checkpointing.
                Defaults to `False`.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If `None`, no specific layers are frozen. If `"all"`, the entire model is frozen.
                Defaults to `None`.
            freeze_norm (bool, optional): If `True`, all normalization layers in the backbone will be frozen.
                Defaults to `False`.
            **kwargs: Additional arguments passed to `BackboneBase` constructor.

        Note:
            The model architecture follows a hierarchical design with four stages of
            increasing complexity. Each stage contains multiple `ConvNeXt` blocks with
            downsampling between stages.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.depths = depths
        self.dims = dims
        self.use_grn = use_grn
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = head_init_scale
        self.export_pre_logits = export_pre_logits

        self.num_features = dims[-1]
        self.num_stages = len(depths)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        use_grn=self.use_grn,
                        layer_scale_init_value=self.layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        if num_classes > 0:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Set the gradient checkpointing for the model.

        This method enables or disables gradient checkpointing to trade compute
        for memory during training.

        Args:
            enable (bool): Whether to enable gradient checkpointing. Defaults to `True`.
        """
        self.activation_checkpoint = enable

    def get_stage_dict(self):
        """Get the stage dictionary for layer freezing and feature extraction.

        Returns a dictionary mapping stage indices to their corresponding modules.
        This is used for selective layer freezing and feature extraction at
        different stages of the hierarchical model.

        Returns:
            dict: Dictionary with stage indices as keys and corresponding modules as values.
                - Stage 0: Stem layer (downsample_layers[0])
                - Stage 1-N: Individual ConvNeXt stages

        Example:
            >>> model = ConvNeXtV2()
            >>> stages = model.get_stage_dict()
            >>> print(stages.keys())  # dict_keys([0, 1, 2, 3, 4])
        """
        stage_dict = {0: self.downsample_layers[0]}
        for i, stage in enumerate(self.stages, start=1):
            # TODO(yuw, hongyuc): Should we freeze the downsample layers?
            stage_dict[i] = stage
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module.

        Returns the classification head of the model, which is used for
        final classification predictions.

        Returns:
            nn.Module: The classifier head (Linear layer or Identity).

        Example:
            >>> model = ConvNeXtV2(num_classes=1000)
            >>> classifier = model.get_classifier()
            >>> print(type(classifier))  # <class 'torch.nn.modules.linear.Linear'>
        """
        return self.head

    def reset_classifier(self, num_classes):
        """Reset the classifier head with a new number of classes.

        This method allows changing the number of output classes without
        reinitializing the entire model. Useful for transfer learning.

        Args:
            num_classes (int): Number of classes for the new classifier.

        Example:
            >>> model = ConvNeXtV2(num_classes=1000)
            >>> model.reset_classifier(num_classes=10)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = model(x)  # Shape: (1, 10)
        """
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(self.head_init_scale)
            self.head.bias.data.mul_(self.head_init_scale)
        else:
            self.head = nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through all ConvNeXt layers including
        downsampling layers and all stages, but stops before the final
        classification head. This is useful for feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Pre-logits features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = ConvNeXtV2()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 768)
        """
        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
            outs.append(x)
        return outs

    def forward(self, x):
        """Complete forward pass through the ConvNeXt V2 model.

        This method performs the full forward pass including all downsampling layers,
        ConvNeXt stages, global pooling, and the classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels (typically 3), and H, W are height
                and width (typically 224x224 for standard ConvNeXt).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) where
                num_classes is the number of output classes.

        Example:
            >>> model = ConvNeXtV2(num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        if self.export_pre_logits:
            return x
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def convnextv2_atto(**kwargs):
    """Create a ConvNeXt V2 Atto model.

    This function creates a ConvNeXt V2 Atto model with the following specifications:
    - Depths: [2, 2, 6, 2] - number of blocks in each stage
    - Dimensions: [40, 80, 160, 320] - feature dimensions for each stage
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Atto model.

    Example:
        >>> model = convnextv2_atto(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 3.7M parameters and is suitable for
        very efficient vision tasks.
    """
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_femto(**kwargs):
    """Create a ConvNeXt V2 Femto model.

    This function creates a ConvNeXt V2 Femto model with the following specifications:
    - Depths: [2, 2, 6, 2] - number of blocks in each stage
    - Dimensions: [48, 96, 192, 384] - feature dimensions for each stage
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Femto model.

    Example:
        >>> model = convnextv2_femto(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 5.2M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_pico(**kwargs):
    """Create a ConvNeXt V2 Pico model.

    This function creates a ConvNeXt V2 Pico model with the following specifications:
    - Depths: [2, 2, 6, 2] - number of blocks in each stage
    - Dimensions: [64, 128, 256, 512] - feature dimensions for each stage
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Pico model.

    Example:
        >>> model = convnextv2_pico(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 9.1M parameters and provides good
        accuracy for efficient vision tasks.
    """
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_nano(**kwargs):
    """Create a ConvNeXt V2 Nano model.

    This function creates a ConvNeXt V2 Nano model with the following specifications:
    - Depths: [2, 2, 8, 2] - deeper middle stages than Pico
    - Dimensions: [80, 160, 320, 640] - feature dimensions for each stage
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Nano model.

    Example:
        >>> model = convnextv2_nano(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 15.6M parameters and provides better
        accuracy than Pico with moderate computational cost.
    """
    return ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_tiny(**kwargs):
    """Create a ConvNeXt V2 Tiny model.

    This function creates a ConvNeXt V2 Tiny model with the following specifications:
    - Depths: [3, 3, 9, 3] - number of blocks in each stage
    - Dimensions: [96, 192, 384, 768] - feature dimensions for each stage
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Tiny model.

    Example:
        >>> model = convnextv2_tiny(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 28M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_base(**kwargs):
    """Create a ConvNeXt V2 Base model.

    This function creates a ConvNeXt V2 Base model with the following specifications:
    - Depths: [3, 3, 27, 3] - deeper middle stages than Tiny
    - Dimensions: [128, 256, 512, 1024] - larger dimensions than Tiny
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Base model.

    Example:
        >>> model = convnextv2_base(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 89M parameters and provides good
        accuracy for various computer vision tasks.
    """
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_large(**kwargs):
    """Create a ConvNeXt V2 Large model.

    This function creates a ConvNeXt V2 Large model with the following specifications:
    - Depths: [3, 3, 27, 3] - same depth as Base
    - Dimensions: [192, 384, 768, 1536] - larger dimensions than Base
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Large model.

    Example:
        >>> model = convnextv2_large(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 198M parameters and provides high
        accuracy for demanding vision tasks.
    """
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@BACKBONE_REGISTRY.register()
def convnextv2_huge(**kwargs):
    """Create a ConvNeXt V2 Huge model.

    This function creates a ConvNeXt V2 Huge model with the following specifications:
    - Depths: [3, 3, 27, 3] - same depth as Large
    - Dimensions: [352, 704, 1408, 2816] - largest dimensions available
    - Global Response Normalization: True (ConvNeXt V2 style)
    - Layer scale initialization: 0.0

    Args:
        **kwargs: Additional arguments passed to ConvNeXtV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXtV2: Configured ConvNeXt V2 Huge model.

    Example:
        >>> model = convnextv2_huge(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 659M parameters and provides state-of-the-art
        accuracy for vision tasks requiring maximum performance.
    """
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
