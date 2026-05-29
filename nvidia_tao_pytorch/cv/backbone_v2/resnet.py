# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ResNet backbone module.

This module provides ResNet implementations for the TAO PyTorch framework.
ResNet (Residual Network) is a deep convolutional neural network architecture
that uses residual connections to enable training of very deep networks.

The ResNet architecture was introduced in "Deep Residual Learning for Image
Recognition" by He et al. This implementation extends the timm library's ResNet
to provide additional functionality for backbone integration and feature extraction.

Key Features:
- Support for multiple ResNet variants (18, 34, 50, 101, 152)
- Deep stem variants (18d, 34d, 50d, 101d, 152d) with improved performance
- Multi-scale feature extraction capabilities
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Residual connections for gradient flow
- Batch normalization and various activation functions

Classes:
    ResNet: Residual Network with enhanced functionality for TAO framework

Functions:
    resnet_18: ResNet-18 model with basic stem
    resnet_18d: ResNet-18D model with deep stem
    resnet_34: ResNet-34 model with basic stem
    resnet_34d: ResNet-34D model with deep stem
    resnet_50: ResNet-50 model with basic stem
    resnet_50d: ResNet-50D model with deep stem
    resnet_101: ResNet-101 model with basic stem
    resnet_101d: ResNet-101D model with deep stem
    resnet_152: ResNet-152 model with basic stem
    resnet_152d: ResNet-152D model with deep stem

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import resnet_50
    >>> model = resnet_50(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.resnet import ResNet as TimmResNet

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class ResNet(TimmResNet, BackboneBase):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net with enhanced TAO integration.

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
    have > 1 stride in the 3x3 conv layer of bottleneck and have conv-bn-act ordering.

    The ResNet architecture uses residual connections to enable training of very deep
    networks by allowing gradients to flow directly through skip connections. This
    addresses the vanishing gradient problem in deep networks.

    Architecture Overview:
    1. Initial convolution layer (conv1) with optional deep stem variants
    2. Four main stages (layer1, layer2, layer3, layer4) with increasing channels
    3. Residual blocks with skip connections
    4. Global pooling and classification head

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Key Features:
    - Residual connections for improved gradient flow
    - Multiple stem variants for different performance characteristics
    - Batch normalization for stable training
    - Multi-scale feature extraction capabilities
    - Integration with TAO backbone framework

    Attributes:
        conv1: Initial convolution layer (or deep stem)
        bn1: Batch normalization after conv1
        act1: Activation function after bn1
        maxpool: Max pooling layer
        layer1-4: Four main ResNet stages
        fc: Final classification layer
        out_indices: Indices of stages to output for feature pyramid

    Example:
        >>> model = ResNet(
        ...     block=Bottleneck,
        ...     layers=[3, 4, 6, 3],
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ResNet model with enhanced configuration options.

        This constructor initializes both the timm ResNet and the TAO BackboneBase
        to provide a unified interface for ResNet models with additional functionality
        for feature extraction and layer freezing.

        Args:
            block (nn.Module, optional): Class for the residual block. Options are BasicBlock, Bottleneck.
                Defaults to BasicBlock.
            layers (List[int], optional): Number of layers in each block. Defaults to [2, 2, 2, 2].
            num_classes (int, optional): Number of classification classes. Defaults to 1000.
            in_chans (int, optional): Number of input (color) channels. Defaults to 3.
            output_stride (int, optional): Output stride of the network, 32, 16, or 8. Defaults to 32.
            global_pool (str, optional): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'.
                Defaults to 'avg'.
            cardinality (int, optional): Number of convolution groups for 3x3 conv in Bottleneck.
                Defaults to 1.
            base_width (int, optional): Bottleneck channels factor. `planes * base_width / 64 * cardinality`.
                Defaults to 64.
            stem_width (int, optional): Number of channels in stem convolutions. Defaults to 64.
            stem_type (str, optional): The type of stem. Defaults to '':
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool, optional): Replace stem max-pooling layer with a 3x3 stride-2 convolution.
                Defaults to False.
            block_reduce_first (int, optional): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2. Defaults to 1.
            down_kernel_size (int, optional): Kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets. Defaults to 1.
            avg_down (bool, optional): Use avg pooling for projection skip connection between stages/downsample.
                Defaults to False.
            act_layer (str, nn.Module, optional): Activation layer. Defaults to nn.ReLU.
            norm_layer (str, nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            aa_layer (nn.Module, optional): Anti-aliasing layer. Defaults to None.
            drop_rate (float, optional): Dropout probability before classifier, for training. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth drop-path rate. Defaults to 0.0.
            drop_block_rate (float, optional): Drop block rate. Defaults to 0.0.
            zero_init_last (bool, optional): Zero-init the last weight in residual path (usually last BN affine weight).
                Defaults to True.
            block_args (dict, optional): Extra kwargs to pass through to block module. Defaults to {}.
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If None, no specific layers are frozen. If `"all"`, the entire model is frozen and set to eval mode.
                Defaults to None.
            freeze_norm (bool, optional): If `True`, all normalization layers in the backbone will be frozen.
                Defaults to False.
            out_indices (tuple, optional): Indices of stages to output for feature pyramid.
                Defaults to (0, 1, 2, 3).
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN

        Note:
            The constructor handles both timm ResNet initialization and TAO BackboneBase
            initialization to provide a unified interface with enhanced functionality.
        """
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)
        export = kwargs.pop("export", False)
        self.out_indices = kwargs.pop("out_indices", [0, 1, 2, 3])

        super().__init__(*args, **kwargs)  # TimmResNet initialization.
        BackboneBase.__init__(
            self,
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )

    def get_stage_dict(self):
        """Get the stage dictionary for layer freezing and feature extraction.

        Returns a dictionary mapping stage indices to their corresponding modules.
        This is used for selective layer freezing and feature extraction at
        different stages of the ResNet model.

        Returns:
            dict: Dictionary with stage indices as keys and corresponding modules as values.
                - Stage 0: Initial convolution layer (conv1)
                - Stage 1: First ResNet stage (layer1)
                - Stage 2: Second ResNet stage (layer2)
                - Stage 3: Third ResNet stage (layer3)
                - Stage 4: Fourth ResNet stage (layer4)

        Example:
            >>> model = ResNet()
            >>> stages = model.get_stage_dict()
            >>> print(stages.keys())  # dict_keys([0, 1, 2, 3, 4])
        """
        return {
            0: self.conv1,
            1: self.layer1,
            2: self.layer2,
            3: self.layer3,
            4: self.layer4,
        }

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through all ResNet layers including
        the initial convolution, batch normalization, activation, max pooling,
        and all four ResNet stages, but stops before the final classification head.
        This is useful for feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Pre-logits features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = ResNet()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 2048)
        """
        x = super().forward_features(x)
        x = super().forward_head(x, pre_logits=True)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps.

        This method extracts multi-scale features from different stages of the
        ResNet model. Each stage provides features at different spatial resolutions,
        useful for tasks like object detection or segmentation.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            dict: Dictionary containing multi-scale features with keys 'p0', 'p1', etc.
                Each value is a tensor of shape (B, C, H, W) at different resolutions.
                - p0: Features from layer1 (typically 64 channels)
                - p1: Features from layer2 (typically 128 channels)
                - p2: Features from layer3 (typically 256 channels)
                - p3: Features from layer4 (typically 512 channels)

        Example:
            >>> model = ResNet(out_indices=(0, 1, 2, 3))
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_feature_pyramid(x)
            >>> print(features.keys())  # dict_keys(['p0', 'p1', 'p2', 'p3'])
            >>> print(features['p0'].shape)  # torch.Size([1, 64, 56, 56])

        Note:
            The output indices determine which stages are included in the feature
            pyramid. Features are extracted after each main ResNet stage.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        outs = {}
        for i, name in enumerate(layer_names):
            x = getattr(self, name)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if i in self.out_indices:
                outs[f"p{i}"] = x
        return outs

    def forward(self, x):
        """Complete forward pass through the ResNet model.

        This method performs the full forward pass including initial convolution,
        batch normalization, activation, max pooling, all ResNet stages, global
        pooling, and the classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels (typically 3), and H, W are height
                and width (typically 224x224 for standard ResNet).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) where
                num_classes is the number of output classes.

        Example:
            >>> model = ResNet(num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        x = self.fc(x)
        return x


@BACKBONE_REGISTRY.register()
def resnet_18(**kwargs):
    """Create a ResNet-18 model with basic stem.

    This function creates a ResNet-18 model with the following specifications:
    - Block type: BasicBlock (2 conv layers per block)
    - Layer configuration: [2, 2, 2, 2] (2 blocks per stage)
    - Stem type: Basic 7x7 convolution
    - Total layers: 18 (counting conv layers)

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-18 model.

    Example:
        >>> model = resnet_18(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 11.7M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_18d(**kwargs):
    """Create a ResNet-18D model with deep stem.

    This function creates a ResNet-18 model with deep stem variant for improved
    performance. The deep stem uses three 3x3 convolutions instead of a single 7x7
    convolution, which can improve accuracy and training stability.

    Specifications:
    - Block type: BasicBlock (2 conv layers per block)
    - Layer configuration: [2, 2, 2, 2] (2 blocks per stage)
    - Stem type: Deep stem with 3x3 convolutions
    - Stem width: 32 channels
    - Average pooling in downsample: True

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-18D model with deep stem.

    Example:
        >>> model = resnet_18d(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 11.7M parameters and provides improved
        accuracy compared to standard ResNet-18 due to the deep stem design.
    """
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_34(**kwargs):
    """Create a ResNet-34 model with basic stem.

    This function creates a ResNet-34 model with the following specifications:
    - Block type: BasicBlock (2 conv layers per block)
    - Layer configuration: [3, 4, 6, 3] (deeper than ResNet-18)
    - Stem type: Basic 7x7 convolution
    - Total layers: 34 (counting conv layers)

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-34 model.

    Example:
        >>> model = resnet_34(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 21.8M parameters and provides better
        accuracy than ResNet-18 with moderate computational cost.
    """
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_34d(**kwargs):
    """Create a ResNet-34D model with deep stem.

    This function creates a ResNet-34 model with deep stem variant for improved
    performance. The deep stem uses three 3x3 convolutions instead of a single 7x7
    convolution.

    Specifications:
    - Block type: BasicBlock (2 conv layers per block)
    - Layer configuration: [3, 4, 6, 3] (deeper than ResNet-18)
    - Stem type: Deep stem with 3x3 convolutions
    - Stem width: 32 channels
    - Average pooling in downsample: True

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-34D model with deep stem.

    Example:
        >>> model = resnet_34d(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 21.8M parameters and provides improved
        accuracy compared to standard ResNet-34 due to the deep stem design.
    """
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_50(**kwargs):
    """Create a ResNet-50 model with basic stem.

    This function creates a ResNet-50 model with the following specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 4, 6, 3] (standard ResNet-50)
    - Stem type: Basic 7x7 convolution
    - Total layers: 50 (counting conv layers)

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-50 model.

    Example:
        >>> model = resnet_50(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 25.6M parameters and is widely used
        for various computer vision tasks due to its good accuracy/speed balance.
    """
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_50d(**kwargs):
    """Create a ResNet-50D model with deep stem.

    This function creates a ResNet-50 model with deep stem variant for improved
    performance. The deep stem uses three 3x3 convolutions instead of a single 7x7
    convolution.

    Specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 4, 6, 3] (standard ResNet-50)
    - Stem type: Deep stem with 3x3 convolutions
    - Stem width: 32 channels
    - Average pooling in downsample: True

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-50D model with deep stem.

    Example:
        >>> model = resnet_50d(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 25.6M parameters and provides improved
        accuracy compared to standard ResNet-50 due to the deep stem design.
    """
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_101(**kwargs):
    """Create a ResNet-101 model with basic stem.

    This function creates a ResNet-101 model with the following specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 4, 23, 3] (deeper middle stages)
    - Stem type: Basic 7x7 convolution
    - Total layers: 101 (counting conv layers)

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-101 model.

    Example:
        >>> model = resnet_101(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 44.6M parameters and provides high
        accuracy for demanding vision tasks.
    """
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_101d(**kwargs):
    """Create a ResNet-101D model with deep stem.

    This function creates a ResNet-101 model with deep stem variant for improved
    performance. The deep stem uses three 3x3 convolutions instead of a single 7x7
    convolution.

    Specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 4, 23, 3] (deeper middle stages)
    - Stem type: Deep stem with 3x3 convolutions
    - Stem width: 32 channels
    - Average pooling in downsample: True

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-101D model with deep stem.

    Example:
        >>> model = resnet_101d(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 44.6M parameters and provides improved
        accuracy compared to standard ResNet-101 due to the deep stem design.
    """
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_152(**kwargs):
    """Create a ResNet-152 model with basic stem.

    This function creates a ResNet-152 model with the following specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 8, 36, 3] (deeper middle stages)
    - Stem type: Basic 7x7 convolution
    - Total layers: 152 (counting conv layers)

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-152 model.

    Example:
        >>> model = resnet_152(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 60.2M parameters and provides very high
        accuracy for state-of-the-art vision tasks.
    """
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], **kwargs)


@BACKBONE_REGISTRY.register()
def resnet_152d(**kwargs):
    """Create a ResNet-152D model with deep stem.

    This function creates a ResNet-152 model with deep stem variant for improved
    performance. The deep stem uses three 3x3 convolutions instead of a single 7x7
    convolution.

    Specifications:
    - Block type: Bottleneck (3 conv layers per block)
    - Layer configuration: [3, 8, 36, 3] (deeper middle stages)
    - Stem type: Deep stem with 3x3 convolutions
    - Stem width: 32 channels
    - Average pooling in downsample: True

    Args:
        **kwargs: Additional arguments passed to ResNet constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        ResNet: Configured ResNet-152D model with deep stem.

    Example:
        >>> model = resnet_152d(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 60.2M parameters and provides improved
        accuracy compared to standard ResNet-152 due to the deep stem design.
        It is suitable for state-of-the-art vision tasks requiring maximum accuracy.
    """
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type="deep", avg_down=True, **kwargs)
