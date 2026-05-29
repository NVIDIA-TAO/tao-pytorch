# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvNeXt backbone module.

This module provides ConvNeXt implementations for the TAO PyTorch framework.
ConvNeXt is a modern convolutional neural network architecture that combines
the best practices from ResNet and Vision Transformers to achieve state-of-the-art
performance on image classification tasks.

The ConvNeXt architecture was introduced in "A ConvNet for the 2020s" by Liu et al.
This implementation extends the ConvNeXtV2 architecture to provide additional
functionality for backbone integration and feature extraction.

Key Features:
- Modern convolutional architecture with transformer-inspired design
- Support for multiple model sizes (Tiny, Small, Base, Large, XLarge)
- Global Response Normalization (GRN) for improved training stability
- Layer scale initialization for better convergence
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design with good accuracy/speed balance

Classes:
    ConvNeXtV2: Modern convolutional neural network with enhanced TAO integration

Functions:
    convnext_tiny: ConvNeXt Tiny model
    convnext_small: ConvNeXt Small model
    convnext_base: ConvNeXt Base model
    convnext_large: ConvNeXt Large model
    convnext_xlarge: ConvNeXt XLarge model

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import convnext_base
    >>> model = convnext_base(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import ConvNeXtV2 as ConvNeXt


@BACKBONE_REGISTRY.register()
def convnext_tiny(**kwargs):
    """Create a ConvNeXt Tiny model.

    This function creates a ConvNeXt Tiny model with the following specifications:
    - Depths: [3, 3, 9, 3] - number of blocks in each stage
    - Dimensions: [96, 192, 384, 768] - feature dimensions for each stage
    - Global Response Normalization: False (ConvNeXt v1 style)
    - Layer scale initialization: 1e-6

    Args:
        **kwargs: Additional arguments passed to ConvNeXt constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXt: Configured ConvNeXt Tiny model.

    Example:
        >>> model = convnext_tiny(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 28M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return ConvNeXt(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_small(**kwargs):
    """Create a ConvNeXt Small model.

    This function creates a ConvNeXt Small model with the following specifications:
    - Depths: [3, 3, 27, 3] - deeper middle stages than Tiny
    - Dimensions: [96, 192, 384, 768] - same dimensions as Tiny
    - Global Response Normalization: False (ConvNeXt v1 style)
    - Layer scale initialization: 1e-6

    Args:
        **kwargs: Additional arguments passed to ConvNeXt constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXt: Configured ConvNeXt Small model.

    Example:
        >>> model = convnext_small(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 50M parameters and provides better
        accuracy than Tiny with moderate computational cost.
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_base(**kwargs):
    """Create a ConvNeXt Base model.

    This function creates a ConvNeXt Base model with the following specifications:
    - Depths: [3, 3, 27, 3] - same depth as Small
    - Dimensions: [128, 256, 512, 1024] - larger dimensions than Small
    - Global Response Normalization: False (ConvNeXt v1 style)
    - Layer scale initialization: 1e-6

    Args:
        **kwargs: Additional arguments passed to ConvNeXt constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXt: Configured ConvNeXt Base model.

    Example:
        >>> model = convnext_base(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 89M parameters and provides good
        accuracy for various computer vision tasks.
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_large(**kwargs):
    """Create a ConvNeXt Large model.

    This function creates a ConvNeXt Large model with the following specifications:
    - Depths: [3, 3, 27, 3] - same depth as Base
    - Dimensions: [192, 384, 768, 1536] - larger dimensions than Base
    - Global Response Normalization: False (ConvNeXt v1 style)
    - Layer scale initialization: 1e-6

    Args:
        **kwargs: Additional arguments passed to ConvNeXt constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXt: Configured ConvNeXt Large model.

    Example:
        >>> model = convnext_large(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 198M parameters and provides high
        accuracy for demanding vision tasks.
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )


@BACKBONE_REGISTRY.register()
def convnext_xlarge(**kwargs):
    """Create a ConvNeXt XLarge model.

    This function creates a ConvNeXt XLarge model with the following specifications:
    - Depths: [3, 3, 27, 3] - same depth as Large
    - Dimensions: [256, 512, 1024, 2048] - largest dimensions available
    - Global Response Normalization: False (ConvNeXt v1 style)
    - Layer scale initialization: 1e-6

    Args:
        **kwargs: Additional arguments passed to ConvNeXt constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        ConvNeXt: Configured ConvNeXt XLarge model.

    Example:
        >>> model = convnext_xlarge(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 350M parameters and provides state-of-the-art
        accuracy for vision tasks requiring maximum performance.
    """
    return ConvNeXt(
        depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], use_grn=False, layer_scale_init_value=1e-6, **kwargs
    )
