# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hiera backbone module.

This module provides Hiera implementations for the TAO PyTorch framework.
Hiera is a hierarchical vision transformer that is fast, powerful, and simple.
It outperforms state-of-the-art models across a wide array of image and video
tasks while being much faster.

The Hiera architecture was introduced in "Hiera: A Hierarchical Vision Transformer
without the Bells-and-Whistles" by Chaitanya et al. This implementation extends
the timm library's Hiera to provide additional functionality for backbone integration.

Key Features:
- Hierarchical vision transformer architecture
- Fast and efficient computation
- Support for multiple model sizes (Tiny, Small, Base, Base+, Large, Huge)
- Masked modeling capabilities
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Simple and clean architecture without complex components

Classes:
    Hiera: Hierarchical Vision Transformer with enhanced TAO integration

Functions:
    hiera_tiny_224: Hiera Tiny model @ 224x224
    hiera_small_224: Hiera Small model @ 224x224
    hiera_base_224: Hiera Base model @ 224x224
    hiera_base_plus_224: Hiera Base+ model @ 224x224
    hiera_large_224: Hiera Large model @ 224x224
    hiera_huge_224: Hiera Huge model @ 224x224

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import hiera_base_224
    >>> model = hiera_base_224(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

import torch
from timm.models.hiera import Hiera as TimmHiera

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class Hiera(TimmHiera, BackboneBase):
    """Hiera model with enhanced TAO integration.

    Hiera is a hierarchical vision transformer that is fast, powerful, and, above all, simple.
    It outperforms the state-of-the-art across a wide array of image and video tasks while
    being much faster than other vision transformers.

    The Hiera architecture is designed to be simple and efficient, removing many of the
    complex components found in other vision transformers while maintaining high performance.
    It uses a hierarchical structure with multiple stages of increasing complexity.

    Key Features:
    - Hierarchical design with multiple stages
    - Fast and efficient computation
    - Masked modeling capabilities for self-supervised learning
    - Simple architecture without complex attention mechanisms
    - Integration with TAO backbone framework
    - Support for activation checkpointing and layer freezing

    Architecture Overview:
    1. Patch embedding layer that divides images into patches
    2. Multiple hierarchical stages with increasing complexity
    3. Masked modeling capabilities for self-supervised learning
    4. Global pooling and classification head

    Attributes:
        blocks: List of transformer blocks for each stage
        patch_embed: Patch embedding layer
        head: Classification head
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        stages: Configuration of stages

    References:
    - [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](
      https://arxiv.org/abs/2306.00989)
    - [https://github.com/facebookresearch/hiera](https://github.com/facebookresearch/hiera)

    Example:
        >>> model = Hiera(
        ...     embed_dim=96,
        ...     num_heads=1,
        ...     stages=(2, 3, 16, 3),
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Hiera model with enhanced configuration options.

        This constructor initializes both the timm Hiera and the TAO `BackboneBase`
        to provide a unified interface for Hiera models with additional functionality
        for feature extraction and layer freezing.

        Args:
            embed_dim (int, optional): Embedding dimension. Defaults to `96`.
            num_heads (int, optional): Number of attention heads. Defaults to `1`.
            stages (tuple, optional): Configuration of stages. Defaults to `(2, 3, 16, 3)`.
            num_classes (int, optional): Number of classification classes. Defaults to `1000`.
            in_chans (int, optional): Number of input image channels. Defaults to `3`.
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to `False`.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If `None`, no specific layers are frozen. If "all", the entire model is frozen.
                Defaults to `None`.
            freeze_norm (bool, optional): If `True`, all normalization layers in the backbone will be frozen.
                Defaults to `False`.
            **kwargs: Additional arguments passed to `TimmHiera` constructor.

        Note:
            The constructor handles both timm Hiera initialization and TAO `BackboneBase`
            initialization to provide a unified interface with enhanced functionality.
        """
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)
        export = kwargs.pop("export", False)

        super().__init__(*args, **kwargs)  # TimmHiera initialization.
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
        different stages of the hierarchical model.

        Returns:
            dict: Dictionary with stage indices as keys and corresponding transformer blocks as values.
                - Stage 1-N: Individual transformer blocks for each hierarchical stage

        Example:
            >>> model = Hiera()
            >>> stages = model.get_stage_dict()
            >>> print(stages.keys())  # dict_keys([1, 2, 3, 4])
        """
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, block in enumerate(self.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    def forward_pre_logits(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through all Hiera layers including
        patch embedding and all transformer blocks, but stops before the final
        classification head. This is useful for feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.
            mask (torch.Tensor, optional): Boolean mask tensor of shape [B, #MUt*#MUy*#MUx]
                where #MU are the number of mask units in that dimension. 1 in mask is *keep*,
                0 is *remove*. mask.sum(dim=-1) should be the same across the batch.
                Defaults to None.

        Returns:
            torch.Tensor: Pre-logits features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = Hiera()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 768)
        """
        return super().forward_features(x, mask, return_intermediates=False)

    def forward_feature_pyramid(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass through the backbone to extract intermediate feature maps."""
        return super().forward_intermediates(x, mask, intermediates_only=True, output_fmt="NCHW")


@BACKBONE_REGISTRY.register()
def hiera_tiny_224(**kwargs):
    """Create a Hiera Tiny model at 224x224 resolution.

    This function creates a Hiera Tiny model with the following specifications:
    - Embedding dimension: 96
    - Number of heads: 1
    - Stages: (1, 2, 7, 2) - configuration of transformer blocks per stage
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        Hiera: Configured Hiera Tiny model.

    Example:
        >>> model = hiera_tiny_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 6M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_small_224(**kwargs):
    """Create a Hiera Small model at 224x224 resolution.

    This function creates a Hiera Small model with the following specifications:
    - Embedding dimension: 96
    - Number of heads: 1
    - Stages: (1, 2, 11, 2) - deeper middle stages than Tiny
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        Hiera: Configured Hiera Small model.

    Example:
        >>> model = hiera_small_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 22M parameters and provides better
        accuracy than Tiny with moderate computational cost.
    """
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_base_224(**kwargs):
    """Create a Hiera Base model at 224x224 resolution.

    This function creates a Hiera Base model with the following specifications:
    - Embedding dimension: 96
    - Number of heads: 1
    - Stages: (2, 3, 16, 3) - deeper stages than Small
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        Hiera: Configured Hiera Base model.

    Example:
        >>> model = hiera_base_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 35M parameters and provides good
        accuracy for various computer vision tasks.
    """
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_base_plus_224(**kwargs):
    """Create a Hiera Base+ model at 224x224 resolution.

    This function creates a Hiera Base+ model with the following specifications:
    - Embedding dimension: 112 (larger than Base)
    - Number of heads: 2 (more attention heads than Base)
    - Stages: (2, 3, 16, 3) - same stage configuration as Base
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        Hiera: Configured Hiera Base+ model.

    Example:
        >>> model = hiera_base_plus_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 47M parameters and provides improved
        accuracy compared to Base due to larger embedding dimension and more
        attention heads.
    """
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_large_224(**kwargs):
    """Create a Hiera Large model at 224x224 resolution.

    This function creates a Hiera Large model with the following specifications:
    - Embedding dimension: 144 (larger than Base+)
    - Number of heads: 2
    - Stages: (2, 6, 36, 4) - much deeper middle stages
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        Hiera: Configured Hiera Large model.

    Example:
        >>> model = hiera_large_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 213M parameters and provides high
        accuracy for demanding vision tasks.
    """
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwargs)


@BACKBONE_REGISTRY.register()
def hiera_huge_224(**kwargs):
    """Create a Hiera Huge model at 224x224 resolution.

    This function creates a Hiera Huge model with the following specifications:
    - Embedding dimension: 256 (largest available)
    - Number of heads: 4 (more attention heads than Large)
    - Stages: (2, 6, 36, 4) - same stage configuration as Large
    - Image size: 224x224

    Args:
        **kwargs: Additional arguments passed to Hiera constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        Hiera: Configured Hiera Huge model.

    Example:
        >>> model = hiera_huge_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 632M parameters and provides state-of-the-art
        accuracy for vision tasks requiring maximum performance.
    """
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwargs)
