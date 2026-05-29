# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer (ViT) backbone module.

This module provides Vision Transformer implementations for the TAO PyTorch framework.
It includes both standard ViT models and MAE (Masked Autoencoder) variants.

The Vision Transformer architecture was introduced in "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale" by Dosovitskiy et al. This implementation
extends the timm library's VisionTransformer to provide additional functionality for
different image sizes and backbone integration.

Key Features:
- Support for different patch sizes (14x14, 16x16)
- Configurable model sizes (Base, Large, Huge)
- MAE (Masked Autoencoder) variants
- Positional encoding interpolation for different input resolutions
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing

Classes:
    VisionTransformer: Standard Vision Transformer implementation
    VisionTransformerMAE: Vision Transformer for Masked Autoencoder training

Functions:
    vit_base_patch16: ViT Base model with 16x16 patches
    vit_base_patch16_mae: ViT Base model for MAE training
    vit_large_patch16: ViT Large model with 16x16 patches
    vit_large_patch16_mae: ViT Large model for MAE training
    vit_huge_patch14: ViT Huge model with 14x14 patches
    vit_huge_patch14_mae: ViT Huge model for MAE training

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import vit_base_patch16
    >>> model = vit_base_patch16(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class VisionTransformer(TimmVisionTransformer, BackboneBase):
    """Vision Transformer model with enhanced functionality for TAO framework.

    This class extends the VisionTransformer class from the timm library to provide
    additional functionality for handling different image sizes, positional encoding
    interpolation, and integration with the TAO backbone framework.

    The Vision Transformer architecture processes images by:
    1. Dividing the input image into fixed-size patches (e.g., 16x16 pixels)
    2. Linearly embedding each patch into a vector
    3. Adding positional embeddings to preserve spatial information
    4. Processing the sequence through a standard Transformer encoder
    5. Using a classification head for final predictions

    Key Features:
    - Automatic positional encoding interpolation for different input resolutions
    - Support for activation checkpointing to reduce memory usage
    - Layer freezing capabilities for transfer learning
    - Integration with TAO backbone framework for feature extraction

    Attributes:
        patch_embed: Patch embedding layer that converts image patches to tokens
        pos_embed: Learnable positional embeddings
        blocks: List of Transformer blocks
        head: Classification head
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of Transformer blocks
        num_heads: Number of attention heads

    Example:
        >>> model = VisionTransformer(
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12,
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ViT model with enhanced configuration options.

        This constructor initializes both the timm VisionTransformer and the TAO
        BackboneBase to provide a unified interface for vision transformer models.

        Args:
            img_size (list, optional): Input image size. Defaults to [224, 224].
            patch_size (int, optional): Size of image patches. Defaults to 16.
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            global_pool (str, optional): Type of global pooling for final sequence.
                Options: 'token', 'avg', 'max'. Defaults to 'token'.
            embed_dim (int, optional): Transformer embedding dimension. Defaults to 768.
            depth (int, optional): Depth of transformer (number of blocks). Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): Enable bias for qkv projections if True. Defaults to True.
            init_values (float, optional): Layer-scale init values (layer-scale enabled if not None).
                Defaults to None.
            class_token (bool, optional): Use class token. Defaults to True.
            no_embed_class (bool, optional): Don't include position embeddings for class tokens.
                Defaults to False.
            reg_tokens (int, optional): Number of register tokens. Defaults to 0.
            pre_norm (bool, optional): Enable norm after embeddings, before transformer blocks.
                Defaults to False.
            final_norm (bool, optional): Enable norm after transformer blocks, before head.
                Defaults to True.
            fc_norm (bool, optional): Move final norm after pool. Defaults to None.
            drop_rate (float, optional): Head dropout rate. Defaults to 0.0.
            pos_drop_rate (float, optional): Position embedding dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
            weight_init (str, optional): Weight initialization scheme. Defaults to ''.
            fix_init (bool, optional): Apply weight initialization fix. Defaults to False.
            embed_layer (callable, optional): Patch embedding layer. Defaults to PatchEmbed.
            embed_norm_layer (callable, optional): Normalization layer for patch embed. Defaults to None.
            norm_layer (callable, optional): Normalization layer. Defaults to nn.LayerNorm.
            act_layer (callable, optional): MLP activation layer. Defaults to nn.GELU.
            block_fn (callable, optional): Transformer block layer. Defaults to Block.
            activation_checkpoint (bool, optional): Whether to use activation checkpointing.
                Defaults to False.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If None, no specific layers are frozen. If "all", the entire model is frozen and set to eval mode. Defaults to None.
            freeze_norm (bool, optional): If `True`, all normalization layers in the backbone will be frozen.
                Defaults to False.

        Note:
            The constructor handles both timm VisionTransformer initialization and TAO
            BackboneBase initialization to provide a unified interface.
        """
        in_chans = kwargs.get("in_chans", 3)
        num_classes = kwargs.get("num_classes", 1000)
        activation_checkpoint = kwargs.pop("activation_checkpoint", False)
        freeze_at = kwargs.pop("freeze_at", None)
        freeze_norm = kwargs.pop("freeze_norm", False)
        export = kwargs.pop("export", False)

        super().__init__(*args, **kwargs)  # TimmVisionTransformer initialization.
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
        different stages of the model.

        Returns:
            dict: Dictionary with stage indices as keys and corresponding modules as values.
                - Stage 0: Patch embedding layer
                - Stage 1-N: Individual transformer blocks

        Example:
            >>> model = VisionTransformer()
            >>> stages = model.get_stage_dict()
            >>> print(stages.keys())  # dict_keys([0, 1, 2, ..., 12])
        """
        stage_dict = {0: self.patch_embed}
        for i, block in enumerate(self.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    def _interpolate_pos_encoding(self, x, w, h):
        """Interpolate positional encoding to match the input resolution.

        This method handles cases where the input image size differs from the
        pre-trained model's expected size by interpolating the positional embeddings
        accordingly. This is crucial for transfer learning and inference on different
        input resolutions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) where N is the number
                of patches plus class token.
            w (int): Width of the input image in patches.
            h (int): Height of the input image in patches.

        Returns:
            torch.Tensor: Interpolated positional embeddings of shape (1, N, D).

        Raises:
            ValueError: If the interpolated positional embedding size doesn't match
                the expected patch grid size.

        Note:
            The interpolation uses bicubic interpolation to maintain smooth
            transitions between different resolutions. A small epsilon (0.1) is added
            to avoid floating point errors during interpolation.
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        reshaped_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        embed_w = patch_pos_embed.shape[-2]
        embed_h = patch_pos_embed.shape[-1]
        if int(w0) != embed_w or int(h0) != embed_h:
            raise ValueError("The interpolated value does not match the positional embedding size.")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_spatial_feat(self, x):
        """Convert token features into spatial feature maps.

        This method transforms the token-based features (excluding the class token)
        back into a spatial feature map format, which is useful for downstream
        tasks like object detection or segmentation.

        Args:
            x (torch.Tensor): Token features of shape (B, N, D) where N is the number
                of tokens (including class token) and D is the feature dimension.

        Returns:
            torch.Tensor: Spatial feature map of shape (B, D, H, W) where H and W
                are the spatial dimensions calculated from the number of patches.

        Example:
            >>> model = VisionTransformer()
            >>> x = torch.randn(1, 197, 768)  # 196 patches + 1 class token
            >>> spatial_feat = model.get_spatial_feat(x)  # Shape: (1, 768, 14, 14)
        """
        b, n, c = x.shape
        h, w = int((n - 1 + 1e-6) ** 0.5), int((n - 1 + 1e-6) ** 0.5)
        x = x[:, 1:].transpose(2, 1).reshape(b, c, h, w)
        return x

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through the patch embedding, positional
        encoding, and transformer blocks, but stops before the final classification
        head. This is useful for feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Pre-logits features of shape (B, L, D) where L is the
                sequence length (number of patches + class token) and D is the
                feature dimension.

        Example:
            >>> model = VisionTransformer()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 197, 768)
        """
        x = super().forward_features(x)
        x = super().forward_head(x, pre_logits=True)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        x = self.forward_intermediates(x, output_fmt='NCHW', intermediates_only=True)
        return x

    def forward(self, x):
        """Complete forward pass through the Vision Transformer model.

        This method performs the full forward pass including patch embedding,
        positional encoding, transformer blocks, and the classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels (typically 3), and H, W are height
                and width (typically 224x224 for standard ViT).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) where
                num_classes is the number of output classes.

        Example:
            >>> model = VisionTransformer(num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


class VisionTransformerMAE(VisionTransformer):
    """Vision Transformer for Masked Autoencoder (MAE) training.

    This class extends the standard Vision Transformer to support Masked Autoencoder
    training, where a portion of the image patches are masked during training and
    the model learns to reconstruct the original image from the visible patches.

    Key differences from standard ViT:
    - Uses mean pooling over patch tokens instead of class token
    - Designed for self-supervised learning with MAE
    - Typically used with `final_norm=False` for better reconstruction

    Attributes:
        Inherits all attributes from VisionTransformer
        Uses mean pooling over patch tokens for feature extraction

    Example:
        >>> model = VisionTransformerMAE(
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12,
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def forward_pre_logits(self, x):
        """Forward pass through the backbone for MAE, using mean pooling over patches.

        This method processes the input through the transformer but uses mean pooling
        over the patch tokens (excluding the class token) instead of using the class
        token for final representation. This is the standard approach for MAE training.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Mean-pooled features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = VisionTransformerMAE()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 768)
        """
        x = super().forward_features(x)
        x = x[:, 1:, :].mean(dim=1)  # Mean pooling over patch tokens
        return x

    def forward(self, x: torch.Tensor):
        """Complete forward pass for MAE Vision Transformer.

        This method performs the full forward pass for MAE training, including
        patch embedding, positional encoding, transformer blocks, mean pooling,
        and the classification head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes).

        Example:
            >>> model = VisionTransformerMAE(num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def vit_base_patch16(**kwargs):
    """Create a ViT Base model with 16x16 patches.

    This function creates a Vision Transformer Base model with the following
    specifications:
    - Patch size: 16x16 pixels
    - Embedding dimension: 768
    - Depth: 12 transformer blocks
    - Number of heads: 12
    - MLP ratio: 4.0
    - Final normalization: True

    Args:
        **kwargs: Additional arguments passed to VisionTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformer: Configured ViT Base model.

    Example:
        >>> model = vit_base_patch16(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 86M parameters and is suitable for
        medium-scale vision tasks.
    """
    return VisionTransformer(
        dynamic_img_size=True,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_base_patch16_mae(**kwargs):
    """Create a ViT Base model for MAE training with 16x16 patches.

    This function creates a Vision Transformer Base model specifically configured
    for Masked Autoencoder (MAE) training. The model uses mean pooling over
    patch tokens and has final normalization disabled for better reconstruction.

    Args:
        **kwargs: Additional arguments passed to VisionTransformerMAE constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformerMAE: Configured ViT Base model for MAE training.

    Example:
        >>> model = vit_base_patch16_mae(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model is specifically designed for self-supervised learning with MAE
        and uses different pooling strategy compared to standard ViT.
    """
    return VisionTransformerMAE(
        dynamic_img_size=True,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch16(**kwargs):
    """Create a ViT Large model with 16x16 patches.

    This function creates a Vision Transformer Large model with the following
    specifications:
    - Patch size: 16x16 pixels
    - Embedding dimension: 1024
    - Depth: 24 transformer blocks
    - Number of heads: 16
    - MLP ratio: 4.0
    - Final normalization: True

    Args:
        **kwargs: Additional arguments passed to VisionTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformer: Configured ViT Large model.

    Example:
        >>> model = vit_large_patch16(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is suitable for
        large-scale vision tasks requiring high accuracy.
    """
    return VisionTransformer(
        dynamic_img_size=True,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch16_mae(**kwargs):
    """Create a ViT Large model for MAE training with 16x16 patches.

    This function creates a Vision Transformer Large model specifically configured
    for Masked Autoencoder (MAE) training. The model uses mean pooling over
    patch tokens and has final normalization disabled for better reconstruction.

    Args:
        **kwargs: Additional arguments passed to VisionTransformerMAE constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformerMAE: Configured ViT Large model for MAE training.

    Example:
        >>> model = vit_large_patch16_mae(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is designed for
        self-supervised learning with MAE.
    """
    return VisionTransformerMAE(
        dynamic_img_size=True,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_huge_patch14(**kwargs):
    """Create a ViT Huge model with 14x14 patches.

    This function creates a Vision Transformer Huge model with the following
    specifications:
    - Patch size: 14x14 pixels
    - Embedding dimension: 1280
    - Depth: 32 transformer blocks
    - Number of heads: 16
    - MLP ratio: 4.0
    - Final normalization: True

    Args:
        **kwargs: Additional arguments passed to VisionTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformer: Configured ViT Huge model.

    Example:
        >>> model = vit_huge_patch14(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 632M parameters and is suitable for
        state-of-the-art vision tasks requiring maximum accuracy. The 14x14
        patch size provides finer spatial resolution compared to 16x16 patches.
    """
    return VisionTransformer(
        dynamic_img_size=True,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=True,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_huge_patch14_mae(**kwargs):
    """Create a ViT Huge model for MAE training with 14x14 patches.

    This function creates a Vision Transformer Huge model specifically configured
    for Masked Autoencoder (MAE) training. The model uses mean pooling over
    patch tokens and has final normalization disabled for better reconstruction.

    Args:
        **kwargs: Additional arguments passed to VisionTransformerMAE constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - img_size (list): Input image size. Default: [224, 224]
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False

    Returns:
        VisionTransformerMAE: Configured ViT Huge model for MAE training.

    Example:
        >>> model = vit_huge_patch14_mae(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 632M parameters and is designed for
        self-supervised learning with MAE. The 14x14 patch size provides
        finer spatial resolution for better reconstruction quality.
    """
    return VisionTransformerMAE(
        dynamic_img_size=True,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        final_norm=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
