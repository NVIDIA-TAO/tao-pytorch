# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Swin Transformer backbone module.

This module provides Swin Transformer implementations for the TAO PyTorch framework.
Swin Transformer (Shifted Window Transformer) is a hierarchical vision transformer
that uses shifted windows for efficient self-attention computation.

The Swin Transformer architecture was introduced in "Swin Transformer: Hierarchical
Vision Transformer using Shifted Windows" by Liu et al. This implementation provides
a hierarchical transformer with shifted windows that brings greater efficiency by
limiting self-attention computation to non-overlapping local windows while allowing
for cross-window connections.

Key Features:
- Hierarchical architecture with multiple stages
- Shifted window attention mechanism
- Patch merging for downsampling
- Support for different window sizes (7x7, 12x12)
- Configurable model sizes (Tiny, Small, Base, Large)
- Multi-scale feature extraction
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing

Classes:
    SwinTransformer: Hierarchical Vision Transformer with shifted windows

Functions:
    swin_tiny_patch4_window7_224: Swin-T model @ 224x224
    swin_small_patch4_window7_224: Swin-S model @ 224x224
    swin_base_patch4_window7_224: Swin-B model @ 224x224
    swin_base_patch4_window12_384: Swin-B model @ 384x384
    swin_large_patch4_window7_224: Swin-L model @ 224x224
    swin_large_patch4_window12_384: Swin-L model @ 384x384

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import swin_tiny_patch4_window7_224
    >>> model = swin_tiny_patch4_window7_224(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import init_weights_vit_timm

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.swin_utils import (
    BasicLayer,
    PatchMerging,
    PatchEmbed,
)


class SwinTransformer(BackboneBase):
    """Swin Transformer using FAN blocks with hierarchical architecture.

    Swin Transformer (the name Swin stands for Shifted window) serves as a general-purpose backbone for computer
    vision. It is basically a hierarchical Transformer whose representation is computed with shifted windows. The
    shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local
    windows while also allowing for cross-window connection.

    The architecture consists of:
    1. Patch embedding layer that divides images into patches
    2. Multiple hierarchical stages with increasing feature dimensions
    3. Shifted window attention within each stage
    4. Patch merging for downsampling between stages
    5. Global average pooling and classification head

    Key Features:
    - Hierarchical design with multiple stages of increasing complexity
    - Shifted window attention for efficient computation
    - Patch merging for progressive downsampling
    - Support for absolute and relative positional embeddings
    - Multi-scale feature extraction capabilities
    - Integration with TAO backbone framework

    Attributes:
        patch_embed: Patch embedding layer that converts image patches to tokens
        layers: List of BasicLayer modules for each stage
        num_features: Final feature dimension after all stages
        num_inter_features: List of feature dimensions for each stage
        out_indices: Indices of stages to output for feature pyramid
        ape: Whether to use absolute positional embeddings
        post_norm: Optional post-normalization layer
        head: Classification head

    References:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
    - [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

    Example:
        >>> model = SwinTransformer(
        ...     img_size=224,
        ...     patch_size=4,
        ...     embed_dim=96,
        ...     depths=(2, 2, 6, 2),
        ...     num_heads=(3, 6, 12, 24),
        ...     window_size=7,
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        weight_init="",
        mlp_type="Mlp",
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        out_indices=(0, 1, 2, 3),
        dilation=False,
        post_norm=False,
        **kwargs,
    ):
        """Initialize the SwinTransformer model with hierarchical architecture.

        This constructor initializes a hierarchical vision transformer with shifted
        window attention. The model consists of multiple stages with increasing
        feature dimensions and decreasing spatial resolution.

        Args:
            img_size (int | tuple(int), optional): Input image size. Defaults to 224.
            patch_size (int | tuple(int), optional): Patch size for initial embedding. Defaults to 4.
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            embed_dim (int, optional): Initial patch embedding dimension. Defaults to 96.
            depths (tuple(int), optional): Depth of each Swin Transformer layer (number of blocks per stage).
                Defaults to (2, 2, 6, 2).
            num_heads (tuple(int), optional): Number of attention heads in different layers.
                Defaults to (3, 6, 12, 24).
            window_size (int, optional): Window size for shifted window attention. Defaults to 7.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            ape (bool, optional): If True, add absolute position embedding to the patch embedding. Defaults to False.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            weight_init (str, optional): Weight initialization scheme. Defaults to "".
            mlp_type (str | list, optional): Type of MLP layer. Defaults to "Mlp".
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If None, no specific layers are frozen. If "all", the entire model is frozen. Defaults to None.
            freeze_norm (bool, optional): If True, all normalization layers in the backbone will be frozen. Defaults to False.
            out_indices (tuple(int), optional): Indices of stages to output for feature pyramid. Defaults to (0, 1, 2, 3).
            dilation (bool, optional): Whether to use dilated convolutions. Defaults to False.
            post_norm (bool, optional): Whether to apply normalization after all layers. Defaults to False.
            **kwargs: Additional arguments passed to BackboneBase.

        Note:
            The model architecture follows a hierarchical design where each stage
            processes features at different spatial resolutions and feature dimensions.
            The shifted window attention mechanism allows efficient computation while
            maintaining global connectivity through window shifting.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.dilation = dilation

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        self.num_inter_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            self.num_inter_features[-1] = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=self.num_inter_features[i_layer],
                depth=depths[i_layer],
                mlp_type=mlp_type[i_layer] if isinstance(mlp_type, list) else mlp_type,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[: i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=downsamplelist[i_layer],
                use_checkpoint=activation_checkpoint,
            )
            self.layers.append(layer)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(self.num_inter_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        if post_norm:
            self.post_norm = nn.LayerNorm(self.num_features)
        else:
            self.post_norm = None

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        assert weight_init in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in weight_init else 0.0
        if weight_init.startswith("jax"):
            for n, m in self.named_modules():
                init_weights_vit_timm(m, n, head_bias=head_bias, jax_impl=True)  # pylint: disable=E1123
        else:
            self.apply(init_weights_vit_timm)

    def get_stage_dict(self):
        """Get the stage dictionary for layer freezing and feature extraction.

        Returns a dictionary mapping stage indices to their corresponding modules.
        This is used for selective layer freezing and feature extraction at
        different stages of the hierarchical model.

        Returns:
            dict: Dictionary with stage indices as keys and corresponding BasicLayer modules as values.
                - Stage 1-N: Individual BasicLayer modules for each hierarchical stage

        Example:
            >>> model = SwinTransformer()
            >>> stages = model.get_stage_dict()
            >>> print(stages.keys())  # dict_keys([1, 2, 3, 4])
        """
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, layer in enumerate(self.layers, start=1):
            stage_dict[i] = layer
        return stage_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        """Get the set of parameter names to exclude from weight decay.

        This method returns parameter names that should be excluded from weight
        decay during training. For Swin Transformer, this includes absolute
        positional embeddings.

        Returns:
            set: Set of parameter names to exclude from weight decay.

        Note:
            Absolute positional embeddings are typically excluded from weight decay
            as they are learned embeddings that should not be regularized.
        """
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Get the set of parameter keywords to exclude from weight decay.

        This method returns parameter keywords that should be excluded from weight
        decay during training. For Swin Transformer, this includes relative
        position bias tables.

        Returns:
            set: Set of parameter keywords to exclude from weight decay.

        Note:
            Relative position bias tables are typically excluded from weight decay
            as they represent learned positional relationships that should not be
            regularized.
        """
        return {"relative_position_bias_table"}

    def get_classifier(self):
        """Get the classifier module.

        Returns the classification head of the model, which is used for
        final classification predictions.

        Returns:
            nn.Module: The classifier head (Linear layer or Identity).

        Example:
            >>> model = SwinTransformer(num_classes=1000)
            >>> classifier = model.get_classifier()
            >>> print(type(classifier))  # <class 'torch.nn.modules.linear.Linear'>
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classifier head with a new number of classes.

        This method allows changing the number of output classes without
        reinitializing the entire model. Useful for transfer learning.

        Args:
            num_classes (int): Number of classes for the new classifier.
            global_pool (str, optional): Global pooling method (not used in Swin). Defaults to "".

        Example:
            >>> model = SwinTransformer(num_classes=1000)
            >>> model.reset_classifier(num_classes=10)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = model(x)  # Shape: (1, 10)
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through the patch embedding, positional
        encoding, and all hierarchical layers, but stops before the final
        classification head. This is useful for feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Features of shape (B, L, D) where L is the sequence length
                and D is the feature dimension.

        Example:
            >>> model = SwinTransformer()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_features(x)  # Shape: (1, 49, 768)
        """
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for layer in self.layers:
            _, _, _, x, Wh, Ww = layer(x, Wh, Ww)
        if self.post_norm:
            x = self.post_norm(x)  # B L C
        return x

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through all layers and applies global
        average pooling to prepare for classification, but stops before the
        final classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Pre-logits features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = SwinTransformer()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 768)
        """
        x = self.forward_features(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps.

        This method extracts multi-scale features from different stages of the
        hierarchical model. Each stage provides features at different spatial
        resolutions, useful for tasks like object detection or segmentation.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            dict: Dictionary containing multi-scale features with keys 'p0', 'p1', etc.
                Each value is a tensor of shape (B, C, H, W) at different resolutions.

        Example:
            >>> model = SwinTransformer(out_indices=(0, 1, 2, 3))
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = model.forward_feature_pyramid(x)
            >>> print(features.keys())  # dict_keys(['p0', 'p1', 'p2', 'p3'])
            >>> print(features['p0'].shape)  # torch.Size([1, 96, 56, 56])
        """
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for idx, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_inter_features[idx]).permute(0, 3, 1, 2).contiguous()
                outs[f'p{idx}'] = out
        return outs

    def forward(self, x):
        """Complete forward pass through the Swin Transformer model.

        This method performs the full forward pass including patch embedding,
        hierarchical processing through all stages, global pooling, and the
        classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels (typically 3), and H, W are height
                and width (typically 224x224 or 384x384 for standard Swin).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) where
                num_classes is the number of output classes.

        Example:
            >>> model = SwinTransformer(num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def swin_tiny_patch4_window7_224(**kwargs):
    """Create a Swin-T model with 4x4 patches and 7x7 windows at 224x224 resolution.

    This function creates a Swin Transformer Tiny model with the following
    specifications:
    - Image size: 224x224
    - Patch size: 4x4 pixels
    - Window size: 7x7
    - Embedding dimension: 96
    - Depths: (2, 2, 6, 2) - number of blocks in each stage
    - Number of heads: (3, 6, 12, 24) - attention heads in each stage
    - Trained on ImageNet-1k

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-T model.

    Example:
        >>> model = swin_tiny_patch4_window7_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 28M parameters and is suitable for
        efficient vision tasks with good accuracy.
    """
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_small_patch4_window7_224(**kwargs):
    """Create a Swin-S model with 4x4 patches and 7x7 windows at 224x224 resolution.

    This function creates a Swin Transformer Small model with the following
    specifications:
    - Image size: 224x224
    - Patch size: 4x4 pixels
    - Window size: 7x7
    - Embedding dimension: 96
    - Depths: (2, 2, 18, 2) - deeper middle stages
    - Number of heads: (3, 6, 12, 24) - attention heads in each stage

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-S model.

    Example:
        >>> model = swin_small_patch4_window7_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 50M parameters and provides a good
        balance between accuracy and computational efficiency.
    """
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window7_224(**kwargs):
    """Create a Swin-B model with 4x4 patches and 7x7 windows at 224x224 resolution.

    This function creates a Swin Transformer Base model with the following
    specifications:
    - Image size: 224x224
    - Patch size: 4x4 pixels
    - Window size: 7x7
    - Embedding dimension: 128 (larger than Tiny/Small)
    - Depths: (2, 2, 18, 2) - deeper middle stages
    - Number of heads: (4, 8, 16, 32) - more attention heads

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-B model.

    Example:
        >>> model = swin_base_patch4_window7_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 88M parameters and is suitable for
        high-accuracy vision tasks.
    """
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_base_patch4_window12_384(**kwargs):
    """Create a Swin-B model with 4x4 patches and 12x12 windows at 384x384 resolution.

    This function creates a Swin Transformer Base model optimized for higher
    resolution images with the following specifications:
    - Image size: 384x384
    - Patch size: 4x4 pixels
    - Window size: 12x12 (larger for higher resolution)
    - Embedding dimension: 128
    - Depths: (2, 2, 18, 2) - deeper middle stages
    - Number of heads: (4, 8, 16, 32) - more attention heads

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-B model for 384x384 resolution.

    Example:
        >>> model = swin_base_patch4_window12_384(num_classes=1000)
        >>> x = torch.randn(1, 3, 384, 384)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 88M parameters and is designed for
        higher resolution inputs, providing better accuracy for detailed
        visual understanding tasks.
    """
    return SwinTransformer(
        img_size=(384, 384),
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window7_224(**kwargs):
    """Create a Swin-L model with 4x4 patches and 7x7 windows at 224x224 resolution.

    This function creates a Swin Transformer Large model with the following
    specifications:
    - Image size: 224x224
    - Patch size: 4x4 pixels
    - Window size: 7x7
    - Embedding dimension: 192 (larger than Base)
    - Depths: (2, 2, 18, 2) - deeper middle stages
    - Number of heads: (6, 12, 24, 48) - more attention heads

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-L model.

    Example:
        >>> model = swin_large_patch4_window7_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 197M parameters and is suitable for
        state-of-the-art vision tasks requiring maximum accuracy.
    """
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def swin_large_patch4_window12_384(**kwargs):
    """Create a Swin-L model with 4x4 patches and 12x12 windows at 384x384 resolution.

    This function creates a Swin Transformer Large model optimized for higher
    resolution images with the following specifications:
    - Image size: 384x384
    - Patch size: 4x4 pixels
    - Window size: 12x12 (larger for higher resolution)
    - Embedding dimension: 192 (larger than Base)
    - Depths: (2, 2, 18, 2) - deeper middle stages
    - Number of heads: (6, 12, 24, 48) - more attention heads

    Args:
        **kwargs: Additional arguments passed to SwinTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: 1000
            - in_chans (int): Number of input channels. Default: 3
            - activation_checkpoint (bool): Enable activation checkpointing. Default: False
            - freeze_at (list): Layers to freeze. Default: None
            - freeze_norm (bool): Freeze normalization layers. Default: False
            - out_indices (tuple): Stages to output for feature pyramid. Default: (0, 1, 2, 3)

    Returns:
        SwinTransformer: Configured Swin-L model for 384x384 resolution.

    Example:
        >>> model = swin_large_patch4_window12_384(num_classes=1000)
        >>> x = torch.randn(1, 3, 384, 384)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 197M parameters and is designed for
        high-resolution inputs, providing state-of-the-art accuracy for
        detailed visual understanding tasks.
    """
    return SwinTransformer(
        img_size=(384, 384),
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )
