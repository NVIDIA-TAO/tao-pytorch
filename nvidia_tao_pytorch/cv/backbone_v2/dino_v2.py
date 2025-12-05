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

"""DINOv2 backbone module.

This module provides DINOv2 implementations for the TAO PyTorch framework.
DINOv2 is a self-supervised learning method that learns robust visual features
without supervision. It extends the Vision Transformer architecture with additional
components for improved feature learning.

The DINOv2 architecture was introduced in "DINOv2: Learning Robust Visual Features
without Supervision" by Oquab et al. This implementation extends the Vision Transformer
to provide additional functionality for backbone integration and feature extraction.

Key Features:
- Self-supervised learning without labels
- Robust visual feature learning
- Support for register tokens for enhanced representation
- SwiGLU activation functions for improved performance
- Multiple model sizes (Large, Giant)
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Flexible image size handling

Classes:
    DINOV2: DINOv2 Vision Transformer with enhanced TAO integration

Functions:
    vit_large_patch14_dinov2_swiglu: DINOv2 ViT Large model with SwiGLU activation
    vit_large_patch14_dinov2_swiglu_legacy: DINOv2 ViT Large model (legacy version)
    vit_giant_patch14_reg4_dinov2_swiglu: DINOv2 ViT Giant model with register tokens

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import vit_large_patch14_dinov2_swiglu
    >>> model = vit_large_patch14_dinov2_swiglu(num_classes=1000)
    >>> x = torch.randn(1, 3, 518, 518)
    >>> output = model(x)
"""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import PatchEmbed, SwiGLUPacked

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.vit import VisionTransformer


class DINOV2(VisionTransformer):
    """DINOv2 model with enhanced TAO integration.

    This class extends the VisionTransformer by adding register tokens and
    handling different image sizes. DINOv2 is designed for self-supervised
    learning of robust visual features without requiring labeled data.

    The DINOv2 architecture introduces several key innovations:
    1. Register tokens for enhanced representation learning
    2. SwiGLU activation functions for improved performance
    3. Flexible image size handling
    4. Self-supervised learning objectives

    Architecture Overview:
    1. Patch embedding layer that divides images into patches
    2. Positional embeddings with interpolation support
    3. Class token and optional register tokens
    4. Transformer blocks with SwiGLU activation
    5. Global pooling and classification head

    Key Features:
    - Self-supervised learning capabilities
    - Register tokens for enhanced representation
    - SwiGLU activation functions
    - Flexible image size handling
    - Integration with TAO backbone framework
    - Support for activation checkpointing and layer freezing

    Attributes:
        patch_embed: Patch embedding layer
        cls_token: Classification token
        reg_token: Register tokens for enhanced representation
        pos_embed: Positional embeddings
        blocks: Transformer blocks
        norm: Final normalization layer
        head: Classification head
        num_reg_tokens: Number of register tokens

    References:
    - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

    Example:
        >>> model = DINOV2(
        ...     patch_size=14,
        ...     embed_dim=1024,
        ...     depth=24,
        ...     num_heads=16,
        ...     reg_tokens=4,
        ...     num_classes=1000
        ... )
        >>> x = torch.randn(1, 3, 518, 518)
        >>> output = model(x)  # Shape: (1, 1000)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DINOV2 model with enhanced configuration options.

        This constructor initializes a DINOv2 model with the specified architecture
        and provides additional functionality for backbone integration and feature extraction.

        Args:
            img_size (list, optional): Input image size. Defaults to [224, 224].
            patch_size (int, optional): Patch size for image division. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            global_pool (str, optional): Type of global pooling for final sequence. Defaults to 'token'.
            embed_dim (int, optional): Transformer embedding dimension. Defaults to 768.
            depth (int, optional): Depth of transformer. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): Enable bias for qkv projections if True. Defaults to True.
            init_values (float, optional): Layer-scale init values. Defaults to None.
            class_token (bool, optional): Use class token. Defaults to True.
            no_embed_class (bool, optional): Don't include position embeddings for class tokens. Defaults to False.
            reg_tokens (int, optional): Number of register tokens. Defaults to 0.
            pre_norm (bool, optional): Enable norm after embeddings. Defaults to False.
            final_norm (bool, optional): Enable norm after transformer blocks. Defaults to True.
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
            block_fn (callable, optional): Transformer block layer. Defaults to None.
            activation_checkpoint (bool, optional): Whether to use activation checkpointing. Defaults to False.
            freeze_at (list, optional): List of keys corresponding to the stages or layers to freeze.
                If None, no specific layers are frozen. If "all", the entire model is frozen.
                Defaults to None.
            freeze_norm (bool, optional): If True, all normalization layers in the backbone will be frozen.
                Defaults to False.
            **kwargs: Additional arguments passed to VisionTransformer constructor.

        Note:
            The DINOv2 model extends the standard Vision Transformer with register tokens
            and self-supervised learning capabilities. Register tokens provide additional
            representation capacity for improved feature learning.
        """
        super().__init__(*args, **kwargs)  # VisionTransformer initialization.

        # CAUTION: NVDINOv2 excludes `reg_token` from the `pos_embed`. This
        # differs from the TIMM implementation. We follow the NVDINOv2 design
        # here by applying the `pos_embed` only to the patch and class tokens.
        if self.num_reg_tokens > 0:
            embed_len = self.patch_embed.num_patches
            if self.has_class_token:
                embed_len += 1
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
            self.num_prefix_tokens -= self.num_reg_tokens

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.
        """
        if "register_tokens" in state_dict:
            state_dict["reg_token"] = state_dict.pop("register_tokens")
        return super().load_state_dict(state_dict, **kwargs)

    def _pos_embed(self, x):
        """Add positional embeddings to the input tokens.

        This method adds positional embeddings to the input tokens, including
        class tokens and register tokens if present. It handles different image
        sizes through interpolation.

        Args:
            x (torch.Tensor): Input tokens of shape (B, S, D) where B is batch size,
                S is sequence length, and D is embedding dimension.

        Returns:
            torch.Tensor: Tokens with positional embeddings added, shape (B, S', D)
                where S' includes class and register tokens.
        """
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x + self._interpolate_pos_encoding(x, w, h)
        # add register tokens
        if self.num_reg_tokens > 0:
            x = torch.cat((x, self.reg_token.expand(B, -1, -1)), dim=1)
        return self.pos_drop(x)

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the classification head.

        This method processes the input through all DINOv2 layers including
        patch embedding, positional embeddings, transformer blocks, and pooling,
        but stops before the final classification head. This is useful for
        feature extraction and transfer learning.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels, and H, W are height and width.

        Returns:
            torch.Tensor: Pre-logits features of shape (B, D) where D is the
                feature dimension.

        Example:
            >>> model = DINOV2()
            >>> x = torch.randn(1, 3, 518, 518)
            >>> features = model.forward_pre_logits(x)  # Shape: (1, 1024)
        """
        x = super().forward_features(x)
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x.flatten(1)

    def forward(self, x):
        """Complete forward pass through the DINOv2 model.

        This method performs the full forward pass including patch embedding,
        positional embeddings, transformer blocks, pooling, and the classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W) where B is batch
                size, C is number of channels (typically 3), and H, W are height
                and width (typically 518x518 for DINOv2).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) where
                num_classes is the number of output classes.

        Example:
            >>> model = DINOV2(num_classes=1000)
            >>> x = torch.randn(1, 3, 518, 518)
            >>> logits = model(x)  # Shape: (1, 1000)
        """
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def vit_large_patch14_dinov2_swiglu(**kwargs):
    """Create a DINOv2 ViT Large model with SwiGLU activation.

    This function creates a DINOv2 ViT Large model with the following specifications:
    - Patch size: 14x14 pixels
    - Embedding dimension: 1024
    - Depth: 24 transformer blocks
    - Number of heads: 16
    - Image size: 518x518
    - MLP ratio: 5472/1024 (approximately 5.34)
    - SwiGLU activation functions
    - Global pooling: token

    Args:
        **kwargs: Additional arguments passed to DINOV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        DINOV2: Configured DINOv2 ViT Large model.

    Example:
        >>> model = vit_large_patch14_dinov2_swiglu(num_classes=1000)
        >>> x = torch.randn(1, 3, 518, 518)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is designed for
        self-supervised learning of robust visual features.
    """
    return DINOV2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        img_size=518,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="token",
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_large_patch14_dinov2_swiglu_legacy(**kwargs):
    """Create a DINOv2 ViT Large model with SwiGLU activation (legacy version).

    This function creates a DINOv2 ViT Large model (legacy version) with the following specifications:
    - Patch size: 14x14 pixels
    - Embedding dimension: 1024
    - Depth: 24 transformer blocks
    - Number of heads: 16
    - MLP ratio: 5472/1024 (approximately 5.34)
    - SwiGLU activation functions
    - Legacy configuration without fixed image size

    Args:
        **kwargs: Additional arguments passed to DINOV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        DINOV2: Configured DINOv2 ViT Large model (legacy version).

    Example:
        >>> model = vit_large_patch14_dinov2_swiglu_legacy(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is the legacy version
        without fixed image size constraints.
    """
    return DINOV2(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def vit_giant_patch14_reg4_dinov2_swiglu(**kwargs):
    """Create a DINOv2 ViT Giant model with SwiGLU activation and register tokens.

    This function creates a DINOv2 ViT Giant model with the following specifications:
    - Patch size: 14x14 pixels
    - Embedding dimension: 1536
    - Depth: 40 transformer blocks
    - Number of heads: 24
    - Image size: 518x518
    - MLP ratio: 8192/1536 (approximately 5.33)
    - SwiGLU activation functions
    - Global pooling: token
    - Register tokens: 4

    Args:
        **kwargs: Additional arguments passed to DINOV2 constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        DINOV2: Configured DINOv2 ViT Giant model.

    Example:
        >>> model = vit_giant_patch14_reg4_dinov2_swiglu(num_classes=1000)
        >>> x = torch.randn(1, 3, 518, 518)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 1.1B parameters and is the largest DINOv2
        model with register tokens for enhanced representation learning.
    """
    return DINOV2(
        patch_size=14,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        init_values=1e-5,
        img_size=518,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=8192 / 1536,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="token",
        reg_tokens=4,
        **kwargs,
    )
