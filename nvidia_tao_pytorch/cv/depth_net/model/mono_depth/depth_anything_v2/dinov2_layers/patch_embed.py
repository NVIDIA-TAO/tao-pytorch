# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchEmbed Modules."""

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    """
    This utility function ensures that the input is a 2-tuple, which is commonly
    used for specifying 2D dimensions like (height, width) in vision models.

    Args:
        x (int or tuple): Input value to convert to 2-tuple.
            If x is already a tuple, it must have length 2.
            If x is an int, it will be converted to (x, x).

    Returns:
        tuple: A 2-tuple (x, x) if x is an int, or x if x is already a 2-tuple.

    Raises:
        AssertionError: If x is a tuple but doesn't have length 2, or if x is not an int.

    Example:
        >>> make_2tuple(16)
        (16, 16)
        >>> make_2tuple((224, 224))
        (224, 224)
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    This module converts 2D images into a sequence of patch embeddings, which is
    the first step in Vision Transformer architectures. It divides the input image
    into non-overlapping patches and projects each patch to a high-dimensional
    embedding space.

    The module performs the transformation: (B, C, H, W) -> (B, N, D) where:
    - B is batch size
    - C is number of input channels
    - H, W are image height and width
    - N is number of patches (H//patch_size * W//patch_size)
    - D is embedding dimension

    Attributes:
        img_size (tuple): Input image size (height, width).
        patch_size (tuple): Patch size (height, width).
        patches_resolution (tuple): Number of patches in each dimension.
        num_patches (int): Total number of patches.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        flatten_embedding (bool): Whether to flatten the embedding.
        proj (nn.Conv2d): Convolutional projection layer.
        norm (nn.Module): Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """Initialize the PatchEmbed module.

        Args:
            img_size (Union[int, Tuple[int, int]], optional): Input image size.
                Can be a single integer (square image) or a tuple (height, width).
                Defaults to 224.
            patch_size (Union[int, Tuple[int, int]], optional): Size of each patch.
                Can be a single integer (square patches) or a tuple (height, width).
                Defaults to 16.
            in_chans (int, optional): Number of input image channels.
                Defaults to 3 (RGB).
            embed_dim (int, optional): Number of output embedding dimensions.
                Defaults to 768.
            norm_layer (Optional[Callable], optional): Normalization layer to apply
                after projection. If None, no normalization is applied.
                Defaults to None.
            flatten_embedding (bool, optional): Whether to flatten the patch embeddings
                into a sequence. If False, maintains spatial structure.
                Defaults to True.

        Note:
            - Image dimensions must be divisible by patch dimensions
            - The number of patches is (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        """
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the patch embedding module.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) where:
                - B is batch size
                - C is number of channels (must match in_chans)
                - H, W are image height and width (must be divisible by patch_size)

        Returns:
            Tensor: Patch embeddings of shape:
                - (B, N, D) if flatten_embedding=True
                - (B, H//patch_size, W//patch_size, D) if flatten_embedding=False
                where N = (H//patch_size) * (W//patch_size)

        Raises:
            AssertionError: If image dimensions are not divisible by patch dimensions.

        Note:
            The forward pass consists of:
            1. Convolutional projection to create patch embeddings
            2. Flattening and transposition to sequence format
            3. Optional normalization
            4. Optional reshaping to maintain spatial structure
        """
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """
        This method calculates the computational complexity of the patch embedding
        operation, which is useful for model analysis and optimization.

        Returns:
            float: Number of floating-point operations required for the forward pass.
                This includes:
                - Convolutional projection operations
                - Normalization operations (if applicable)

        Note:
            The FLOPs calculation assumes:
            - Convolution: Ho * Wo * embed_dim * in_chans * (patch_size[0] * patch_size[1])
            - Normalization: Ho * Wo * embed_dim (if norm is applied)
            where Ho, Wo are the output spatial dimensions
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
