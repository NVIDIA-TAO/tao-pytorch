# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic 3D blocks for NVPanoptix3D model using WarpConvNet."""

import torch.nn as nn
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.normalizations import InstanceNorm
from warpconvnet.nn.modules.activations import ReLU
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.sparse_utils import add_voxels


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, sparse=False):
    """Create a 3×3 3D convolution layer (dense or sparse).

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Convolution stride. Use `stride>1` for downsampling.
        groups: Number of blocked connections from input channels to output
            channels (dense path only).
        dilation: Convolution dilation factor.
        sparse: If True, return a WarpConvNet sparse convolution; otherwise
            return a PyTorch dense convolution.

    Returns:
        A convolution module implementing a 3×3×3 kernel with `bias=False`.
    """
    if sparse:
        return SparseConv3d(
            in_planes, out_planes, kernel_size=3,
            stride=stride, dilation=dilation,
            bias=False
        )
    else:
        return nn.Conv3d(
            in_planes, out_planes, kernel_size=3,
            stride=stride, padding=dilation,
            groups=groups, bias=False,
            dilation=dilation
        )


class BasicBlock3D(nn.Module):
    """Basic 3D block for NVPanoptix3D model using WarpConvNet."""

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None, sparse=False
    ):
        """Initialize the residual block.

        Args:
            inplanes: Number of input channels.
            planes: Number of channels in the residual branch.
            stride: Stride for the first convolution. If `stride>1`, the
                residual branch is downsampled; the identity path must be
                downsampled to match (via `downsample`).
            downsample: Optional module applied to the identity path to match
                spatial resolution and/or channel count (e.g., a 1×1×1 conv).
            groups: Groups parameter for the dense convolution path. Must be 1
                for this block variant.
            base_width: Base width for the block. Must be 64 for this block
                variant.
            dilation: Dilation factor for 3×3 convolutions. Dilation > 1 is not
                supported for this block variant.
            norm_layer: Normalization layer constructor. If None, defaults to
                `nn.InstanceNorm3d` for dense, or WarpConvNet `InstanceNorm` for
                sparse.
            sparse: If True, construct layers using WarpConvNet sparse modules.
                For sparse residual addition semantics, use `SparseBasicBlock3D`.

        Raises:
            ValueError: If `groups != 1` or `base_width != 64`.
            NotImplementedError: If `dilation > 1`.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d if not sparse else InstanceNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, sparse=sparse)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) if not sparse else ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, sparse=sparse)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Apply the residual block.

        Args:
            x: Input feature map.
                - Dense path: a 5D tensor of shape (N, C, D, H, W).
                - Sparse path: a WarpConvNet sparse voxel tensor.

        Returns:
            Output feature map with the same tensor type as `x`.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SparseBasicBlock3D(BasicBlock3D):
    """Sparse 3D block for NVPanoptix3D model using WarpConvNet."""

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        """Initialize the sparse residual block.

        Args:
            inplanes: Number of input channels.
            planes: Number of channels in the residual branch.
            stride: Stride for the first sparse convolution.
            downsample: Optional module applied to the identity path (sparse).
            groups: Must be 1 (kept for API compatibility with `BasicBlock3D`).
            base_width: Must be 64 (kept for API compatibility with `BasicBlock3D`).
            dilation: Dilation factor for sparse 3×3 convolutions. Must be 1 for
                this block variant.
            norm_layer: Optional normalization layer constructor. If None,
                defaults to WarpConvNet `InstanceNorm`.
        """
        super().__init__(
            inplanes, planes, stride=stride,
            downsample=downsample, groups=groups,
            base_width=base_width, dilation=dilation,
            norm_layer=norm_layer, sparse=True
        )

    def forward(self, x):
        """Apply the sparse residual block.

        Args:
            x: WarpConvNet sparse voxel tensor.

        Returns:
            A sparse voxel tensor with residual + identity combined via
            `add_voxels(...)`, then activated with ReLU.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = add_voxels(out, identity)
        out = self.relu(out)

        return out
