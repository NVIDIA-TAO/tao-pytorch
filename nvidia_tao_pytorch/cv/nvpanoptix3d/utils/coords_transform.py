# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate transform utils for NVPanoptix3D."""

import torch
from typing import List
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.nn.modules.sparse_pool import SparseMaxPool


def fuse_sparse_tensors(tensor1: Voxels, tensor2: Voxels) -> Voxels:
    """
    Fuse two sparse voxel tensors by unioning coordinates and concatenating features.

    For each unique coordinate across ``tensor1`` and ``tensor2``:
    - If the coordinate exists in both tensors: output feature is
      ``[feat1, feat2]``.
    - If the coordinate exists only in ``tensor1``: output feature is
      ``[feat1, zeros]``.
    - If the coordinate exists only in ``tensor2``: output feature is
      ``[zeros, feat2]``.

    Args:
        tensor1: First :class:`~warpconvnet.geometry.types.voxels.Voxels`.
        tensor2: Second :class:`~warpconvnet.geometry.types.voxels.Voxels`.

    Returns:
        A new ``Voxels`` instance with the union of coordinates and concatenated
        feature channels.

    Notes:
        - This function does not require the coordinate sets to be identical.
        - The output ``tensor_stride`` is taken from ``tensor1``.
    """
    device = tensor1.device
    dtype = tensor1.feature_tensor.dtype

    # get coordinates and features
    coords1 = tensor1.batch_indexed_coordinates
    coords2 = tensor2.batch_indexed_coordinates
    feats1 = tensor1.feature_tensor
    feats2 = tensor2.feature_tensor

    feat_dim1, feat_dim2 = feats1.shape[1], feats2.shape[1]
    fused_feat_dim = feat_dim1 + feat_dim2

    # concatenate coordinates and create source tracking
    all_coords = torch.cat([coords1, coords2], dim=0)
    n_coords1 = coords1.shape[0]

    # convert each coordinate row to a view that can be uniqued
    coord_view = all_coords.view(all_coords.shape[0], -1)

    # use torch.unique with return_inverse to get mapping
    unique_coord_view, inverse_indices = torch.unique(coord_view, dim=0, return_inverse=True)
    unique_coords = unique_coord_view.view(-1, coords1.shape[1])
    n_unique = unique_coords.shape[0]

    # split inverse indices for each tensor
    inv_indices_1 = inverse_indices[:n_coords1]
    inv_indices_2 = inverse_indices[n_coords1:]

    # pre-allocate with zeros for automatic padding
    fused_features = torch.zeros(n_unique, fused_feat_dim, device=device, dtype=dtype)

    # tensor1 features go to positions [0:feat_dim1]
    fused_features[inv_indices_1, :feat_dim1] = feats1

    # tensor2 features go to positions [feat_dim1:feat_dim1+feat_dim2]
    fused_features[inv_indices_2, feat_dim1:] = feats2

    # Extract batch indices and spatial coordinates
    batch_indices = unique_coords[:, 0]
    spatial_coords = unique_coords[:, 1:]

    # Create offsets
    offsets = offsets_from_batch_index(batch_indices)

    # Create fused Voxels
    fused_tensor = Voxels(
        batched_coordinates=IntCoords(spatial_coords, offsets=offsets, tensor_stride=tensor1.tensor_stride),
        batched_features=CatFeatures(fused_features, offsets=offsets),
    )
    return fused_tensor


def generate_multiscale_feat3d(feat3d: Voxels) -> List[Voxels]:
    """
    Generate a multi-scale pyramid of sparse 3D features via sparse max-pooling.

    Args:
        feat3d: Input sparse voxel features.

    Returns:
        A list of :class:`~warpconvnet.geometry.types.voxels.Voxels` corresponding
        to progressively downsampled feature maps (currently 3 levels).

    Notes:
        The current implementation targets strides ``[2, 4, 8]``.
    """
    pooling_op = SparseMaxPool(kernel_size=3, stride=2)

    multi_scale_feat3d = []

    current_tensor = feat3d
    target_strides = [2, 4, 8]

    for _ in target_strides:
        # Pool features to get downsampled features
        pooled_tensor = pooling_op(current_tensor)
        coords = pooled_tensor.batch_indexed_coordinates
        stride = pooled_tensor.tensor_stride

        # rescale spatial coords to the original space
        coords = coords.clone()
        original_offsets = offsets_from_batch_index(coords[:, 0])
        features = pooled_tensor.feature_tensor
        multiscale_voxel = Voxels(
            batched_coordinates=IntCoords(coords[:, 1:], offsets=original_offsets, tensor_stride=stride),
            batched_features=CatFeatures(features, offsets=original_offsets),
        )

        multi_scale_feat3d.append(multiscale_voxel)
        current_tensor = pooled_tensor

    return multi_scale_feat3d
