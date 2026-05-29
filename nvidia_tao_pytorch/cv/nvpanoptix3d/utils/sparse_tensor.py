# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse tensor utils for NVPanoptix3D."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index


def _thicken_grid(grid, grid_dims, frustum_mask):
    """
    Thicken a grid by expanding each occupied voxel to its 3x3x3 neighborhood.
    This function is the same for both MinkowskiEngine and WarpConvNet as it works on dense tensors.

    Args:
        grid: Boolean tensor indicating occupied voxels
        grid_dims: Grid dimensions [x, y, z]
        frustum_mask: Boolean mask for frustum culling

    Returns:
        Thickened grid (boolean tensor)
    """
    device = frustum_mask.device
    # Create 3x3x3 neighborhood offsets
    offsets = torch.nonzero(torch.ones(3, 3, 3, device=device)).long() - 1  # Center at 0

    # Get occupied voxel locations
    locs_grid = grid.nonzero(as_tuple=False)

    if locs_grid.shape[0] == 0:
        return grid.clone()

    # Expand each location by 3x3x3 neighborhood
    locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
    locs += offsets
    locs = locs.view(-1, 3)

    # Filter out locations outside grid bounds
    grid_dims_tensor = torch.as_tensor(grid_dims, device=device)
    masks = ((locs >= 0) & (locs < grid_dims_tensor)).all(-1)
    locs = locs[masks]

    # Create thickened grid
    thicken = torch.zeros(grid_dims, dtype=torch.bool, device=device)
    if locs.shape[0] > 0:
        thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True

    # Apply frustum culling
    thicken = thicken & frustum_mask

    return thicken


def prepare_instance_masks_thicken(
    instances: torch.Tensor,
    semantic_mapping: Dict[int, int],
    distance_field: torch.Tensor,
    frustum_mask: torch.Tensor,
    iso_value: float = 1.0,
    truncation: float = 3.0,
    downsample_factor: int = 1
) -> Dict[int, Tuple[torch.Tensor, int]]:
    """
    Prepare thickened instance masks from distance field and semantic mapping.
    This function is the same for both MinkowskiEngine and WarpConvNet as it works on dense tensors.

    Args:
        instances: Instance segmentation (H, W, D)
        semantic_mapping: Mapping from instance ID to semantic class
        distance_field: TSDF distance field (H, W, D)
        frustum_mask: Frustum mask for valid regions (H, W, D)
        iso_value: Iso-surface value for extracting surface
        truncation: TSDF truncation value
        downsample_factor: Factor to downsample the grid

    Returns:
        Dictionary mapping instance_id to (mask, semantic_class)
    """
    # Validate downsample factor
    if not isinstance(downsample_factor, int) or 256 % downsample_factor != 0:
        raise ValueError("downsample_factor must be an integer divisor of 256")

    grid_dims = [256, 256, 256]
    need_rescale = downsample_factor != 1

    if need_rescale:
        grid_dims = (torch.as_tensor(grid_dims) // downsample_factor).tolist()
        frustum_mask = F.interpolate(
            frustum_mask[None, None].float(),
            size=grid_dims,
            mode="nearest"
        ).squeeze(0, 1).bool()

    instance_information = {}

    for instance_id, semantic_class in semantic_mapping.items():
        # Extract instance mask
        instance_mask = (instances == instance_id)

        # Create instance-specific distance field
        instance_distance_field = torch.full_like(
            instance_mask,
            dtype=torch.float,
            fill_value=truncation
        )
        instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]

        # Extract surface using iso-value
        instance_distance_field_masked = instance_distance_field.abs() < iso_value

        # Downsample if needed
        if need_rescale:
            instance_distance_field_masked = F.max_pool3d(
                instance_distance_field_masked[None, None].float(),
                kernel_size=downsample_factor + 1,
                stride=downsample_factor,
                padding=1
            ).squeeze(0, 1).bool()

        # Thicken the grid
        instance_grid = _thicken_grid(
            instance_distance_field_masked,
            grid_dims,
            frustum_mask
        )

        # Move to CPU for storage
        instance_grid = instance_grid.to(torch.device("cpu"), non_blocking=True)
        instance_information[instance_id] = (instance_grid, semantic_class)

    return instance_information


def mask_invalid_sparse_voxels(
    voxels: Voxels,
    mask: Optional[torch.Tensor] = None,
    frustum_dim: list = [256, 256, 256]
) -> Voxels:
    """
    Mask out voxels which are outside of the grid.
    WarpConvNet version of the MinkowskiEngine mask_invalid_sparse_voxels function.

    Args:
        voxels: Input Voxels object
        mask: Optional additional boolean mask
        frustum_dim: Frustum dimensions [x, y, z]

    Returns:
        Pruned Voxels with invalid voxels removed
    """
    # Get batch indexed coordinates
    coords = voxels.batch_indexed_coordinates  # (N, 4) -> [batch, x, y, z]

    # Create validity mask - check if coordinates are within bounds
    valid_mask = (coords[:, 1] < frustum_dim[0] - 1) & (coords[:, 1] >= 0) & \
                 (coords[:, 2] < frustum_dim[1] - 1) & (coords[:, 2] >= 0) & \
                 (coords[:, 3] < frustum_dim[2] - 1) & (coords[:, 3] >= 0)

    # Apply additional mask if provided
    if mask is not None:
        valid_mask = valid_mask & mask

    num_valid_coordinates = valid_mask.sum().item()

    # Handle empty case
    if num_valid_coordinates == 0:
        # Return empty Voxels
        # WarpConvNet requires at least 2 elements: [0, 0] for empty batch
        return Voxels(
            batched_coordinates=IntCoords(
                torch.empty((0, 3), dtype=torch.int32, device=voxels.device),
                offsets=torch.tensor([0, 0], dtype=torch.int64, device=voxels.device),
                tensor_stride=voxels.tensor_stride
            ),
            batched_features=CatFeatures(
                torch.empty(
                    (0, voxels.num_channels), dtype=voxels.feature_tensor.dtype, device=voxels.device
                ),
                offsets=torch.tensor([0, 0], dtype=torch.int64, device=voxels.device)
            )
        )

    num_masked_voxels = coords.size(0) - num_valid_coordinates
    grids_needs_to_be_pruned = num_masked_voxels > 0

    # Only prune if there are invalid voxels
    if grids_needs_to_be_pruned:
        # Filter coordinates and features
        filtered_coords = coords[valid_mask]
        filtered_features = voxels.feature_tensor[valid_mask]

        # Extract batch indices and spatial coordinates - ensure int type for bincount
        batch_indices = filtered_coords[:, 0].int()
        spatial_coords = filtered_coords[:, 1:]

        # Create new offsets - handle empty case
        # WarpConvNet requires at least 2 elements: [0, 0] for empty
        if batch_indices.numel() == 0:
            offsets = torch.tensor([0, 0], dtype=torch.int64, device=batch_indices.device)
        else:
            offsets = offsets_from_batch_index(batch_indices)

        # Create pruned Voxels
        voxels = Voxels(
            batched_coordinates=IntCoords(
                spatial_coords,
                offsets=offsets,
                tensor_stride=voxels.tensor_stride
            ),
            batched_features=CatFeatures(
                filtered_features,
                offsets=offsets
            )
        )

    return voxels


def prune_voxels(voxels: Voxels, mask: torch.Tensor) -> Voxels:
    """
    Prune voxels based on a boolean mask.
    Equivalent to MinkowskiEngine's MinkowskiPruning.

    MinkowskiPruning behavior:
    - Takes a SparseTensor and a boolean mask of length N (number of coordinates)
    - Mask values of True indicate coordinates to KEEP
    - Returns a new SparseTensor with only the masked coordinates

    Args:
        voxels: Input Voxels
        mask: Boolean mask of length N indicating which voxels to keep (True = keep, False = remove)

    Returns:
        Pruned Voxels with only the coordinates where mask is True

    Raises:
        AssertionError: If mask length doesn't match number of voxels
    """
    # Validate mask length matches number of voxels (same as MinkowskiPruning requirement)
    num_voxels = voxels.coordinate_tensor.shape[0]
    assert mask.shape[0] == num_voxels, \
        f"Mask length ({mask.shape[0]}) must match number of voxels ({num_voxels})"

    # Get coordinates and features using boolean indexing
    coords = voxels.coordinate_tensor[mask]
    feats = voxels.feature_tensor[mask]

    # Get batch indices - ensure int type for bincount
    batch_indices = voxels.batch_indexed_coordinates[:, 0][mask].int()

    # Create offsets - handle empty case
    # WarpConvNet requires at least 2 elements: [0, 0] for empty batch
    if batch_indices.numel() == 0:
        offsets = torch.tensor([0, 0], dtype=torch.int64, device=batch_indices.device)
    else:
        offsets = offsets_from_batch_index(batch_indices)

    # Create new Voxels with pruned coordinates and features
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets, tensor_stride=voxels.tensor_stride),
        batched_features=CatFeatures(feats, offsets=offsets),
    )


def sigmoid_voxels(voxels: Voxels) -> Voxels:
    """
    Apply sigmoid to voxel features.
    Similar to MinkowskiEngine's MinkowskiSigmoid.

    Args:
        voxels: Input Voxels

    Returns:
        Voxels with sigmoid applied to features
    """
    return voxels.replace(batched_features=torch.sigmoid(voxels.feature_tensor))


def get_voxel_coordinates_at_batch(voxels: Voxels, batch_idx: int) -> torch.Tensor:
    """
    Get coordinates for a specific batch index.
    Similar to MinkowskiEngine's coordinates_at.

    Args:
        voxels: Input Voxels
        batch_idx: Batch index

    Returns:
        Coordinates for the specified batch (N, 3)
    """
    batch_mask = voxels.batch_indexed_coordinates[:, 0] == batch_idx
    return voxels.coordinate_tensor[batch_mask]


def get_voxel_features_at_batch(voxels: Voxels, batch_idx: int) -> torch.Tensor:
    """
    Get features for a specific batch index.
    Similar to MinkowskiEngine's features_at.

    Args:
        voxels: Input Voxels
        batch_idx: Batch index

    Returns:
        Features for the specified batch (N, C)
    """
    batch_mask = voxels.batch_indexed_coordinates[:, 0] == batch_idx
    return voxels.feature_tensor[batch_mask]


def sparse_collate(
    coords_list: list,
    feats_list: list,
    stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate coordinates and features from multiple samples.
    Similar to MinkowskiEngine's sparse_collate.

    Args:
        coords_list: List of coordinate tensors
        feats_list: List of feature tensors
        stride: Tensor stride

    Returns:
        Tuple of (batched_coordinates, batched_features)
        batched_coordinates: (N, 4) with [batch_idx, x, y, z]
        batched_features: (N, C)
    """
    # Create batched coordinates
    batched_coords_list = []
    valid_feats_list = []

    for batch_idx, (coords, feats) in enumerate(zip(coords_list, feats_list)):
        if coords.shape[0] > 0:
            # IMPORTANT: Use int32 for batch indices to avoid bincount error
            batch_col = torch.full(
                (coords.shape[0], 1),
                batch_idx,
                device=coords.device,
                dtype=torch.int32
            )
            # Ensure coordinates are also int32
            batch_coords = torch.cat([batch_col, coords.int()], dim=1)
            batched_coords_list.append(batch_coords)
            valid_feats_list.append(feats)

    if len(batched_coords_list) == 0:
        # Return empty tensors
        device = coords_list[0].device if len(coords_list) > 0 else torch.device("cpu")
        feat_dim = feats_list[0].shape[-1] if len(feats_list) > 0 and feats_list[0].shape[0] > 0 else 1
        return (
            torch.empty((0, 4), dtype=torch.int32, device=device),
            torch.empty((0, feat_dim), device=device)
        )

    all_coords = torch.cat(batched_coords_list, dim=0)
    all_feats = torch.cat(valid_feats_list, dim=0)

    return all_coords, all_feats
