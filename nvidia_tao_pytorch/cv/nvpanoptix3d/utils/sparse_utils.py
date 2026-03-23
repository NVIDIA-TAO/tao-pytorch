# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for NVPanoptix3D sparse tensors."""

import collections.abc
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.modules.prune import SparsePrune


def _is_empty_sparse(voxels: Voxels) -> bool:
    """Check whether a sparse voxel tensor has any active coordinates.

    Args:
        voxels: Sparse voxel tensor whose active features should be inspected.

    Returns:
        ``True`` when the sparse tensor contains no active feature rows and
        therefore no active coordinates; ``False`` otherwise.
    """
    return voxels.feature_tensor.numel() == 0 or voxels.feature_tensor.shape[0] == 0


def sparse_cat_union(a: Voxels, b: Voxels) -> Voxels:
    """Union coordinates and concatenate features from two sparse voxel tensors.

    Args:
        a: First :class:`~warpconvnet.geometry.types.voxels.Voxels` object.
        b: Second :class:`~warpconvnet.geometry.types.voxels.Voxels` object.

    Returns:
        A new ``Voxels`` instance with unioned coordinates and concatenated
        features. When one input is empty, the result still preserves the full
        concatenated channel dimension by zero-padding the missing side instead
        of returning the non-empty tensor unchanged.

    Raises:
        AssertionError: If ``tensor_stride`` or ``batch_size`` do not match.
    """
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"
    assert a.batch_size == b.batch_size, "different batch size"

    coords_a = a.batch_indexed_coordinates
    coords_b = b.batch_indexed_coordinates
    feats_a = a.feature_tensor
    feats_b = b.feature_tensor

    all_coords = torch.cat([coords_a, coords_b], dim=0)
    unique_coords, inverse_indices = torch.unique(
        all_coords, dim=0, return_inverse=True, sorted=True
    )

    n_a = coords_a.shape[0]
    inv_a = inverse_indices[:n_a]
    inv_b = inverse_indices[n_a:]

    feat_dim_a = feats_a.shape[1]
    feat_dim_b = feats_b.shape[1]
    fused_features = feats_a.new_zeros(
        (unique_coords.shape[0], feat_dim_a + feat_dim_b)
    )
    fused_features[inv_a, :feat_dim_a] = feats_a
    fused_features[inv_b, feat_dim_a:] = feats_b

    batch_indices = unique_coords[:, 0].to(torch.int64)
    batch_size = max(a.batch_size, b.batch_size)
    if batch_indices.numel() > 0:
        batch_size = max(batch_size, int(batch_indices.max().item() + 1))

    counts = torch.bincount(batch_indices.cpu(), minlength=batch_size)
    offsets = torch.zeros(
        batch_size + 1,
        dtype=a.offsets.dtype,
        device=a.offsets.device,
    )
    offsets[1:] = counts.to(dtype=offsets.dtype).cumsum(dim=0)

    spatial_coords = unique_coords[:, 1:].to(dtype=a.coordinate_tensor.dtype)

    coord_kwargs = {
        "offsets": offsets,
        "tensor_stride": a.tensor_stride,
    }
    voxel_size = getattr(a.batched_coordinates, "voxel_size", None)
    if voxel_size is not None:
        coord_kwargs["voxel_size"] = voxel_size
    voxel_origin = getattr(a.batched_coordinates, "voxel_origin", None)
    if voxel_origin is not None:
        coord_kwargs["voxel_origin"] = voxel_origin

    new_coords = IntCoords(spatial_coords, **coord_kwargs)
    new_feats = CatFeatures(fused_features, offsets=offsets)
    return a.replace(
        batched_coordinates=new_coords,
        batched_features=new_feats,
    )


def to_dense(
    voxels: Voxels,
    shape: Optional[torch.Size] = None,
    min_coordinate: Optional[torch.IntTensor] = None,
    contract_stride: bool = False,
    default_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.IntTensor, torch.IntTensor]:
    """
    Convert NVPanoptix3D Voxels to a dense tensor.

    Fully aligned with MinkowskiEngine's SparseTensor.dense() behavior.

    Args:
        voxels: Input Voxels sparse tensor
        shape (torch.Size, optional): The size of the output tensor.
        min_coordinate (torch.IntTensor, optional): The min coordinates of the
            output sparse tensor. Must be divisible by the current tensor_stride.
            If 0 is given, it will use the origin for the min coordinate.
        contract_stride (bool, optional): The output coordinates will be divided
            by the tensor stride to make features spatially contiguous. True by default.
        default_value (float, optional): Value to fill empty coordinates. Default 0.0.

    Returns:
        tensor (torch.Tensor): the torch tensor with size `[Batch Dim, Feature Dim,
            Spatial Dim..., Spatial Dim]`. The coordinate of each feature can be
            accessed via `min_coordinate + tensor_stride * [the coordinate of the dense tensor]`.
        min_coordinate (torch.IntTensor): the D-dimensional vector defining the
            minimum coordinate of the output tensor.
        tensor_stride (torch.IntTensor): the D-dimensional vector defining the
            stride between tensor elements.
    """
    num_spatial_dims = voxels.num_spatial_dims
    features = voxels.feature_tensor
    device = features.device

    # Validate and normalize shape input
    if shape is not None:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        if len(shape) != num_spatial_dims + 2:
            raise ValueError(
                f"shape length {len(shape)} must be {num_spatial_dims + 2} "
                "(batch + channel + spatial)"
            )
        # Correct channel dimension if it doesn't match
        if shape[1] != voxels.num_channels:
            shape = torch.Size([shape[0], voxels.num_channels, *shape[2:]])

    # Normalize tensor_stride to tuple
    tensor_stride = voxels.tensor_stride
    if tensor_stride is None:
        tensor_stride = (1,) * num_spatial_dims
    elif isinstance(tensor_stride, int):
        tensor_stride = (tensor_stride,) * num_spatial_dims

    # Handle empty tensor case
    if len(voxels) == 0:
        if shape is None:
            raise ValueError("shape is required to densify an empty Voxels")
        dense = torch.full(
            shape,
            fill_value=default_value,
            dtype=features.dtype,
            device=device,
        )
        return (
            dense,
            torch.zeros(num_spatial_dims, dtype=torch.int32, device=device),
            torch.IntTensor(tensor_stride),
        )

    # Validate min_coordinate input
    is_valid_min_coordinate = isinstance(min_coordinate, torch.IntTensor) or (
        isinstance(min_coordinate, int) and min_coordinate == 0
    )
    if min_coordinate is not None and not is_valid_min_coordinate:
        raise TypeError("min_coordinate must be torch.IntTensor or integer 0")
    if isinstance(min_coordinate, torch.IntTensor):
        if min_coordinate.numel() != num_spatial_dims:
            raise ValueError(
                f"min_coordinate size {min_coordinate.numel()} must match "
                f"spatial dims {num_spatial_dims}"
            )

    # Use int tensor for all stride operations (matches MinkowskiEngine)
    tensor_stride_tensor = torch.IntTensor(tensor_stride).to(device)

    # Extract batch indices and spatial coordinates
    batch_coords = voxels.batch_indexed_coordinates.to(device=device)
    batch_indices = batch_coords[:, 0]

    # Handle min_coordinate: compute or use provided value
    # This section matches MinkowskiEngine's logic exactly
    if min_coordinate is None:
        # Compute min from batch_indexed_coordinates, then extract spatial part
        min_coordinate_from_data, _ = batch_coords.min(0, keepdim=True)
        min_coordinate_tensor = min_coordinate_from_data[:, 1:]  # Exclude batch dimension
        if not torch.all(min_coordinate_tensor >= 0):
            raise ValueError(
                f"Coordinate has a negative value: {min_coordinate_tensor}. "
                "Please provide min_coordinate argument"
            )
        min_coordinate_tensor = min_coordinate_tensor.to(dtype=torch.int32)
        # Use original spatial coordinates without subtraction
        coords = voxels.coordinate_tensor.to(device=device)
    elif isinstance(min_coordinate, int) and min_coordinate == 0:
        # Use origin as min coordinate
        min_coordinate_tensor = torch.zeros(
            (1, num_spatial_dims), dtype=torch.int32, device=device
        )
        # Use original spatial coordinates without subtraction
        coords = voxels.coordinate_tensor.to(device=device)
    else:
        # Use provided min_coordinate and subtract from spatial coordinates
        min_coordinate_tensor = min_coordinate.to(device=device)
        if min_coordinate_tensor.ndim == 1:
            min_coordinate_tensor = min_coordinate_tensor.unsqueeze(0)
        min_coordinate_tensor = min_coordinate_tensor.to(dtype=torch.int32)
        # Subtract min_coordinate from spatial coordinates
        coords = voxels.coordinate_tensor.to(device=device) - \
            min_coordinate_tensor.to(dtype=voxels.coordinate_tensor.dtype)

    # Validate that min_coordinate is divisible by stride
    assert (
        min_coordinate_tensor % tensor_stride_tensor
    ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

    # Ensure coords is 2D
    if coords.ndim == 1:
        coords = coords.unsqueeze(1)

    # Contract stride: divide coordinates by stride for spatial contiguity
    if contract_stride:
        coords = torch.div(coords, tensor_stride_tensor, rounding_mode="floor")

    # Determine output shape if not provided
    nchannels = voxels.num_channels
    if shape is None:
        size = coords.max(0)[0] + 1
        batch_count = int(batch_indices.max().item()) + 1 if batch_indices.numel() > 0 else 1
        shape = torch.Size([batch_count, nchannels, *size.cpu().tolist()])

    # Initialize dense tensor with default value
    dense_F = torch.full(
        shape,
        fill_value=default_value,
        dtype=features.dtype,
        device=features.device,
    )

    # Populate dense tensor with sparse features
    # This matches MinkowskiEngine's indexing behavior
    tcoords = coords.t().long()
    batch_indices_long = batch_indices.long()

    # Build index tuple for efficient indexing
    index = [batch_indices_long, slice(None)]
    for i in range(len(tcoords)):
        index.append(tcoords[i])
    dense_F[tuple(index)] = features

    tensor_stride_out = torch.IntTensor(tensor_stride)
    return dense_F, min_coordinate_tensor, tensor_stride_out


def get_voxel_coordinates_at_batch(voxels: Voxels, batch_idx: int) -> torch.Tensor:
    """
    Get coordinates for a specific batch index.
    Similar to MinkowskiEngine's coordinates_at.

    Args:
        voxels: Input Voxels
        batch_idx: Batch index

    Returns:
        Coordinates for the specified batch
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
        Features for the specified batch
    """
    batch_mask = voxels.batch_indexed_coordinates[:, 0] == batch_idx
    return voxels.feature_tensor[batch_mask]


def sparse_collate(coords, feats, labels=None, dtype=torch.int32, device=None):
    """Collate sparse coordinates/features (and optional labels) into batched tensors.


    Args:
        coords: Sequence of coordinate arrays/tensors, one per sample, with shape
            ``(N_i, D)`` where D is the number of spatial dimensions.
        feats: Sequence of feature arrays/tensors, one per sample, with shape
            ``(N_i, C)``.
        labels: Optional sequence of labels aligned with ``coords``/``feats``.
            If provided, labels are collated and returned as the third output.
        dtype: Dtype for the output batched coordinates. Only ``torch.int32`` and
            ``torch.float32`` are supported.
        device: Optional device for the returned tensors. If None, inferred from
            the first coordinate tensor when available, else defaults to CPU.

    Returns:
        If ``labels`` is None:
            ``(batched_coords, batched_feats)``

        If ``labels`` is provided:
            ``(batched_coords, batched_feats, batched_labels)``

        Where ``batched_coords`` has shape ``(sum_i N_i, D+1)`` and the leading
        column contains the batch index.

    Raises:
        TypeError / ValueError: If inputs are not sequences or sizes are inconsistent.
    """
    if not isinstance(coords, collections.abc.Sequence):
        raise TypeError("coords must be a sequence")
    if not isinstance(feats, collections.abc.Sequence):
        raise TypeError("feats must be a sequence")

    use_label = labels is not None
    if use_label and not isinstance(labels, collections.abc.Sequence):
        raise TypeError("labels must be a sequence when provided")

    if len(coords) == 0:
        if use_label:
            raise ValueError("labels provided but coords/feats are empty")
        if device is None:
            device = "cpu"
        return (
            torch.empty((0, 4), dtype=dtype, device=device),
            torch.empty((0, 0), device=device),
        )

    coord_dims = {coord.shape[1] for coord in coords}
    if len(coord_dims) != 1:
        raise ValueError(f"Coordinate dimension mismatch: {coord_dims}")
    dim = coord_dims.pop()

    if device is None:
        if isinstance(coords[0], torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"

    if dtype not in (torch.int32, torch.float32):
        raise ValueError("dtype must be torch.int32 or torch.float32 for coordinates")

    coord_lengths = torch.tensor([len(coord) for coord in coords], dtype=torch.long)
    feat_lengths = torch.tensor([len(feat) for feat in feats], dtype=torch.long)
    total_coords = int(coord_lengths.sum().item())
    total_feats = int(feat_lengths.sum().item())
    if total_coords != total_feats:
        raise ValueError(
            f"Coordinate length {total_coords} != feature length {total_feats}"
        )

    batched_coords = torch.zeros(
        (total_coords, dim + 1), dtype=dtype, device=device
    )
    feats_batch: List[torch.Tensor] = []
    labels_batch: List[torch.Tensor] = []

    cursor = 0
    for batch_idx, (coord, feat) in enumerate(zip(coords, feats)):
        if isinstance(coord, torch.Tensor):
            coord_tensor = coord
        else:
            coord_tensor = torch.from_numpy(coord)

        if dtype == torch.int32 and coord_tensor.dtype in (torch.float32, torch.float64):
            coord_tensor = coord_tensor.floor()
        coord_tensor = coord_tensor.to(device=device, dtype=dtype)

        if isinstance(feat, torch.Tensor):
            feat_tensor = feat.to(device=device)
        else:
            feat_tensor = torch.from_numpy(feat).to(device=device)

        count = coord_tensor.shape[0]
        if count == 0:
            continue

        batched_coords[cursor: cursor + count, 1:] = coord_tensor
        batched_coords[cursor: cursor + count, 0] = batch_idx

        feats_batch.append(feat_tensor)

        if use_label:
            label_entry = labels[batch_idx]
            if isinstance(label_entry, torch.Tensor):
                labels_batch.append(label_entry.to(device=device))
            else:
                labels_batch.append(torch.from_numpy(label_entry).to(device=device))

        cursor += count

    feats_batch_tensor = torch.cat(feats_batch, dim=0) if feats_batch else torch.empty(
        (0, 0), device=device
    )

    if use_label:
        if not labels_batch:
            labels_batch_tensor = torch.empty((0,), device=device)
        elif isinstance(labels_batch[0], torch.Tensor) and labels_batch[0].ndim > 0:
            labels_batch_tensor = torch.cat(labels_batch, dim=0)
        else:
            labels_batch_tensor = torch.tensor(labels_batch, device=device)
        return batched_coords, feats_batch_tensor, labels_batch_tensor

    return batched_coords, feats_batch_tensor


def _thicken_grid(grid, grid_dims, frustum_mask):
    """Dilate a boolean occupancy grid by a 3x3x3 neighborhood and frustum-mask it.

    Args:
        grid: Boolean tensor of shape ``grid_dims``.
        grid_dims: Spatial dimensions (x, y, z) used for bounds checking.
        frustum_mask: Boolean tensor of the same shape used to cull voxels outside
            the camera frustum.

    Returns:
        A boolean tensor of shape ``grid_dims`` representing the thickened grid.
    """
    device = frustum_mask.device
    offsets = torch.nonzero(torch.ones(3, 3, 3, device=device)).long()
    locs_grid = grid.nonzero(as_tuple=False)
    locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
    locs += offsets
    locs = locs.view(-1, 3)
    masks = ((locs >= 0) & (locs < torch.as_tensor(grid_dims, device=device))).all(-1)
    locs = locs[masks]

    thicken = torch.zeros(grid_dims, dtype=torch.bool, device=device)
    thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True
    # frustum culling
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
    """Build thickened per-instance voxel masks from instance IDs and a distance field.

    Args:
        instances: Tensor of instance IDs (typically a dense 3D volume).
        semantic_mapping: Dict mapping ``instance_id -> semantic_class``.
        distance_field: Signed distance field tensor aligned with ``instances``.
        frustum_mask: Boolean frustum mask tensor used to cull invalid voxels.
        iso_value: Iso-surface threshold for selecting near-surface voxels.
        truncation: Fill value used outside an instance when constructing its
            per-instance distance field.
        downsample_factor: Integer downsample factor. Must evenly divide 256.

    Returns:
        Dict mapping ``instance_id`` to a tuple ``(instance_mask_grid, semantic_class)``,
        where ``instance_mask_grid`` is a boolean tensor on CPU.
    """
    # check if downsample factor is valid
    assert isinstance(downsample_factor, int) and 256 % downsample_factor == 0
    grid_dims = [256, 256, 256]
    need_rescale = downsample_factor != 1
    if need_rescale:
        grid_dims = (torch.as_tensor(grid_dims) // downsample_factor).tolist()
        frustum_mask = F.interpolate(frustum_mask[None, None].float(),
                                     size=grid_dims, mode="nearest").squeeze(0, 1).bool()

    instance_information = {}

    for instance_id, semantic_class in semantic_mapping.items():
        instance_mask: torch.Tensor = (instances == instance_id)
        instance_distance_field = torch.full_like(
            instance_mask,
            dtype=torch.float,
            fill_value=truncation,
        )
        instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
        instance_distance_field_masked = instance_distance_field.abs() < iso_value

        if need_rescale:
            instance_distance_field_masked = F.max_pool3d(
                instance_distance_field_masked[None, None].float(),
                kernel_size=downsample_factor + 1,
                stride=downsample_factor,
                padding=1,
            ).squeeze(0, 1).bool()

        # instance_grid = instance_grid & frustum_mask
        instance_grid = _thicken_grid(
            instance_distance_field_masked,
            grid_dims,
            frustum_mask,
        )
        instance_grid: torch.Tensor = instance_grid.to(torch.device("cpu"), non_blocking=True)
        instance_information[instance_id] = instance_grid, semantic_class

    return instance_information


def mask_invalid_sparse_voxels(
    voxels: Voxels,
    mask: Optional[torch.Tensor] = None,
    frustum_dim=[256, 256, 256],
) -> Voxels:
    """Prune sparse voxels that fall outside frustum bounds (and optional mask).

    Args:
        voxels: Input :class:`~warpconvnet.geometry.types.voxels.Voxels`.
        mask: Optional boolean mask aligned with ``voxels.batch_indexed_coordinates``.
        frustum_dim: Spatial bounds as ``[X, Y, Z]`` in voxel coordinates.

    Returns:
        A new ``Voxels`` instance containing only valid coordinates/features.

    Notes:
        If all coordinates are invalid, an empty ``Voxels`` with preserved metadata
        (tensor_stride/offsets/voxel_size/origin) is returned.
    """
    coords = voxels.batch_indexed_coordinates

    valid_mask = (
        (coords[:, 1] < frustum_dim[0] - 1) &
        (coords[:, 1] >= 0) &
        (coords[:, 2] < frustum_dim[1] - 1) &
        (coords[:, 2] >= 0) &
        (coords[:, 3] < frustum_dim[2] - 1) &
        (coords[:, 3] >= 0)
    )
    if mask is not None:
        valid_mask = valid_mask & mask

    num_valid_coordinates = valid_mask.sum().item()
    # handle empty case
    if num_valid_coordinates == 0:
        offsets = torch.zeros_like(voxels.offsets)
        empty_coords = torch.empty(
            (0, voxels.num_spatial_dims),
            dtype=voxels.coordinate_tensor.dtype,
            device=voxels.coordinate_tensor.device,
        )
        empty_feats = torch.empty(
            (0, voxels.num_channels),
            dtype=voxels.feature_tensor.dtype,
            device=voxels.feature_tensor.device,
        )
        coord_kwargs = {
            "offsets": offsets,
            "tensor_stride": voxels.tensor_stride,
        }
        voxel_size = getattr(voxels.batched_coordinates, "voxel_size", None)
        if voxel_size is not None:
            coord_kwargs["voxel_size"] = voxel_size
        voxel_origin = getattr(voxels.batched_coordinates, "voxel_origin", None)
        if voxel_origin is not None:
            coord_kwargs["voxel_origin"] = voxel_origin
        new_coords = IntCoords(empty_coords, **coord_kwargs)
        new_feats = CatFeatures(empty_feats, offsets=offsets)
        return voxels.replace(
            batched_coordinates=new_coords,
            batched_features=new_feats,
        )

    num_masked_voxels = coords.size(0) - num_valid_coordinates
    if num_masked_voxels > 0:
        voxels = SparsePrune()(voxels, valid_mask)

    return voxels


def add_voxels(a: Voxels, b: Voxels) -> Voxels:
    """
    Add two Voxels objects, fully aligned with MinkowskiEngine's addition behavior.

    This function implements the same logic as MinkowskiEngine's MinkowskiUnion:
    - Creates a union of coordinates from both input tensors
    - Sums features at overlapping coordinates
    - Preserves features at non-overlapping coordinates

    Args:
        a: First Voxels object
        b: Second Voxels object

    Returns:
        New Voxels with unified coordinates and summed features
    """
    # Handle empty tensors
    if a.coordinate_tensor.shape[0] == 0:
        return b
    if b.coordinate_tensor.shape[0] == 0:
        return a

    # Validate inputs match MinkowskiEngine requirements
    assert (
        a.tensor_stride == b.tensor_stride
    ), f"Tensor stride mismatch: {a.tensor_stride} != {b.tensor_stride}"

    assert (
        a.num_channels == b.num_channels
    ), f"Feature dimension mismatch: {a.num_channels} != {b.num_channels}"

    assert (
        a.batch_size == b.batch_size
    ), f"Batch size mismatch: {a.batch_size} != {b.batch_size}"

    # Extract batch-indexed coordinates and features
    coords_a = a.batch_indexed_coordinates
    coords_b = b.batch_indexed_coordinates
    feats_a = a.feature_tensor
    feats_b = b.feature_tensor

    # Ensure coordinates are on the same device for concatenation
    device = coords_a.device
    coords_b = coords_b.to(device)
    feats_b = feats_b.to(feats_a.device)

    # Concatenate coordinates from both tensors
    all_coords = torch.cat([coords_a, coords_b], dim=0)

    # Find unique coordinates and inverse indices
    # This matches MinkowskiEngine's coordinate union behavior
    unique_coords, inverse = torch.unique(
        all_coords, dim=0, return_inverse=True, sorted=True
    )

    # Initialize feature tensor for unique coordinates
    # Using feats_a.new_zeros ensures correct dtype and device
    combined = feats_a.new_zeros((unique_coords.shape[0], feats_a.shape[1]))

    # Sum features at overlapping coordinates using index_add_
    # This is the key operation that matches MinkowskiEngine's addition:
    # - For non-overlapping coords: feature is added once
    # - For overlapping coords: features from both tensors are summed
    split = coords_a.shape[0]
    combined.index_add_(0, inverse[:split], feats_a)
    combined.index_add_(0, inverse[split:], feats_b)

    # Reconstruct batch offsets for the unified coordinate set
    batch_indices = unique_coords[:, 0].to(torch.int64)
    batch_size = a.batch_size

    if batch_indices.numel() == 0:
        offsets = torch.zeros(
            batch_size + 1,
            dtype=a.offsets.dtype,
            device=a.offsets.device
        )
    else:
        # Count coordinates per batch
        counts = torch.bincount(batch_indices.cpu(), minlength=batch_size)
        offsets = torch.zeros(
            batch_size + 1,
            dtype=a.offsets.dtype,
            device=a.offsets.device
        )
        offsets[1:] = counts.to(dtype=offsets.dtype).cumsum(dim=0)

    # Preserve coordinate metadata (tensor_stride, voxel_size, voxel_origin)
    coord_kwargs = {
        "offsets": offsets,
        "tensor_stride": a.tensor_stride,
    }
    voxel_size = getattr(a.batched_coordinates, "voxel_size", None)
    if voxel_size is not None:
        coord_kwargs["voxel_size"] = voxel_size
    voxel_origin = getattr(a.batched_coordinates, "voxel_origin", None)
    if voxel_origin is not None:
        coord_kwargs["voxel_origin"] = voxel_origin

    # Create new coordinate and feature objects
    # Extract spatial coordinates (remove batch index)
    spatial_coords = unique_coords[:, 1:].to(dtype=a.coordinate_tensor.dtype)
    new_coords = IntCoords(spatial_coords, **coord_kwargs)
    new_feats = CatFeatures(combined, offsets=offsets)

    # Return new Voxels with unified coordinates and summed features
    return a.replace(
        batched_coordinates=new_coords,
        batched_features=new_feats,
    )
