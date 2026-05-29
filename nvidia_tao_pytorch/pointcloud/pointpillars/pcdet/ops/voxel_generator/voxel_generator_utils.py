# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxel Generator custom op in PyTorch"""
import numpy as np
import torch
from . import voxel_generator_cuda


# int batchSize, int maxNumPoints, at::Tensor pointCloudTensor, at::Tensor pointNum,
# float mMinXRange, float mMaxXRange, float mMinYRange, float mMaxYRange,
# float mMinZRange, float mMaxZRange, float mPillarXSize, float mPillarYSize,
# float mPillarZSize, int mPointFeatureNum,
# int mPointNum, at::Tensor maskTensor, at::Tensor voxelsTensor, int mPillarNum,
# at::Tensor paramsDataTensor, at::Tensor voxelFeaturesTensor, int mFeatureNum,
# at::Tensor voxelNumPointsTensor, at::Tensor coordsDataTensor,
# at::Tensor pillarFeaturesDataTensor
def voxel_generator_gpu(
    batch_size, max_num_points_per_file, points, valid_points_num_per_file,
    pc_range, pillar_size, point_feature_num,
    max_point_num_per_voxel, max_pillar_num, output_feature_num, device
):
    """Custom op for Voxel Generator."""
    grid_size = (np.array(pc_range[3:6]) - np.array(pc_range[0:3])) / np.array(pillar_size)
    grid_size = np.round(grid_size).astype(np.int64)
    assert grid_size[2] == 1, f"Grid size at Z dimension should be 1, got {grid_size[2]}."
    dense_pillar_num = grid_size[0] * grid_size[1] * grid_size[2]
    mask = torch.zeros((batch_size, dense_pillar_num), dtype=torch.int32, device=device)
    voxels = torch.zeros((batch_size, dense_pillar_num, max_point_num_per_voxel, point_feature_num), dtype=torch.float32, device=device)
    params_data = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    voxel_features = torch.zeros((batch_size, dense_pillar_num, max_point_num_per_voxel, point_feature_num), dtype=torch.float32, device=device)
    voxel_num_points = torch.zeros((batch_size, dense_pillar_num), dtype=torch.int32, device=device)
    coords_data = torch.zeros((batch_size, max_pillar_num, 4), dtype=torch.int32, device=device)
    pillar_features = torch.zeros((batch_size, max_pillar_num, max_point_num_per_voxel, output_feature_num), dtype=torch.float32, device=device)
    voxel_generator_cuda.voxel_generator_gpu(
        batch_size,
        max_num_points_per_file,
        torch.from_numpy(points).float().to(device),
        torch.from_numpy(valid_points_num_per_file).int().to(device),
        pc_range[0],
        pc_range[3],
        pc_range[1],
        pc_range[4],
        pc_range[2],
        pc_range[5],
        pillar_size[0],
        pillar_size[1],
        pillar_size[2],
        point_feature_num,
        max_point_num_per_voxel,
        mask,
        voxels,
        max_pillar_num,
        params_data,
        voxel_features,
        output_feature_num,
        voxel_num_points,
        coords_data,
        pillar_features
    )
    actual_pillar_num_per_frame = params_data.cpu().numpy()
    pillar_features = pillar_features.cpu().numpy()[0, :actual_pillar_num_per_frame[0], ..., :4]
    coordinates = coords_data.cpu().numpy()[0, :actual_pillar_num_per_frame[0], 1:]
    sparse_voxel_num_points = np.sum(np.sum(np.absolute(pillar_features), axis=-1) > 0, axis=-1)
    return pillar_features, coordinates, sparse_voxel_num_points
