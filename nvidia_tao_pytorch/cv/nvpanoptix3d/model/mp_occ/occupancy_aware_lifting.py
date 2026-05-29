# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Occupancy aware lifting module for NVPanoptix3D using WarpConvNet."""

import torch
from torch import nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.mp_occ.back_projection import BackProjection
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.sparse_utils import \
    mask_invalid_sparse_voxels


class OccupancyAwareLifting(nn.Module):
    """Occupancy aware lifting module for NVPanoptix3D using WarpConvNet."""

    def __init__(self, cfg):
        """Initialize the OccupancyAwareLifting module."""
        super(OccupancyAwareLifting, self).__init__()
        self.back_projection = BackProjection(cfg)

    def forward(self, pred, kept, mapping, occupancy2d, room_mask=None):
        """Forward pass
        Args:
            pred: List of dictionaries containing depth, semantic, and occupancy.
            kept: Tensor of kept voxels.
            mapping: Tensor of mapping.
            occupancy2d: Tensor of occupancy2d.
            room_mask: Tensor of room mask.
        Returns:
            proj_feat: Voxels of projected features.
        """
        # get the depth, semantic, and occupancy
        depth = torch.stack([p["depth"][None] for p in pred])
        features = torch.stack([p["sem_seg"] for p in pred])
        depth_weight = occupancy2d.to(depth.device)
        # Number of depth bins is inferred from the occupancy2d channels.
        num_bins = int(depth_weight.shape[1])
        kept = kept.to(depth.device)
        mapping = mapping.to(depth.device)

        semantic = features.argmax(1)
        depth_max_value = self.back_projection.depth_max
        batch = semantic.shape[0]

        # clip depth in range [0, depth_max_value]
        depth[depth > depth_max_value] = depth_max_value

        # get the bin index of depth 0 - num_bins
        depth_feat = (depth / depth_max_value * float(num_bins))

        depth_index = depth_feat.long()
        depth_weight_kept = torch.ones_like(
            depth_weight, dtype=torch.long
        ) * torch.arange(0, num_bins, device=depth.device, dtype=torch.long)[None, :, None, None]

        # stuff: wall, floor, or ceiling, erode the stuff class
        stuff = (-F.max_pool2d(-(semantic >= 10).float(), 5, 1, 2)).bool()
        # get the depth of the stuff class
        stuff_depth = depth[:, 0] * stuff

        # get the max depth of the stuff class in x direction: (batch_size, h)
        stuff_x_max = stuff_depth.max(1)[0]
        # get the max depth of the stuff class in y direction: (batch_size, w)
        stuff_y_max = stuff_depth.max(2)[0]

        stuff_depth_l = stuff_depth[:, 0].clone()
        stuff_depth_r = stuff_depth[:, -1].clone()
        stuff_depth_t = stuff_depth[:, :, 0].clone()
        stuff_depth_d = stuff_depth[:, :, -1].clone()

        for bi in range(batch):
            stuff_depth[bi, 0] = stuff_padding(stuff_depth_l[bi], stuff_y_max[bi])
            stuff_depth[bi, -1] = stuff_padding(stuff_depth_r[bi], stuff_y_max[bi].flip(0))
            stuff_depth[bi, :, 0] = stuff_padding(stuff_depth_t[bi], stuff_x_max[bi])
            stuff_depth[bi, :, -1] = stuff_padding(stuff_depth_d[bi], stuff_x_max[bi].flip(0))

        stuff_x = stuff_depth.max(1)[0]
        stuff_y = stuff_depth.max(2)[0]

        for bi in range(batch):
            stuff_x[bi] = find_none(stuff_x[bi])
            stuff_y[bi] = find_none(stuff_y[bi])

        depth_pixels_xy = torch.ones_like(depth).nonzero()

        depth_max = torch.cat(
            [
                stuff_x[depth_pixels_xy[:, 0], depth_pixels_xy[:, 3]][..., None],
                stuff_y[depth_pixels_xy[:, 0], depth_pixels_xy[:, 2]][..., None]
            ],
            dim=-1
        ).min(-1)[0].reshape(*depth.shape)

        depth_max = (depth_max / depth_max_value * float(num_bins)).long()  # get the min bin index of stuff class
        depth_feat = (depth_weight_kept - depth_index) / float(num_bins) * depth_max_value

        # get the sign and the distance of voxel to the surface
        depth_feat = torch.cat([depth_feat.sign()[:, None], depth_feat[:, None].abs()], 1)

        # keep voxel 3 bins before surface to 5 bins after stuff class max depth
        depth_weight_kept = (depth_weight_kept > (depth_index - 3)) * (
            depth_weight_kept < (depth_max + 5)
        )

        depth_weight = depth_weight.sigmoid() * depth_weight_kept

        feat_kept = kept.clone()

        if room_mask is not None:
            room_mask = room_mask.unsqueeze(1)
            depth_weight_kept = depth_weight_kept * room_mask

        mapping_kept = mapping[kept]
        mapping_kept[:, -1] = mapping_kept[:, -1] * num_bins / 6
        mapping_kept = mapping_kept.long().to(depth.device)

        # only keep voxel before 3 bins before surface
        # and after 5 bins after stuff class max depth and in the frustum:
        feat_kept[kept] = depth_weight_kept[
            mapping_kept[:, 0], mapping_kept[:, -1], mapping_kept[:, 2], mapping_kept[:, 1]]

        features = torch.cat([features[:, :, None].repeat(1, 1, num_bins, 1, 1),
                              depth_weight[:, None], depth_feat], 1)

        coord_sparse = feat_kept.nonzero()
        mapping_feat_kept = mapping[feat_kept]

        # convert to bin index:
        mapping_feat_kept[:, -1] = mapping_feat_kept[:, -1] * num_bins / depth_max_value
        mapping_feat_kept = mapping_feat_kept.long()
        feat_sparse = features[
            mapping_feat_kept[:, 0], :,
            mapping_feat_kept[:, -1],
            mapping_feat_kept[:, 2],
            mapping_feat_kept[:, 1]
        ]

        padding_kept = F.max_pool3d(feat_kept.float(), 5, 1, 2).bool()
        padding_kept[~kept] = False

        batch_point = padding_kept.flatten(1, -1).sum(-1)
        batch_zero = (batch_point == 0).nonzero().view(-1)

        # fix no points
        if len(batch_zero) > 0:
            padding_kept[batch_zero, 127, 127, 127] = True
        padding_kept[feat_kept] = False
        coord_padding = padding_kept.nonzero().contiguous().float()

        coord_padding[:, 1:] = coord_padding[:, 1:] // 2 * 2
        feat_padding = torch.zeros(
            (len(coord_padding), features.shape[1]),
            device=features.device, dtype=torch.float
        )

        feat_sparse = torch.cat([feat_sparse, feat_padding])
        coord_sparse = torch.cat([coord_sparse, coord_padding])
        coord_sparse[:, 1:] = coord_sparse[:, 1:] - 1.0

        proj_feat = Voxels(
            batched_coordinates=IntCoords(
                coord_sparse[:, 1:].contiguous().int(),
                offsets=offsets_from_batch_index(coord_sparse[:, 0].int()),
                tensor_stride=1
            ),
            batched_features=CatFeatures(
                feat_sparse, offsets=offsets_from_batch_index(coord_sparse[:, 0].int())
            ),
            voxel_size=self.back_projection.voxel_size,  # voxel_size: 0.03
        )

        proj_feat = mask_invalid_sparse_voxels(proj_feat)
        return proj_feat, None


def stuff_padding(padding, max_value):
    """Padding the stuff class.
    Args:
        padding: Tensor of padding.
        max_value: Tensor of max value.
    Returns:
        padding: Tensor of padded stuff class.
    """
    padding = padding.clone()
    padding_mask = padding == 0
    if padding_mask.sum() > 0:
        for value in max_value:
            if value != 0:
                break
            padding[padding_mask] = value
    return padding


def find_none(stuff_a, min_value=0):
    """Find the none value in the stuff class.
    Args:
        stuff_a: Tensor of stuff class.
        min_value: Tensor of min value.
    Returns:
        stuff_a: Tensor of stuff class with none value.
    """
    none_v = torch.nonzero(stuff_a == 0)
    for v in none_v:
        l_stuff = stuff_a[:v]
        l_stuff = l_stuff[l_stuff != 0]
        l_stuff = min(l_stuff) if len(l_stuff) else min_value
        r_stuff = stuff_a[v + 1:]
        r_stuff = r_stuff[r_stuff != 0]
        r_stuff = min(r_stuff) if len(r_stuff) else min_value
        stuff_a[v] = max(l_stuff, r_stuff)
    return stuff_a
