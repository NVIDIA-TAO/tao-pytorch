# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection3D blocks for Sparse4D."""

import torch
import torch.nn as nn
from torch.nn import Linear
import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.model.blocks import linear_relu_ln
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX


class SparseBox3DEncoder(nn.Module):
    """Sparse box 3D encoder."""

    def __init__(
        self,
        embed_dims,
        pos_embed_only=False,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        """Initialize the SparseBox3DEncoder.

        Args:
            embed_dims: Embedding dimensions
            pos_embed_only: Whether to only embed position
            vel_dims: Velocity dimensions
            mode: Mode for embedding
        """
        super().__init__()
        assert mode in ["add", "cat"], f"mode: {mode} must be one of 'add', 'cat'"
        embed_dims = list(embed_dims)
        self.pos_embed_only = pos_embed_only
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            """Create an embedding layer."""
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        if not self.pos_embed_only:
            self.size_fc = embedding_layer(3, embed_dims[1])
            self.yaw_fc = embedding_layer(2, embed_dims[2])
            if vel_dims > 0:
                self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        """Forward pass through the SparseBox3DEncoder."""
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        if not self.pos_embed_only:
            size_feat = self.size_fc(box_3d[..., [W, L, H]])
            yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
            if self.mode == "add":
                output = pos_feat + size_feat + yaw_feat
            elif self.mode == "cat":
                output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

            if self.vel_dims > 0:
                vel_feat = self.vel_fc(box_3d[..., VX: VX + self.vel_dims])
                if self.mode == "add":
                    output = output + vel_feat
                elif self.mode == "cat":
                    output = torch.cat([output, vel_feat], dim=-1)
        else:
            output = pos_feat

        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        """Initialize the Scale layer.

        Args:
            scale (float): Initial value of scale factor. Default: 1.0
        """
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Scale layer."""
        return x * self.scale


class SparseBox3DRefinementModule(nn.Module):
    """Sparse box 3D refinement module."""

    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        """Initialize the SparseBox3DRefinementModule.

        Args:
            embed_dims (int): Embedding dimensions
            output_dim (int): Output dimensions
            num_cls (int): Number of classes
            normalize_yaw (bool): Whether to normalize yaw
            refine_yaw (bool): Whether to refine yaw
            with_cls_branch (bool): Whether to use classification branch
            with_quality_estimation (bool): Whether to use quality estimation branch
        """
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

    def init_weight(self):
        """Initialize the weights of the SparseBox3DRefinementModule."""
        if self.with_cls_branch:
            bias_init = -np.log((1 - 0.01) / 0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        """Forward pass through the SparseBox3DRefinementModule.

        Args:
            instance_feature: Instance feature
            anchor: Anchor
            anchor_embed: Anchor embedding
        """
        feature = instance_feature + anchor_embed
        output = self.layers(feature)

        # Ensure output and anchor have the same data type
        if output.dtype != anchor.dtype:
            output = output.to(anchor.dtype)

        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity  # anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality


class SparseBox3DKeyPointsGenerator(nn.Module):
    """Sparse box 3D key points generator."""

    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        """Initialize the SparseBox3DKeyPointsGenerator.

        Args:
            embed_dims (int): Embedding dimensions
            num_learnable_pts (int): Number of learnable points
            fix_scale (tuple): Fixed scale
        """
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        """Initialize the weights of the SparseBox3DKeyPointsGenerator."""
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.constant_(self.learnable_fc.bias, 0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        """Forward pass through the SparseBox3DKeyPointsGenerator.

        Args:
            anchor (torch.Tensor): Anchor
            instance_feature (torch.Tensor): Instance feature
            T_cur2temp_list (list): Current to temporary list
            cur_timestamp (torch.Tensor): Current timestamp
            temp_timestamps (torch.Tensor): Temporary timestamps
        """
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_learnable_pts, 3).sigmoid() - 0.5)
            # Ensure same dtype before concatenation
            if learnable_scale.dtype != scale.dtype:
                learnable_scale = learnable_scale.to(scale.dtype)
            scale = torch.cat([scale, learnable_scale], dim=-2)
        key_points = scale * anchor[..., None, [W, L, H]].exp()
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        # Ensure same dtype before addition
        if key_points.dtype != anchor[..., None, [X, Y, Z]].dtype:
            key_points = key_points.to(anchor.dtype)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (cur_timestamp is None or temp_timestamps is None or T_cur2temp_list is None or len(temp_timestamps) == 0):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (velocity * time_interval.to(dtype=velocity.dtype)[:, None, None])
            # Ensure same dtype before subtraction
            if translation.dtype != key_points.dtype:
                translation = translation.to(key_points.dtype)
            temp_key_points = key_points - translation[:, :, None]
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)

            # Ensure consistent dtypes for concatenation
            ones_tensor = torch.ones_like(temp_key_points[..., :1])
            if ones_tensor.dtype != temp_key_points.dtype:
                ones_tensor = ones_tensor.to(temp_key_points.dtype)

            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        ones_tensor,
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        """Project the anchor to the destination.

        Args:
            anchor: Anchor
            T_src2dst_list: Source to destination list
            src_timestamp: Source timestamp
            dst_timestamps: Destination timestamps
            time_intervals: Time intervals
        """
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            center = anchor[..., [X, Y, Z]]
            size = anchor[..., [W, L, H]]
            yaw = anchor[..., [COS_YAW, SIN_YAW]]

            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                # object motion compensation
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                # Ensure same dtype before addition
                if translation.dtype != center.dtype:
                    translation = translation.to(center.dtype)
                center = center - translation

            if T_src2dst_list[i] is not None:
                # ego-motion compensation if ego-motion is defined
                T_src2dst = torch.unsqueeze(
                    T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
                )
                center_transformed = torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                # Ensure same dtype before addition
                if center_transformed.dtype != T_src2dst[..., :3, 3].dtype:
                    center_transformed = center_transformed.to(T_src2dst[..., :3, 3].dtype)
                center = center_transformed + T_src2dst[..., :3, 3]

                yaw = torch.matmul(
                    T_src2dst[..., :2, :2], yaw[..., None],
                ).squeeze(-1)
                vel = torch.matmul(
                    T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
                ).squeeze(-1)
            yaw = yaw[..., [1, 0]]
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        """Calculate the distance of the anchor."""
        return torch.norm(anchor[..., :2], p=2, dim=-1)
