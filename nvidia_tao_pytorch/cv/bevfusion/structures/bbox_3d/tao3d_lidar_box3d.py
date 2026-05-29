# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion lidar structure functions"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.structures import limit_period, BasePoints, BaseInstance3DBoxes, rotation_3d_in_axis
from mmdet3d.structures import Box3DMode

from .utils import get_rotation_matrix_3d, convert_cooridnates


class TAOLiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LiDAR coordinates.
    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0.0),
    """

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 9,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0.0),
    ) -> None:
        """
        Args:
            tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
                data with shape (N, box_dim).
            box_dim (int): Number of the dimension of a box. Each row is
                (x, y, z, x_size, y_size, z_size, x_rot, y_rot, z_rot). Defaults to 9.
            origin (Tuple[float]): Relative position of the box origin.
                Defaults to (0.5, 0.5, 0.0).

        Attributes:
            tensor (Tensor): Float matrix with shape (N, box_dim).
            box_dim (int): Integer indicating the dimension of a box. Each row is
                (x, y, z, x_size, y_size, z_size, x_rot, y_rot, z_rot).

        """
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')
        assert box_dim == 9
        self.box_dim = box_dim
        self.tensor = tensor.clone()
        self.default_origin = (0.5, 0.5, 0.0)
        if origin != self.default_origin:
            dst = self.tensor.new_tensor(self.default_origin)
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        # the positive direction is down rather than up
        return self.bottom_height - self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        dim_vec = dims.new_tensor(self.default_origin)
        corners_norm = corners_norm - dim_vec
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        rot_xyz = self.tensor[:, 6:9]
        if rot_xyz.shape[0] == 1:
            rot_mat = get_rotation_matrix_3d(rot_xyz[0, ...])  # single box
            corners = corners @ rot_mat
            corners += self.tensor[:, :3].view(-1, 1, 3)
        else:
            corners_stack = []
            for batch_idx in range(rot_xyz.shape[0]):
                rot_mat = get_rotation_matrix_3d(rot_xyz[batch_idx, :])  # multiple boxes batched
                single_corner = corners[batch_idx, ...]
                single_corner = single_corner @ rot_mat
                single_corner += self.tensor[:, :3][batch_idx, ...].view([1, 3])
                corners_stack.append(single_corner)
            corners = torch.stack(corners_stack, 0)
        return corners

    @classmethod
    def cat(cls, boxes_list: Sequence['BaseInstance3DBoxes']
            ) -> 'BaseInstance3DBoxes':
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (Sequence[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].box_dim)
        return cat_boxes

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            box_dim=self.box_dim)

    def cpu(self) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cpu device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(), box_dim=self.box_dim)

    def cuda(self, *args, **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cuda device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            box_dim=self.box_dim)

    def clone(self) -> 'BaseInstance3DBoxes':
        """Clone the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim)

    def __getitem__(
            self, item: Union[int, slice, np.ndarray,
                              Tensor]) -> 'BaseInstance3DBoxes':
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            output = original_type(self.tensor[item].view(1, -1),
                                   box_dim=self.box_dim)
        else:
            b = self.tensor[item]
            assert b.dim() == 2, \
                f'Indexing on Boxes with {item} failed to return a matrix!'
            output = original_type(b, box_dim=self.box_dim)
        return output

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=2,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 8] += angle

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T
        else:
            raise ValueError

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1] = -self.tensor[:, 1]
            self.tensor[:, 8] = -self.tensor[:, 8]
        elif bev_direction == 'vertical':
            self.tensor[:, 0] = -self.tensor[:, 0]
            self.tensor[:, 8] = -self.tensor[:, 8] + np.pi

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                elif bev_direction == 'vertical':
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points
        else:
            raise ValueError

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   yaw_dim: int = -1) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix. (Left-Hand)
        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        if dst != Box3DMode.CAM:
            raise NotImplementedError(
                f'Conversion to {dst} through {type(self)} '
                'is not supported yet')
        else:  # LiDAR to CAM
            return convert_cooridnates(tensor=self.tensor, src=Box3DMode.LIDAR, dst=Box3DMode.CAM, yaw_dim=yaw_dim, proj_mat=rt_mat)

    def limit_yaw(self, offset: float = 0.5, period: float = np.pi) -> None:
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw. Defaults to 0.5.
            period (float): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 8] = limit_period(self.tensor[:, 8], offset, period)
