# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion structure utilty functions"""

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation as R

from mmdet3d.utils import array_converter

from mmdet3d.structures import Box3DMode, BaseInstance3DBoxes
import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


def get_box_type_tao3d(box_type: str) -> Tuple[type, int]:
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure. The valid value are "LiDAR",
            "Camera" and "Depth".

    Raises:
        ValueError: A ValueError is raised when ``box_type`` does not belong to
            the three valid types.

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = tao_structures.TAOLiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == 'camera':
        box_type_3d = tao_structures.TAOCameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    else:
        raise ValueError('Only "box_type" of "camera", "lidar" are '
                         f'supported, got {box_type}')

    return box_type_3d, box_mode_3d


def get_trig(angles):
    """get elements for rotation matrix"""
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    return rot_cos, rot_sin, ones, zeros


def rot_z(angles):
    """get rotaion in z-axis"""
    rot_cos, rot_sin, ones, zeros = get_trig(angles)
    rot_cos = torch.cos(angles)
    mat_z = torch.stack([torch.stack([rot_cos, rot_sin, zeros]),
                         torch.stack([-rot_sin, rot_cos, zeros]),
                         torch.stack([zeros, zeros, ones])])
    return mat_z


def rot_x(angles):
    """get rotaion in x-axis"""
    rot_cos, rot_sin, ones, zeros = get_trig(angles)
    mat_x = torch.stack([torch.stack([ones, zeros, zeros]),
                         torch.stack([zeros, rot_cos, rot_sin]),
                         torch.stack([zeros, -rot_sin, rot_cos])])
    return mat_x


def rot_y(angles):
    """get rotaion in y-axis"""
    rot_cos, rot_sin, ones, zeros = get_trig(angles)
    mat_y = torch.stack([torch.stack([rot_cos, zeros, -rot_sin]),
                         torch.stack([zeros, ones, zeros]),
                         torch.stack([rot_sin, zeros, rot_cos])])
    return mat_y


def get_rotation_matrix_3d(rotation_xyz):
    """get 3D Rotation Matrix"""
    return rot_x(rotation_xyz[0]) @ rot_y(rotation_xyz[1]) @ rot_z(rotation_xyz[2])


def quats_to_euler_angles(quaternions: np.ndarray, degrees: bool = False, device=None) -> np.ndarray:
    """Vectorized version of converting quaternions (scalar first) to euler angles

    Args:
        quaternions (np.ndarray): quaternions with shape (N, 4) or (4,) - scalar first
        degrees (bool, optional): Return euler angles in degrees if True, radians if False.
        Defaults to False.

    Returns:
        np.ndarray: Euler angles in extrinsic coordinates XYZ order with shape (N, 3) or (3,)
        corresponding to the quaternion rotations
    """
    if len(quaternions.shape) == 1:
        q = quaternions[[1, 2, 3, 0]]
    else:
        q = quaternions[:, [1, 2, 3, 0]]
    rot = R.from_quat(q)
    result = rot.as_euler("xyz", degrees)
    return result


def single_box_convert_sensors(object_location, object_rotation, proj_mat):
    """convert between sensors for single bounding box"""
    # proj_mat : Multiply vector from left
    model_space = np.array([0, 0, 0, 1])
    object_camera_xform = np.identity(4)
    rot_mat = get_rotation_matrix_3d(object_rotation)
    object_camera_xform[:3, :3] = rot_mat[:3, :3]
    object_camera_xform[3, 0] = object_location[0]
    object_camera_xform[3, 1] = object_location[1]
    object_camera_xform[3, 2] = object_location[2]
    object_lidar_xform = object_camera_xform @ proj_mat
    object_lidar_locs = model_space @ object_lidar_xform
    quaternion_xyzw = R.from_matrix(np.transpose((object_lidar_xform)[:3, :3])).as_quat()

    quaternion = np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
    rotations = quats_to_euler_angles(quaternion, False)
    location = [object_lidar_locs[0], object_lidar_locs[1], object_lidar_locs[2]]
    return torch.as_tensor(location), torch.as_tensor(rotations)


@array_converter(apply_to=('points_3d', 'proj_mat'))
def project_cam2img(points_3d: Union[Tensor, np.ndarray],
                    proj_mat: Union[Tensor, np.ndarray],
                    with_depth: bool = False) -> Union[Tensor, np.ndarray]:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates. Left-Hand
        norm_dim (int) : Which dimension to be used for normalization
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, \
        'The dimension of the projection matrix should be 2 ' \
        f'instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or \
        (d1 == 4 and d2 == 4), 'The shape of the projection matrix ' \
        f'({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    points_4d = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
    point_2d = points_4d @ proj_mat  # (Nx4)x (4x4) = (Nx4)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


@array_converter(apply_to=('points_3d', 'proj_mat'))
def project_lidar2img(points_3d: Union[Tensor, np.ndarray],
                      proj_mat: Union[Tensor, np.ndarray],
                      with_depth: bool = False) -> Union[Tensor, np.ndarray]:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates. Left-Hand
        norm_dim (int) : Which dimension to be used for normalization
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, \
        'The dimension of the projection matrix should be 2 ' \
        f'instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or \
        (d1 == 4 and d2 == 4), 'The shape of the projection matrix ' \
        f'({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    points_4d = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
    point_2d = points_4d @ proj_mat  # (Nx4)x (4x4) = (Nx4)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


def convert_cooridnates(tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]], src: int, dst: int,
                        yaw_dim: int = -1, proj_mat: Optional[Union[Tensor, np.ndarray]] = None) -> 'BaseInstance3DBoxes':
    """Convert tensor to ``dst`` mode.

    Args:
        dst (int): The target Box mode.
        proj_mat (Tensor or np.ndarray, optional): The rotation and
            translation matrix between different coordinates.
            Defaults to None. The conversion from ``src`` coordinates to
            ``dst`` coordinates usually comes along the change of sensors,
            e.g., from camera to LiDAR. This requires a transformation
            matrix. Left-Hand

    Returns:
        :obj:`BaseInstance3DBoxes`: The converted box of the same type in
        the ``dst`` mode.
    """
    if dst == Box3DMode.CAM:
        target_type = tao_structures.TAOCameraInstance3DBoxes
    elif dst == Box3DMode.LIDAR:
        target_type = tao_structures.TAOLiDARInstance3DBoxes

    if src == Box3DMode.CAM:
        src_type = tao_structures.TAOCameraInstance3DBoxes
    elif src == Box3DMode.LIDAR:
        src_type = tao_structures.TAOLiDARInstance3DBoxes

    if (src == Box3DMode.CAM and dst != Box3DMode.LIDAR) or (src == Box3DMode.LIDAR and dst != Box3DMode.CAM):
        raise NotImplementedError(
            f'Conversion to {type(target_type)} through {type(src_type)} '
            'is not supported yet')

    if yaw_dim != -1:
        if (src == Box3DMode.CAM and dst == Box3DMode.LIDAR):
            keep_dim = 2  # keep z rotation  (converting camera to lidar)
        elif (src == Box3DMode.LIDAR and dst == Box3DMode.CAM):
            keep_dim = 1  # keep y rotation (converting lidar to camera)
        else:
            keep_dim = -1  # keep all rotations
    else:
        keep_dim = -1

    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        arr = torch.from_numpy(np.asarray(tensor)).clone()
    else:
        arr = tensor.clone()

    # arr = tensor.clone()
    # convert cam coord to lidar coord
    if proj_mat is None:  # need to provide default one!
        raise NotImplementedError('Camera to LiDAR projection matrix needs to be defined.')

    x_size, y_size, z_size = tensor[..., 3:4], tensor[..., 4:5], tensor[..., 5:6]

    lidar_locs = []
    lidar_rots = []
    if arr.shape[0] == 1:  # single box
        locs, rots = single_box_convert_sensors(arr[0, 0:3], arr[0, 6:9], proj_mat)
        if keep_dim != -1:  # zero-pad rotations for zero-padded data
            for idx in range(rots.shape[0]):
                if idx != keep_dim:
                    rots[idx] = 0.0
        lidar_locs.append(locs)
        lidar_rots.append(rots)
    else:
        for batch_idx in range(arr.shape[0]):
            lidar_loc, lidar_rot = single_box_convert_sensors(arr[batch_idx, 0:3], arr[batch_idx, 6:9], proj_mat)
            if keep_dim != -1:  # zero-pad rotations for zero-padded data
                for idx in range(lidar_rot.shape[0]):
                    if idx != keep_dim:
                        lidar_rot[idx] = 0.0
            lidar_locs.append(lidar_loc)
            lidar_rots.append(lidar_rot)

    if len(lidar_locs) != 0:
        locs = torch.stack(lidar_locs, 0)
        rots = torch.stack(lidar_rots, 0)
        if yaw_dim == -1:
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
        else:  # swtich size for coordinate changes
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
    else:  # empty GT
        locs = torch.tensor([])
        locs = locs.reshape((0, 3))
        rots = torch.tensor([])
        rots = rots.reshape((0, 3))
        xyz_size = torch.tensor([])
        xyz_size = xyz_size.reshape((0, 3))

    arr = torch.cat([locs, xyz_size, rots], dim=-1)

    return target_type(arr, box_dim=arr.size(-1))
