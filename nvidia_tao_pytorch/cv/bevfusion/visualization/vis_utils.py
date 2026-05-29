# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BEVFusion visualizer utility functions. """

import copy
import numpy as np
import torch

import nvidia_tao_pytorch.cv.bevfusion.structures as tao_structures


# project the camera bboxes 3d to image
def proj_bbox3d_cam2img(bboxes_3d: tao_structures.TAOCameraInstance3DBoxes,
                        input_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes_3d (:obj:`TAOCameraInstance3DBoxes`): 3D bbox in camera coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    cam2img = copy.deepcopy(input_meta['cam2img'])
    # make  corners
    corners_3d = bboxes_3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3]) or
            cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    imgfov_pts_2d = tao_structures.project_cam2img(points_3d, cam2img)
    imgfov_pts_2d = imgfov_pts_2d.reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


def proj_bbox3d_lidar2img(bboxes_3d: tao_structures.TAOLiDARInstance3DBoxes,
                          input_meta: dict) -> np.ndarray:
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`TAOLiDARInstance3DBoxes`): 3D bbox in lidar coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    lidar2img = copy.deepcopy(input_meta['lidar2img'])

    corners_3d = bboxes_3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(lidar2img, torch.Tensor):
        lidar2img = torch.from_numpy(np.array(lidar2img))

    assert (lidar2img.shape == torch.Size([3, 3]) or
            lidar2img.shape == torch.Size([4, 4]))
    lidar2img = lidar2img.float().cpu()

    imgfov_pts_2d = tao_structures.project_lidar2img(points_3d, lidar2img)
    imgfov_pts_2d = imgfov_pts_2d.reshape(num_bbox, 8, 2)
    return imgfov_pts_2d
