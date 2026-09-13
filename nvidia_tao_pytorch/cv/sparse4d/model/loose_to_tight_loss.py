# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable geometry and loss helpers for 2D box distillation."""

import json
from typing import Tuple

import numpy as np
import torch

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import assemble_features


__all__ = [
    "load_2dgt_index",
    "box3d_to_corners_torch",
    "project_cuboid_aabb_torch",
    "camera_view_geometry_torch",
    "loose_and_features",
    "giou",
    "loose_to_tight_2d_loss",
]


_EPS = 1e-6


def load_2dgt_index(npz_path: str):
    """Load a per-scene 2D-ground-truth side cache.

    Returns:
        Tuple ``(index, metadata)``. ``index`` is keyed by frame ID, instance
        ID, and camera name, and each leaf contains ``(visible_box, weight)``.
    """
    with np.load(npz_path, allow_pickle=False) as cache:
        metadata = (
            json.loads(bytes(cache["_meta"]).decode("utf-8"))
            if "_meta" in cache
            else {}
        )
        frame_ids = cache["frame_id"]
        instance_ids = cache["instance_id"]
        camera_ids = cache["cam"]
        visible_boxes = cache["box3"]
        occlusion_weights = cache["occ"]

    camera_names = metadata.get("cam_names", [])
    index = {}
    for row in range(len(frame_ids)):
        camera_id = int(camera_ids[row])
        camera_name = camera_names[camera_id] if camera_names else camera_id
        frame_index = index.setdefault(int(frame_ids[row]), {})
        instance_index = frame_index.setdefault(int(instance_ids[row]), {})
        instance_index[camera_name] = (
            visible_boxes[row],
            float(occlusion_weights[row]),
        )
    return index, metadata


def box3d_to_corners_torch(boxes7: torch.Tensor) -> torch.Tensor:
    """Convert decoded ``[..., x, y, z, w, l, h, yaw]`` boxes to corners."""
    if boxes7.shape[-1] < 7:
        raise ValueError("boxes7 must have at least seven values per box")

    center = boxes7[..., :3]
    width, length, height = boxes7[..., 3], boxes7[..., 4], boxes7[..., 5]
    yaw = boxes7[..., 6]
    signs = boxes7.new_tensor(
        [
            [x_sign, y_sign, z_sign]
            for x_sign in (-0.5, 0.5)
            for y_sign in (-0.5, 0.5)
            for z_sign in (-0.5, 0.5)
        ]
    )
    dimensions = torch.stack([width, length, height], dim=-1).unsqueeze(-2)
    corners = signs * dimensions

    cosine = torch.cos(yaw).unsqueeze(-1)
    sine = torch.sin(yaw).unsqueeze(-1)
    rotated_x = cosine * corners[..., 0] - sine * corners[..., 1]
    rotated_y = sine * corners[..., 0] + cosine * corners[..., 1]
    rotated = torch.stack([rotated_x, rotated_y, corners[..., 2]], dim=-1)
    return rotated + center.unsqueeze(-2)


def project_cuboid_aabb_torch(
    boxes7: torch.Tensor,
    proj_mat: torch.Tensor,
    image_wh: torch.Tensor,
    eps: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project cuboids into one camera and return clipped loose 2D boxes.

    Args:
        boxes7: Decoded boxes with shape ``(N, >=7)`` in ego coordinates.
        proj_mat: Homogeneous ego-to-image projection with shape ``(4, 4)``.
        image_wh: Image width and height with shape ``(2,)``.
        eps: Minimum valid camera-space corner depth.

    Returns:
        A tuple containing boxes with shape ``(N, 4)`` and a validity mask with
        shape ``(N,)``. Projection remains differentiable with respect to boxes.
    """
    if boxes7.ndim != 2:
        raise ValueError("boxes7 must have shape (N, >=7)")
    if proj_mat.shape != (4, 4):
        raise ValueError("proj_mat must have shape (4, 4)")

    projection = proj_mat.to(dtype=boxes7.dtype, device=boxes7.device)
    image_size = torch.as_tensor(
        image_wh,
        dtype=boxes7.dtype,
        device=boxes7.device,
    ).reshape(2)
    image_width, image_height = image_size.clamp_min(0.0).unbind()

    corners = box3d_to_corners_torch(boxes7)
    homogeneous = torch.cat(
        [corners, corners.new_ones((corners.shape[0], 8, 1))],
        dim=-1,
    )
    projected = torch.matmul(homogeneous, projection.T)
    depth = projected[..., 2]
    valid = (depth > eps).all(dim=1)
    pixels = projected[..., :2] / depth.clamp_min(eps).unsqueeze(-1)

    x_min = torch.minimum(pixels[..., 0].amin(dim=1).clamp_min(0.0), image_width)
    x_max = torch.minimum(pixels[..., 0].amax(dim=1).clamp_min(0.0), image_width)
    y_min = torch.minimum(pixels[..., 1].amin(dim=1).clamp_min(0.0), image_height)
    y_max = torch.minimum(pixels[..., 1].amax(dim=1).clamp_min(0.0), image_height)
    boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    return boxes_xyxy, valid


def camera_view_geometry_torch(
    boxes7: torch.Tensor,
    ego2cam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute relative yaw, elevation, bearing, and distance in torch."""
    if boxes7.shape[-1] < 7:
        raise ValueError("boxes7 must have at least seven values per box")
    if ego2cam.shape != (4, 4):
        raise ValueError("ego2cam must have shape (4, 4)")

    transform = ego2cam.to(dtype=boxes7.dtype, device=boxes7.device)
    centers = boxes7[..., :3]
    homogeneous_centers = torch.cat(
        [centers, centers.new_ones(centers.shape[:-1] + (1,))],
        dim=-1,
    )
    centers_camera = torch.matmul(homogeneous_centers, transform.T)[..., :3]
    x_camera, y_camera, z_camera = centers_camera.unbind(dim=-1)

    distance = torch.linalg.norm(centers_camera, dim=-1)
    horizontal_distance = torch.sqrt(
        (x_camera.square() + z_camera.square()).clamp_min(_EPS)
    )
    bearing = torch.atan2(x_camera, z_camera)
    elevation = torch.atan2(-y_camera, horizontal_distance)
    forward_world = transform[2, :3]
    camera_yaw = torch.atan2(forward_world[1], forward_world[0])
    relative_yaw = boxes7[..., 6] - camera_yaw
    return relative_yaw, elevation, bearing, distance


def loose_and_features(
    boxes7: torch.Tensor,
    class_id: torch.Tensor,
    proj_mat: torch.Tensor,
    ego2cam: torch.Tensor,
    image_wh: torch.Tensor,
    num_classes: int,
    eps: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loose boxes, MLP features, and projection validity for one camera."""
    loose_boxes, valid = project_cuboid_aabb_torch(
        boxes7,
        proj_mat,
        image_wh,
        eps=eps,
    )
    relative_yaw, elevation, bearing, distance = camera_view_geometry_torch(
        boxes7,
        ego2cam,
    )
    image_size = (
        torch.as_tensor(
            image_wh,
            dtype=boxes7.dtype,
            device=boxes7.device,
        )
        .reshape(1, 2)
        .expand(boxes7.shape[0], 2)
    )
    features = assemble_features(
        class_id,
        boxes7[..., 3:6],
        relative_yaw,
        elevation,
        bearing,
        loose_boxes,
        image_size,
        distance,
        num_classes=num_classes,
    )
    return loose_boxes, features, valid


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
    heights = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
    return widths * heights


def _intersection_area(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    x_min = torch.maximum(boxes_a[..., 0], boxes_b[..., 0])
    y_min = torch.maximum(boxes_a[..., 1], boxes_b[..., 1])
    x_max = torch.minimum(boxes_a[..., 2], boxes_b[..., 2])
    y_max = torch.minimum(boxes_a[..., 3], boxes_b[..., 3])
    return (x_max - x_min).clamp_min(0.0) * (y_max - y_min).clamp_min(0.0)


def giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return element-wise generalized intersection over union."""
    intersection = _intersection_area(boxes_a, boxes_b)
    union = (_box_area(boxes_a) + _box_area(boxes_b) - intersection).clamp_min(_EPS)
    iou = intersection / union

    enclosing_x_min = torch.minimum(boxes_a[..., 0], boxes_b[..., 0])
    enclosing_y_min = torch.minimum(boxes_a[..., 1], boxes_b[..., 1])
    enclosing_x_max = torch.maximum(boxes_a[..., 2], boxes_b[..., 2])
    enclosing_y_max = torch.maximum(boxes_a[..., 3], boxes_b[..., 3])
    enclosing_area = (
        (enclosing_x_max - enclosing_x_min).clamp_min(0.0) *
        (enclosing_y_max - enclosing_y_min).clamp_min(0.0)
    ).clamp_min(_EPS)
    return iou - (enclosing_area - union) / enclosing_area


def loose_to_tight_2d_loss(
    pred_amodal: torch.Tensor,
    gt_visible: torch.Tensor,
    occ_weight: torch.Tensor,
    img_diag: torch.Tensor,
    tight_l1_weight: float = 1.0,
    containment_weight: float = 1.0,
) -> torch.Tensor:
    """Compute asymmetric, occlusion-weighted tightness and containment loss.

    Tightness uses GIoU and diagonal-normalized L1 and is attenuated by the
    visible-to-amodal area ratio. Containment remains active for every sample.
    The returned scalar is summed over samples.
    """
    generalized_iou = giou(pred_amodal, gt_visible)
    normalized_l1 = (pred_amodal - gt_visible).abs().sum(dim=-1) / img_diag.clamp_min(
        _EPS
    )
    tightness = 1.0 - generalized_iou + tight_l1_weight * normalized_l1

    intersection = _intersection_area(pred_amodal, gt_visible)
    uncovered = 1.0 - intersection / _box_area(gt_visible).clamp_min(_EPS)
    per_sample = occ_weight * tightness + containment_weight * uncovered
    return per_sample.sum()
