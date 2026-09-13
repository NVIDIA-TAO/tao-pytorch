# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry-only correction from loose projected boxes to tight 2D boxes.

A projected 3D cuboid produces a loose axis-aligned 2D box around all eight
corners. ``LooseToTightMLP`` predicts four per-edge scale factors that pull that
box toward its center. The model uses geometry only and can therefore be frozen
during Sparse4D training while gradients continue through the box projection.
"""

import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "WAREHOUSE_V4_CLASSES",
    "NUM_GEOM_FEATURES",
    "LooseToTightMLP",
    "assemble_features",
    "camera_view_geometry",
]


WAREHOUSE_V4_CLASSES = [
    "person",
    "gr1_t2",
    "agility_digit",
    "nova_carter",
    "transporter",
    "forklift",
    "pallet_truck",
]

# log-extent(3), relative yaw(2), elevation(2), bearing(2), normalized
# loose-box center(2) and size(2), and log-distance(1).
NUM_GEOM_FEATURES = 14

_EPS = 1e-6


def _as_tensor(value, like: torch.Tensor) -> torch.Tensor:
    """Convert ``value`` to a tensor matching ``like``."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return value.to(dtype=like.dtype, device=like.device)


def assemble_features(
    class_id: torch.Tensor,
    extent_wlh: torch.Tensor,
    theta_rel: torch.Tensor,
    phi: torch.Tensor,
    psi: torch.Tensor,
    loose_xyxy: torch.Tensor,
    img_wh: Union[torch.Tensor, Sequence[float]],
    distance: torch.Tensor,
    num_classes: int = len(WAREHOUSE_V4_CLASSES),
) -> torch.Tensor:
    """Assemble the canonical ``(..., num_classes + 14)`` feature vector.

    Args:
        class_id: Integer class IDs with shape ``(...)``.
        extent_wlh: Box dimensions ``(W, L, H)`` with shape ``(..., 3)``.
        theta_rel: Object yaw relative to camera yaw, in radians.
        phi: Camera elevation angle to the object, in radians.
        psi: Camera bearing angle to the object, in radians.
        loose_xyxy: Loose box coordinates in pixels with shape ``(..., 4)``.
        img_wh: Image width and height, broadcastable to ``(..., 2)``.
        distance: Camera-to-object distance in metres.
        num_classes: Size of the one-hot class block.

    Returns:
        Floating-point tensor with shape ``(..., num_classes + 14)``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    extent_wlh = extent_wlh.float()
    theta_rel = theta_rel.float()
    phi = phi.float()
    psi = psi.float()
    loose_xyxy = loose_xyxy.float()
    distance = _as_tensor(distance, loose_xyxy)

    one_hot = F.one_hot(
        class_id.long().clamp(0, num_classes - 1), num_classes=num_classes
    ).to(dtype=extent_wlh.dtype, device=extent_wlh.device)
    log_extent = torch.log(extent_wlh.clamp_min(_EPS))

    image_size = _as_tensor(img_wh, loose_xyxy)
    image_size = torch.broadcast_to(image_size, loose_xyxy.shape[:-1] + (2,))
    image_width = image_size[..., 0].clamp_min(_EPS)
    image_height = image_size[..., 1].clamp_min(_EPS)

    x_min, y_min, x_max, y_max = loose_xyxy.unbind(dim=-1)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    box_width = (x_max - x_min).clamp_min(0.0)
    box_height = (y_max - y_min).clamp_min(0.0)

    geometry = torch.stack(
        [
            log_extent[..., 0],
            log_extent[..., 1],
            log_extent[..., 2],
            torch.sin(theta_rel),
            torch.cos(theta_rel),
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(psi),
            torch.cos(psi),
            center_x / image_width,
            center_y / image_height,
            box_width / image_width,
            box_height / image_height,
            torch.log(distance.clamp_min(_EPS)),
        ],
        dim=-1,
    )
    return torch.cat([one_hot, geometry], dim=-1)


def camera_view_geometry(
    boxes3d_world: np.ndarray,
    world2cam: np.ndarray,
):
    """Compute relative yaw, elevation, bearing, and distance for 3D boxes.

    Boxes use ``[x, y, z, w, l, h, yaw]`` with yaw around world +Z.
    ``world2cam`` maps world homogeneous coordinates into an OpenCV camera
    frame (+Z forward, +X right, +Y down).
    """
    boxes3d_world = np.asarray(boxes3d_world, dtype=np.float64).reshape(-1, 7)
    world2cam = np.asarray(world2cam, dtype=np.float64).reshape(4, 4)

    centers_world = boxes3d_world[:, :3]
    yaw_world = boxes3d_world[:, 6]
    homogeneous_centers = np.concatenate(
        [centers_world, np.ones((centers_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    centers_camera = (homogeneous_centers @ world2cam.T)[:, :3]
    x_camera, y_camera, z_camera = centers_camera.T

    distance = np.linalg.norm(centers_camera, axis=1)
    horizontal_distance = np.sqrt(x_camera * x_camera + z_camera * z_camera)
    bearing = np.arctan2(x_camera, z_camera)
    elevation = np.arctan2(-y_camera, horizontal_distance)

    forward_world = world2cam[2, :3]
    camera_yaw_world = np.arctan2(forward_world[1], forward_world[0])
    relative_yaw = yaw_world - camera_yaw_world
    return relative_yaw, elevation, bearing, distance


class LooseToTightMLP(nn.Module):
    """Map geometric features to per-edge loose-to-tight scale factors."""

    def __init__(
        self,
        num_classes: int = len(WAREHOUSE_V4_CLASSES),
        hidden_dim: int = 64,
        shrink_min: float = 0.5,
        shrink_max: float = 1.0,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the correction network."""
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if shrink_max <= shrink_min:
            raise ValueError("shrink_max must be greater than shrink_min")

        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.shrink_min = float(shrink_min)
        self.shrink_max = float(shrink_max)
        self.class_names = list(class_names) if class_names is not None else None
        self.in_dim = self.num_classes + NUM_GEOM_FEATURES

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return side scales with shape ``(..., 4)`` in the configured range."""
        scales = torch.sigmoid(self.net(features))
        return scales * (self.shrink_max - self.shrink_min) + self.shrink_min

    def featurize(
        self,
        class_id: torch.Tensor,
        extent_wlh: torch.Tensor,
        theta_rel: torch.Tensor,
        phi: torch.Tensor,
        psi: torch.Tensor,
        loose_xyxy: torch.Tensor,
        img_wh: Union[torch.Tensor, Sequence[float]],
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """Build features using this model's class count."""
        return assemble_features(
            class_id,
            extent_wlh,
            theta_rel,
            phi,
            psi,
            loose_xyxy,
            img_wh,
            distance,
            num_classes=self.num_classes,
        )

    @torch.no_grad()
    def predict_shrinkage(
        self,
        features: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Run inference, optionally returning an explicitly detached tensor."""
        scales = self.forward(features)
        return scales.detach() if detach else scales

    @staticmethod
    def apply_to_loose(
        loose_xyxy: torch.Tensor,
        side_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Pull each edge of a loose box toward the loose-box center."""
        x_min, y_min, x_max, y_max = loose_xyxy.unbind(dim=-1)
        scale_left, scale_right, scale_top, scale_bottom = side_scales.unbind(dim=-1)
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        new_x_min = center_x - scale_left * (center_x - x_min)
        new_x_max = center_x + scale_right * (x_max - center_x)
        new_y_min = center_y - scale_top * (center_y - y_min)
        new_y_max = center_y + scale_bottom * (y_max - center_y)
        return torch.stack(
            [new_x_min, new_y_min, new_x_max, new_y_max],
            dim=-1,
        )

    def shrinkage_target(
        self,
        loose_xyxy: torch.Tensor,
        tight_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        """Recover bounded side scales that map ``loose_xyxy`` to ``tight_xyxy``."""
        x_min, y_min, x_max, y_max = loose_xyxy.unbind(dim=-1)
        tight_x_min, tight_y_min, tight_x_max, tight_y_max = tight_xyxy.unbind(dim=-1)
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        scale_left = (center_x - tight_x_min) / (center_x - x_min).clamp_min(_EPS)
        scale_right = (tight_x_max - center_x) / (x_max - center_x).clamp_min(_EPS)
        scale_top = (center_y - tight_y_min) / (center_y - y_min).clamp_min(_EPS)
        scale_bottom = (tight_y_max - center_y) / (y_max - center_y).clamp_min(_EPS)
        scales = torch.stack(
            [scale_left, scale_right, scale_top, scale_bottom],
            dim=-1,
        )
        return scales.clamp(self.shrink_min, self.shrink_max)

    def freeze(self) -> "LooseToTightMLP":
        """Disable parameter gradients and switch the model to evaluation mode."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @property
    def hparams(self) -> dict:
        """Return architecture metadata needed to reload a checkpoint."""
        return {
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "shrink_min": self.shrink_min,
            "shrink_max": self.shrink_max,
            "class_names": self.class_names,
        }

    def save(self, path: str, extra_meta: Optional[dict] = None) -> None:
        """Save model weights and architecture metadata."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": self.hparams,
                "meta": extra_meta or {},
                "format": "loose_to_tight_mlp/v1",
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Union[str, torch.device] = "cpu",
        freeze: bool = True,
    ) -> "LooseToTightMLP":
        """Load a checkpoint produced by :meth:`save`."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                f"{path} is not a LooseToTightMLP checkpoint saved with save()."
            )
        model = cls(**checkpoint.get("hparams", {}))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(map_location)
        return model.freeze() if freeze else model
