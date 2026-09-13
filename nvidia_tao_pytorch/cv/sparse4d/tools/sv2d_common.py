# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared taxonomy, manifest, and virtual-camera geometry for SV2D tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    WAREHOUSE_V4_CLASSES,
)


CLASS_NAMES = list(WAREHOUSE_V4_CLASSES)
_NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
COCO_NAME_TO_WH = {name: _NAME_TO_ID[name] for name in CLASS_NAMES}
DROP_NAMES = {"pallet"}

CANON_W = 1920
CANON_H = 1080

VCAM = {
    "width": CANON_W,
    "height": CANON_H,
    "fov_h_deg": 70.0,
    "cam_xyz": (-12.0, 0.0, 6.0),
    "target_xyz": (18.0, 0.0, 0.87),
    "world_up": (0.0, 0.0, 1.0),
}


def map_coco_name(
    name: str,
    name_to_id: Optional[dict] = None,
) -> Optional[int]:
    """Map a COCO category name through the selected taxonomy."""
    if name in DROP_NAMES:
        return None
    mapping = COCO_NAME_TO_WH if name_to_id is None else name_to_id
    return mapping.get(name)


def validate_scene_name(scene_name: str) -> str:
    """Validate a scene name used as an artifact filename component."""
    if not isinstance(scene_name, str) or not scene_name.strip():
        raise ValueError("SV2D scene_name must be a non-empty string")
    if scene_name in {".", ".."} or any(
        separator in scene_name for separator in ("/", "\\")
    ):
        raise ValueError(
            "SV2D scene_name must be a plain filename component without "
            "path separators or traversal"
        )
    if "\x00" in scene_name:
        raise ValueError("SV2D scene_name must not contain a NUL character")
    return scene_name


def build_intrinsic(width: int, height: int, fov_h_deg: float) -> np.ndarray:
    """Construct a centred pinhole intrinsic matrix from horizontal FOV."""
    if width <= 0 or height <= 0 or not 0.0 < fov_h_deg < 180.0:
        raise ValueError("width/height must be positive and fov_h_deg in (0, 180)")
    focal = (width / 2.0) / np.tan(np.deg2rad(fov_h_deg) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_world2cam(
    camera_xyz: Sequence[float],
    target_xyz: Sequence[float],
    world_up: Sequence[float],
) -> np.ndarray:
    """Build world-to-OpenCV-camera transform (+Z forward, +Y down)."""
    camera = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    forward = target - camera
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError("Virtual camera target must differ from camera position")
    forward /= forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        raise ValueError("world_up must not be parallel to the camera view")
    right /= right_norm
    down = np.cross(forward, right)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.stack([right, down, forward], axis=1)
    camera_to_world[:3, 3] = camera
    return np.linalg.inv(camera_to_world)


def build_virtual_camera(vcam: Optional[dict] = None):
    """Return intrinsic and world-to-camera matrices for one virtual camera."""
    values = dict(VCAM if vcam is None else vcam)
    intrinsic = build_intrinsic(
        int(values["width"]), int(values["height"]), float(values["fov_h_deg"])
    )
    world2cam = build_world2cam(
        values["cam_xyz"], values["target_xyz"], values["world_up"]
    )
    return intrinsic, world2cam


def project_world_points(
    intrinsic: np.ndarray,
    world2cam: np.ndarray,
    points_xyz: np.ndarray,
):
    """Project world points using TAO Sparse4D's ``K_pad @ world2cam`` convention."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    intrinsic_pad = np.eye(4, dtype=np.float64)
    intrinsic_pad[:3, :3] = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    projection = intrinsic_pad @ np.asarray(world2cam, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    projected = homogeneous @ projection.T
    depth = projected[:, 2].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv, depth


def strip_h5_uri(file_name: str) -> str:
    """Convert ``h5://<tag>:<name>`` to ``<name>`` without the scheme bug."""
    name = str(file_name)
    if name.startswith("h5://"):
        name = name[len("h5://"):]
        if ":" in name:
            name = name.split(":", 1)[1]
    return name.lstrip("/")


def resolve_image_ref(dataset: dict, file_name: str):
    """Resolve a COCO image name to a plain path or ``(HDF5, key)`` tuple."""
    name = strip_h5_uri(file_name)
    if dataset["kind"] == "h5":
        return (os.fspath(dataset["h5_path"]), f"rgb/{name}")
    if dataset["kind"] != "file":
        raise ValueError(f"Unsupported SV2D dataset kind {dataset['kind']!r}")
    return os.path.join(os.fspath(dataset["image_root"]), name)


def load_dataset_manifest(path: Optional[os.PathLike | str]) -> list:
    """Load a user-supplied JSON manifest of portable COCO sources."""
    if path is None:
        raise ValueError("An SV2D dataset manifest path is required")
    manifest_path = Path(path).expanduser().resolve()
    with open(manifest_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    datasets = document.get("datasets") if isinstance(document, dict) else document
    if not isinstance(datasets, list):
        raise ValueError("SV2D manifest must be a list or {'datasets': [...]} object")
    if not datasets:
        raise ValueError("SV2D manifest must contain at least one dataset")
    result = []
    for raw in datasets:
        if not isinstance(raw, dict):
            raise ValueError("Each SV2D manifest entry must be an object")
        dataset = dict(raw)
        required = {"name", "scene_name", "coco", "kind"}
        missing = required.difference(dataset)
        if missing:
            raise ValueError(f"SV2D manifest entry is missing {sorted(missing)}")
        dataset["scene_name"] = validate_scene_name(dataset["scene_name"])
        kind = dataset["kind"]
        if kind not in {"file", "h5"}:
            raise ValueError(
                f"SV2D dataset {dataset['name']!r} has unsupported kind {kind!r}"
            )
        path_keys = ["coco"] + (["h5_path"] if kind == "h5" else ["image_root"])
        for key in path_keys:
            if key not in dataset:
                raise ValueError(f"SV2D dataset {dataset['name']!r} is missing {key}")
            value = Path(dataset[key]).expanduser()
            if not value.is_absolute():
                value = manifest_path.parent / value
            dataset[key] = str(value.resolve())
        dataset.setdefault("weight", 1)
        result.append(dataset)
    names = [dataset["name"] for dataset in result]
    if len(names) != len(set(names)):
        raise ValueError("SV2D manifest dataset names must be unique")
    scene_names = [dataset["scene_name"] for dataset in result]
    if len(scene_names) != len(set(scene_names)):
        raise ValueError("SV2D manifest scene_name values must be unique")
    return result
