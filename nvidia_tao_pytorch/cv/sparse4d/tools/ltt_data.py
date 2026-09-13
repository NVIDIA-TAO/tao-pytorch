# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, pickle-free data contract for Loose-to-Tight artifact tools.

The cache intentionally stores raw geometry instead of model features.  This
keeps extraction independent from feature revisions while allowing the trainer
to reuse :class:`LooseToTightMLP`'s canonical ``featurize`` implementation.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    WAREHOUSE_V4_CLASSES,
)


SCHEMA_VERSION = "ltt_data/v2"
PACK_LEN = 18
SL_EXTENT = slice(0, 3)
I_THETA = 3
I_PHI = 4
I_PSI = 5
I_DIST = 6
SL_LOOSE = slice(7, 11)
SL_TIGHT = slice(11, 15)
SL_IMGWH = slice(15, 17)
I_VIS = 17


def normalize_npz_path(path: os.PathLike | str) -> Path:
    """Return the path NumPy will use, explicitly adding ``.npz`` if needed."""
    output = Path(path).expanduser()
    return output if str(output).endswith(".npz") else Path(f"{output}.npz")


def encode_meta(meta: dict) -> np.ndarray:
    """Encode JSON metadata as a safe ``uint8`` NPZ array."""
    return np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )


def decode_meta(value: np.ndarray) -> dict:
    """Decode metadata written by :func:`encode_meta`."""
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("NPZ _meta must be a one-dimensional uint8 JSON buffer")
    result = json.loads(array.tobytes().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("NPZ _meta JSON must decode to an object")
    return result


def load_class_taxonomy(
    config_path: Optional[os.PathLike | str] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Return class names and a name/subclass-to-ID map.

    ``config_path`` may point at a Python class config defining ``CLASS_LIST``
    and optionally ``SUB_CLASS_DICT``.  Without it, the model's built-in
    warehouse-v4 taxonomy is used.
    """
    if config_path is None:
        names = list(WAREHOUSE_V4_CLASSES)
        return names, {name: index for index, name in enumerate(names)}

    path = Path(config_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(
        f"_sparse4d_classes_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import class taxonomy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CLASS_LIST"):
        raise ValueError(f"{path} does not define CLASS_LIST")
    names = [str(name) for name in module.CLASS_LIST]
    if not names or len(names) != len(set(names)):
        raise ValueError("CLASS_LIST must be non-empty and contain unique names")
    name_to_id = {name: index for index, name in enumerate(names)}
    for parent, subclasses in getattr(module, "SUB_CLASS_DICT", {}).items():
        if parent in name_to_id:
            for subclass in subclasses:
                name_to_id.setdefault(str(subclass), name_to_id[parent])
    return names, name_to_id


def pack(
    extent_wlh: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    dist: np.ndarray,
    loose_xyxy: np.ndarray,
    tight_xyxy: np.ndarray,
    img_wh: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """Stack raw per-sample values into an ``(N, 18)`` float32 matrix."""
    theta = np.asarray(theta).reshape(-1)
    count = len(theta)
    values = {
        "extent_wlh": (np.asarray(extent_wlh), (count, 3)),
        "phi": (np.asarray(phi), (count,)),
        "psi": (np.asarray(psi), (count,)),
        "dist": (np.asarray(dist), (count,)),
        "loose_xyxy": (np.asarray(loose_xyxy), (count, 4)),
        "tight_xyxy": (np.asarray(tight_xyxy), (count, 4)),
        "img_wh": (np.asarray(img_wh), (count, 2)),
        "visibility": (np.asarray(visibility), (count,)),
    }
    for name, (value, expected) in values.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")

    output = np.empty((count, PACK_LEN), dtype=np.float32)
    output[:, SL_EXTENT] = values["extent_wlh"][0]
    output[:, I_THETA] = theta
    output[:, I_PHI] = values["phi"][0]
    output[:, I_PSI] = values["psi"][0]
    output[:, I_DIST] = values["dist"][0]
    output[:, SL_LOOSE] = values["loose_xyxy"][0]
    output[:, SL_TIGHT] = values["tight_xyxy"][0]
    output[:, SL_IMGWH] = values["img_wh"][0]
    output[:, I_VIS] = values["visibility"][0]
    if not np.isfinite(output).all():
        raise ValueError("packed Loose-to-Tight data contains non-finite values")
    return output


def project_cuboid_aabb(
    box7: Sequence[float],
    world2cam: np.ndarray,
    intrinsic: np.ndarray,
    img_wh: Optional[Sequence[float]] = None,
    origin_z: float = 0.5,
    eps: float = 0.1,
) -> Optional[np.ndarray]:
    """Project ``[x,y,z,w,l,h,yaw]`` to an image-clipped xyxy AABB."""
    x, y, z, width, length, height, yaw = (float(value) for value in box7)
    z_min, z_max = (-height / 2.0, height / 2.0) if origin_z == 0.5 else (0.0, height)
    corners = np.asarray(
        [
            (x_sign, y_sign, z_offset)
            for x_sign in (-width / 2.0, width / 2.0)
            for y_sign in (-length / 2.0, length / 2.0)
            for z_offset in (z_min, z_max)
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    world = corners @ rotation.T + np.asarray([x, y, z])
    transform = np.asarray(world2cam, dtype=np.float64)
    if transform.size == 12:
        transform = np.vstack([transform.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    else:
        transform = transform.reshape(4, 4)
    camera = (
        np.concatenate([world, np.ones((8, 1), dtype=np.float64)], axis=1) @ transform.T
    )[:, :3]
    if np.any(camera[:, 2] <= eps):
        return None
    pixels = camera @ np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).T
    uv = pixels[:, :2] / pixels[:, 2:3]
    box = np.asarray(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        dtype=np.float32,
    )
    if img_wh is not None:
        image_width, image_height = (float(value) for value in img_wh)
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, image_width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, image_height)
    return box


class ClassBalancedReservoir:
    """Memory-bounded uniform reservoir with an independent cap per class."""

    def __init__(
        self,
        num_classes: int,
        max_per_class: Optional[int],
        seed: int = 0,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.max_per_class = (
            int(max_per_class) if max_per_class and max_per_class > 0 else None
        )
        self.num_classes = int(num_classes)
        self.rng = np.random.default_rng(seed)
        self.heaps: List[list] = [[] for _ in range(self.num_classes)]
        self.counts = [0] * self.num_classes
        self._tie = 0

    def add_batch(
        self,
        class_ids: np.ndarray,
        packed: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        """Add samples and their source-frame group IDs."""
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        packed = np.asarray(packed, dtype=np.float32)
        group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if packed.shape != (len(class_ids), PACK_LEN):
            raise ValueError(
                f"packed must have shape {(len(class_ids), PACK_LEN)}, got {packed.shape}"
            )
        if group_ids.shape != class_ids.shape:
            raise ValueError("group_ids must contain one value per packed row")
        keys = self.rng.random(len(class_ids))
        for index, class_id in enumerate(class_ids):
            if class_id < 0 or class_id >= self.num_classes:
                continue
            self.counts[class_id] += 1
            item = (
                float(keys[index]),
                self._tie,
                packed[index].copy(),
                int(group_ids[index]),
            )
            self._tie += 1
            heap = self.heaps[class_id]
            if self.max_per_class is None:
                heap.append(item)
            elif len(heap) < self.max_per_class:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled packed rows, class IDs, and source-frame group IDs."""
        rows, class_ids, group_ids = [], [], []
        for class_id, heap in enumerate(self.heaps):
            for _, _, row, group_id in heap:
                rows.append(row)
                class_ids.append(class_id)
                group_ids.append(group_id)
        return (
            np.asarray(rows, dtype=np.float32).reshape(-1, PACK_LEN),
            np.asarray(class_ids, dtype=np.int16),
            np.asarray(group_ids, dtype=np.int64),
        )

    def kept_per_class(self) -> List[int]:
        """Return number retained for each class."""
        return [len(heap) for heap in self.heaps]

    def seen_per_class(self) -> List[int]:
        """Return number observed for each class."""
        return list(self.counts)


def save_cache(
    path: os.PathLike | str,
    packed: np.ndarray,
    class_id: np.ndarray,
    group_id: np.ndarray,
    meta: Optional[dict] = None,
) -> None:
    """Write a compressed, numeric-only ``ltt_data/v2`` NPZ cache."""
    packed = np.asarray(packed, dtype=np.float32)
    class_id = np.asarray(class_id, dtype=np.int16).reshape(-1)
    group_id = np.asarray(group_id, dtype=np.int64).reshape(-1)
    if packed.shape != (len(class_id), PACK_LEN):
        raise ValueError(
            f"packed must have shape {(len(class_id), PACK_LEN)}, got {packed.shape}"
        )
    if group_id.shape != class_id.shape:
        raise ValueError("group_id must contain one value per packed row")
    if np.any(group_id < 0):
        raise ValueError("group_id values must be non-negative")
    metadata = dict(meta or {})
    metadata.setdefault("schema_version", SCHEMA_VERSION)
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    output = normalize_npz_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        packed=packed,
        class_id=class_id,
        group_id=group_id,
        _meta=encode_meta(metadata),
    )


def load_cache(
    paths: os.PathLike | str | Sequence[os.PathLike | str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Load and concatenate validated caches with shard-unique group IDs."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    packed_parts, class_parts, group_parts, metadata = [], [], [], []
    group_offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            required = {"packed", "class_id", "group_id", "_meta"}
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(f"{path} is missing NPZ keys: {sorted(missing)}")
            packed = np.asarray(cache["packed"], dtype=np.float32)
            class_id = np.asarray(cache["class_id"], dtype=np.int64)
            raw_group_id = np.asarray(cache["group_id"])
            meta = decode_meta(cache["_meta"])
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has unsupported schema {meta.get('schema_version')!r}"
            )
        if packed.ndim != 2 or packed.shape[1] != PACK_LEN:
            raise ValueError(f"{path}: packed must have shape (N, {PACK_LEN})")
        if class_id.shape != (packed.shape[0],):
            raise ValueError(f"{path}: class_id length does not match packed rows")
        if raw_group_id.shape != (packed.shape[0],):
            raise ValueError(f"{path}: group_id length does not match packed rows")
        if raw_group_id.dtype.kind not in "iu":
            raise ValueError(f"{path}: group_id must have an integer dtype")
        if not np.isfinite(packed).all():
            raise ValueError(f"{path}: packed contains non-finite values")
        unique_groups, group_inverse = np.unique(
            raw_group_id.astype(np.int64), return_inverse=True
        )
        group_id = group_inverse.astype(np.int64) + group_offset
        group_offset += len(unique_groups)
        packed_parts.append(packed)
        class_parts.append(class_id)
        group_parts.append(group_id)
        metadata.append(meta)
    if not packed_parts:
        raise ValueError("No Loose-to-Tight cache paths were provided")
    return (
        np.concatenate(packed_parts, axis=0),
        np.concatenate(class_parts, axis=0),
        np.concatenate(group_parts, axis=0),
        metadata,
    )
