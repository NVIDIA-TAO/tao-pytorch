# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert RT-DETR KITTI label archives to Sparse4D pseudo-label NPZs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tarfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_data


SCHEMA_VERSION = "ltt_rtdetr2d/v1"

# Bare storage pallets are deliberately not pallet-truck vehicles.
RTDETR_TO_WAREHOUSE = {
    "person": "person",
    "gr1_t2": "gr1_t2",
    "agility_digit": "agility_digit",
    "nova_carter": "nova_carter",
    "transporter": "transporter",
    "forklift": "forklift",
    "pallet_truck": "pallet_truck",
}

_FRAME_RE = re.compile(r"(\d+)")


def frame_id_from_name(member_name: str) -> Optional[int]:
    """Use the final integer in a KITTI label filename as its frame ID."""
    numbers = _FRAME_RE.findall(Path(member_name).stem)
    return int(numbers[-1]) if numbers else None


def parse_camera_archive(
    tar_path: os.PathLike | str,
    camera_index: int,
    name_to_id: Dict[str, int],
    *,
    confidence_threshold: float = 0.4,
    frame_stride: int = 1,
    max_frames: int = 0,
    raw_counts: Optional[dict] = None,
) -> Tuple[dict, int]:
    """Parse one camera archive without extracting untrusted tar paths."""
    raw_counts = raw_counts if raw_counts is not None else {}
    frame_ids, class_ids, boxes, scores = [], [], [], []
    seen_frames = set()
    stride = max(1, int(frame_stride))
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            frame_id = frame_id_from_name(member.name)
            if frame_id is None or frame_id % stride:
                continue
            if (
                max_frames and
                len(seen_frames) >= max_frames and
                frame_id not in seen_frames
            ):
                continue
            seen_frames.add(frame_id)
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for line in stream.read().decode("utf-8", "ignore").splitlines():
                parts = line.split()
                if len(parts) < 9:
                    continue
                detector_name = parts[0]
                raw_counts[detector_name] = raw_counts.get(detector_name, 0) + 1
                warehouse_name = RTDETR_TO_WAREHOUSE.get(detector_name)
                class_id = name_to_id.get(warehouse_name) if warehouse_name else None
                if class_id is None:
                    continue
                try:
                    x_min, y_min, x_max, y_max = (
                        float(parts[index]) for index in range(4, 8)
                    )
                    confidence = float(parts[-1])
                except ValueError:
                    continue
                if not np.isfinite([x_min, y_min, x_max, y_max, confidence]).all():
                    continue
                if (
                    confidence < confidence_threshold or
                    x_max - x_min <= 1.0 or
                    y_max - y_min <= 1.0
                ):
                    continue
                frame_ids.append(frame_id)
                class_ids.append(class_id)
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(confidence)
    count = len(frame_ids)
    valid_frame_ids = np.asarray(sorted(seen_frames), dtype=np.int32)
    return (
        {
            "frame_id": np.asarray(frame_ids, dtype=np.int32),
            "cam": np.full(count, camera_index, dtype=np.int16),
            "class_id": np.asarray(class_ids, dtype=np.int16),
            "box": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "score": np.asarray(scores, dtype=np.float32),
            # Detection rows cannot represent a processed frame whose label file
            # is empty (or whose detections are all filtered). Keep an explicit
            # frame/camera join so the trainer can distinguish that valid
            # background-only frame from a missing cache join.
            "valid_frame_id": valid_frame_ids,
            "valid_cam": np.full(len(valid_frame_ids), camera_index, dtype=np.int16),
        },
        len(seen_frames),
    )


def _concat(parts: list, key: str, empty: np.ndarray) -> np.ndarray:
    return np.concatenate([part[key] for part in parts]) if parts else empty


def build_cache(
    rtdetr_dir: os.PathLike | str,
    output_path: os.PathLike | str,
    class_names: Sequence[str],
    name_to_id: Dict[str, int],
    *,
    camera_map: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.4,
    frame_stride: int = 1,
    max_frames_per_camera: int = 0,
    scene_name: Optional[str] = None,
) -> dict:
    """Build one safe ``ltt_rtdetr2d/v1`` scene cache."""
    rtdetr_dir = Path(rtdetr_dir).expanduser().resolve()
    label_dirs = sorted(
        path
        for path in rtdetr_dir.iterdir()
        if path.is_dir() and (path / "labels.tar.gz").is_file()
    )
    if not label_dirs:
        raise ValueError(f"No <camera>/labels.tar.gz found under {rtdetr_dir}")
    camera_map = dict(camera_map or {})
    camera_names = [camera_map.get(path.name, path.name) for path in label_dirs]
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("camera_map produces duplicate camera names")

    raw_counts: dict = {}
    parts, frame_counts = [], {}
    for camera_index, label_dir in enumerate(label_dirs):
        columns, num_frames = parse_camera_archive(
            label_dir / "labels.tar.gz",
            camera_index,
            name_to_id,
            confidence_threshold=confidence_threshold,
            frame_stride=frame_stride,
            max_frames=max_frames_per_camera,
            raw_counts=raw_counts,
        )
        parts.append(columns)
        frame_counts[camera_names[camera_index]] = num_frames
    columns = {
        "frame_id": _concat(parts, "frame_id", np.zeros(0, dtype=np.int32)),
        "cam": _concat(parts, "cam", np.zeros(0, dtype=np.int16)),
        "class_id": _concat(parts, "class_id", np.zeros(0, dtype=np.int16)),
        "box": _concat(parts, "box", np.zeros((0, 4), dtype=np.float32)),
        "score": _concat(parts, "score", np.zeros(0, dtype=np.float32)),
    }
    if len(columns["frame_id"]):
        order = np.lexsort((columns["class_id"], columns["cam"], columns["frame_id"]))
        columns = {key: value[order] for key, value in columns.items()}
    valid_frame_ids = _concat(parts, "valid_frame_id", np.zeros(0, dtype=np.int32))
    valid_cameras = _concat(parts, "valid_cam", np.zeros(0, dtype=np.int16))
    if len(valid_frame_ids):
        valid_order = np.lexsort((valid_cameras, valid_frame_ids))
        valid_frame_ids = valid_frame_ids[valid_order]
        valid_cameras = valid_cameras[valid_order]
    columns["valid_frame_id"] = valid_frame_ids
    columns["valid_cam"] = valid_cameras
    scene = scene_name or rtdetr_dir.parent.name
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "cam_names": camera_names,
        "class_names": list(class_names),
        "conf_thr": float(confidence_threshold),
        "frame_stride": int(frame_stride),
        "num_rows": int(len(columns["frame_id"])),
        "num_valid_frame_cameras": int(len(valid_frame_ids)),
        "source": "RT-DETR KITTI labels.tar.gz",
        "class_map": RTDETR_TO_WAREHOUSE,
        "raw_class_counts": raw_counts,
        "frames_per_camera": frame_counts,
    }
    output = ltt_data.normalize_npz_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, _meta=ltt_data.encode_meta(metadata), **columns)
    return metadata


def parse_camera_map(value: Optional[str]) -> dict:
    """Parse ``label-dir=calibration-name`` comma pairs."""
    if not value:
        return {}
    mapping = {}
    for pair in value.split(","):
        if pair.count("=") != 1:
            raise ValueError(f"Invalid camera mapping {pair!r}")
        source, target = (part.strip() for part in pair.split("=", 1))
        if not source or not target:
            raise ValueError(f"Invalid camera mapping {pair!r}")
        mapping[source] = target
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtdetr-dir", required=True)
    parser.add_argument("--class-config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-name")
    parser.add_argument("--cam-map")
    parser.add_argument("--conf-thr", type=float, default=0.4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-cam", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    output_path = ltt_data.normalize_npz_path(args.out)
    class_names, name_to_id = ltt_data.load_class_taxonomy(args.class_config)
    metadata = build_cache(
        args.rtdetr_dir,
        output_path,
        class_names,
        name_to_id,
        camera_map=parse_camera_map(args.cam_map),
        confidence_threshold=args.conf_thr,
        frame_stride=args.frame_stride,
        max_frames_per_camera=args.max_frames_per_cam,
        scene_name=args.scene_name,
    )
    print(
        f"[saved] {output_path}: {metadata['num_rows']} detections from "
        f"{len(metadata['cam_names'])} cameras"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
