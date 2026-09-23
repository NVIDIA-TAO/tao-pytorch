# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a compact Loose-to-Tight MLP cache from raw Sparse4D scenes.

The extractor reads global-camera ground truth directly, avoiding duplicated
camera groups in training PKLs.  It supports NVSchema ``calibration.json`` and
legacy ``calibration_bevformer.json`` without importing MMDetection, MMCV, or
``spatialai_data_utils``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import ijson as _ijson
except ImportError:
    _ijson = None

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    camera_view_geometry,
)
from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_data


def bbox_area(box: Sequence[float]) -> float:
    """Return non-negative xyxy box area."""
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def iter_gt_frames(
    scene_dir: os.PathLike | str,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> Iterator[Tuple[int, list]]:
    """Yield frames from per-frame or monolithic ground truth.

    Per-frame files are sorted numerically. Monolithic JSON preserves source
    order with either the optional streaming parser or the standard-library
    fallback, so installed dependencies cannot change sampling results.
    """
    scene = Path(scene_dir).expanduser()
    stride = max(1, int(frame_stride))
    limit = max(0, int(max_frames))
    per_frame_dir = scene / "ground_truth_final"
    emitted = 0
    if per_frame_dir.is_dir():
        indexed = {}
        for path in sorted(per_frame_dir.glob("ground_truth_*.json")):
            try:
                frame_id = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            prior_path = indexed.get(frame_id)
            if prior_path is not None:
                raise ValueError(
                    f"Duplicate normalized frame ID {frame_id} in "
                    f"{per_frame_dir}: {prior_path.name!r} and {path.name!r}"
                )
            indexed[frame_id] = path
        for frame_id, path in sorted(indexed.items()):
            if frame_id % stride:
                continue
            with path.open("r", encoding="utf-8") as stream:
                annotations = json.load(stream)
            if not isinstance(annotations, list):
                raise ValueError(f"Ground-truth frame must contain a list: {path}")
            yield frame_id, annotations
            emitted += 1
            if limit and emitted >= limit:
                return
        return

    ground_truth_path = scene / "ground_truth.json"
    if not ground_truth_path.is_file():
        return

    def selected_frames(items):
        """Yield validated, strided frames from key/annotation pairs."""
        emitted_count = 0
        normalized_keys = {}
        for key, annotations in items:
            try:
                frame_id = int(key)
            except (TypeError, ValueError):
                continue
            key_text = str(key)
            prior_key = normalized_keys.get(frame_id)
            if prior_key is not None:
                raise ValueError(
                    f"Duplicate normalized frame ID {frame_id} in "
                    f"{ground_truth_path}: {prior_key!r} and {key_text!r}"
                )
            normalized_keys[frame_id] = key_text
            if frame_id % stride:
                continue
            if limit and emitted_count >= limit:
                continue
            if not isinstance(annotations, list):
                raise ValueError(
                    f"Ground-truth frame {frame_id!r} must contain a list"
                )
            yield frame_id, annotations
            emitted_count += 1

    if _ijson is None:
        with ground_truth_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(
                "Monolithic ground truth must contain an object: "
                f"{ground_truth_path}"
            )
        yield from selected_frames(document.items())
    else:
        with ground_truth_path.open("rb") as stream:
            yield from selected_frames(_ijson.kvitems(stream, ""))


def has_ground_truth(scene_dir: os.PathLike | str) -> bool:
    """Return whether a scene contains supported ground-truth storage."""
    scene_dir = os.fspath(scene_dir)
    return os.path.isdir(scene_dir) and (
        os.path.isdir(os.path.join(scene_dir, "ground_truth_final")) or
        os.path.isfile(os.path.join(scene_dir, "ground_truth.json"))
    )


def scene_from_split_line(line: str, dedup_regex: str = r"^CT[\w.]+?__") -> str:
    """Extract and de-duplicate a scene name from a Sparse4D split row."""
    scene = os.path.basename(line.split()[0]).split("+")[0]
    for suffix in (
        "_infos_train.pkl",
        "_infos_test.pkl",
        "_infos_val.pkl",
        "_infos.pkl",
        ".pkl",
    ):
        if scene.endswith(suffix):
            scene = scene[: -len(suffix)]
            break
    return re.sub(dedup_regex, "", scene) if dedup_regex else scene


def resolve_scene_dir(
    data_root: Optional[os.PathLike | str], scene: os.PathLike | str
) -> Optional[str]:
    """Resolve a bare scene under a root, including one subset-directory level."""
    scene = os.fspath(scene)
    if os.path.isabs(scene):
        return scene if has_ground_truth(scene) else None
    if data_root is None:
        return scene if has_ground_truth(scene) else None
    direct = os.path.join(os.fspath(data_root), scene)
    if has_ground_truth(direct):
        return direct
    for candidate in sorted(glob.glob(os.path.join(os.fspath(data_root), "*", scene))):
        if has_ground_truth(candidate):
            return candidate
    return None


def resolve_scene_paths(
    data_root: Optional[os.PathLike | str] = None,
    scenes: Optional[Sequence[str]] = None,
    scenes_file: Optional[os.PathLike | str] = None,
    scene_dirs: Optional[Sequence[os.PathLike | str]] = None,
    train_split: Optional[os.PathLike | str] = None,
    dedup_regex: str = r"^CT[\w.]+?__",
) -> List[str]:
    """Resolve all CLI scene-selection forms to unique GT directories."""
    paths = [os.fspath(path) for path in (scene_dirs or [])]
    names = list(scenes or [])
    if scenes_file:
        with open(scenes_file, "r", encoding="utf-8") as stream:
            names.extend(
                line.strip()
                for line in stream
                if line.strip() and not line.lstrip().startswith("#")
            )
    if train_split:
        split_scenes = set()
        with open(train_split, "r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and not line.lstrip().startswith("#"):
                    split_scenes.add(scene_from_split_line(line, dedup_regex))
        names.extend(sorted(split_scenes))
    for name in names:
        resolved = resolve_scene_dir(data_root, name)
        if resolved:
            paths.append(resolved)
    if not paths and data_root and os.path.isdir(data_root):
        paths.extend(
            os.path.join(os.fspath(data_root), name)
            for name in sorted(os.listdir(data_root))
            if has_ground_truth(os.path.join(os.fspath(data_root), name))
        )
    unique = list(dict.fromkeys(os.path.abspath(path) for path in paths))
    if not unique:
        raise ValueError("No scenes with ground truth were resolved")
    return unique


def _attribute(sensor: dict, name: str):
    for attribute in sensor.get("attributes", []) or []:
        if attribute.get("name") == name:
            return attribute.get("value")
    return None


def _reshape_world2cam(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 12:
        return np.vstack([matrix.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    return matrix.reshape(4, 4)


def _decode_camera(name: str, value: dict) -> Optional[Tuple[str, dict]]:
    intrinsic = None
    world2cam = None
    for key in (
        "intrinsicMatrix",
        "intrinsic_matrix",
        "intrinsic matrix",
        "cam_intrinsic",
        "K",
    ):
        if key in value:
            intrinsic = np.asarray(value[key], dtype=np.float64).reshape(3, 3)
            break
    for key in (
        "extrinsicMatrix",
        "w2c_matrix",
        "projection matrix w2c",
        "sensor2world_transform",
        "world2cam",
    ):
        if key in value:
            world2cam = _reshape_world2cam(value[key])
            break
    if intrinsic is None or world2cam is None:
        camera_matrix = value.get("cameraMatrix")
        if camera_matrix is not None:
            raise ValueError(
                f"Camera {name!r}: cameraMatrix-only calibration is a projection, "
                "not a rigid transform. LTT geometry requires separate "
                "intrinsics and world-to-camera extrinsics."
            )
    if intrinsic is None or world2cam is None:
        return None

    image_size = value.get("image_size") or value.get("imageSize")
    if image_size is not None:
        image_width, image_height = (float(v) for v in image_size)
    else:
        width = value.get("width", _attribute(value, "frameWidth"))
        height = value.get("height", _attribute(value, "frameHeight"))
        image_width = float(width) if width else 2.0 * float(intrinsic[0, 2])
        image_height = float(height) if height else 2.0 * float(intrinsic[1, 2])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Camera {name!r} has an invalid image size")
    return name, {"K": intrinsic, "w2c": world2cam, "wh": (image_width, image_height)}


def load_scene_calibration(
    scene_dir: os.PathLike | str,
    calibration_file: Optional[os.PathLike | str] = None,
    calib_mode: str = "aic25",
) -> Dict[str, dict]:
    """Load all global cameras from NVSchema or legacy calibration JSON."""
    if calibration_file is None:
        filename = (
            "calibration_bevformer.json"
            if calib_mode == "aic24"
            else "calibration.json"
        )
        path = os.path.join(os.fspath(scene_dir), filename)
    else:
        path = os.fspath(calibration_file)
        if not os.path.isabs(path):
            path = os.path.join(os.fspath(scene_dir), path)
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)

    cameras: Dict[str, dict] = {}
    if isinstance(document, dict) and isinstance(document.get("sensors"), list):
        for sensor in document["sensors"]:
            if not isinstance(sensor, dict) or sensor.get("type", "camera") != "camera":
                continue
            name = str(sensor.get("id", ""))
            decoded = _decode_camera(name, sensor)
            if name and decoded:
                cameras[decoded[0]] = decoded[1]
    else:

        def visit(mapping: dict) -> None:
            for name, value in mapping.items():
                if not isinstance(value, dict):
                    continue
                decoded = _decode_camera(str(name), value)
                if decoded:
                    cameras.setdefault(decoded[0], decoded[1])
                else:
                    visit(value)

        if isinstance(document, dict):
            visit(document)
    if not cameras:
        raise ValueError(f"No supported camera calibration found in {path}")
    return cameras


def parse_objects(
    annotations: Sequence[dict],
    name_to_id: Dict[str, int],
    anno_version: str = "v0.1",
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Convert supported 3D annotations to class IDs and box7 arrays."""
    indices, classes, boxes = [], [], []
    for index, annotation in enumerate(annotations):
        class_id = name_to_id.get(str(annotation.get("object type", "")))
        if class_id is None:
            continue
        try:
            location = np.asarray(annotation["3d location"], dtype=np.float64).reshape(
                3
            )
            dimensions = np.asarray(
                annotation["3d bounding box scale"], dtype=np.float64
            ).reshape(3)
            yaw = float(annotation["3d bounding box rotation"][2])
        except (KeyError, TypeError, ValueError):
            continue
        if anno_version == "v0.0":
            yaw = -yaw
        indices.append(index)
        classes.append(class_id)
        boxes.append([*location, *dimensions, yaw])
    return (
        indices,
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float64).reshape(-1, 7),
    )


def extract_cache(
    scene_paths: Sequence[os.PathLike | str],
    output_path: os.PathLike | str,
    class_names: Sequence[str],
    name_to_id: Dict[str, int],
    *,
    calibration_file: Optional[os.PathLike | str] = None,
    calib_mode: str = "aic25",
    anno_version: str = "v0.1",
    frame_stride: int = 10,
    max_frames_per_scene: int = 0,
    min_visibility: float = 0.0,
    max_per_class: int = 50000,
    image_size_override: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Extract scenes and write a frame-grouped ``ltt_data/v2`` cache."""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0, 1]")
    reservoir = ltt_data.ClassBalancedReservoir(
        len(class_names), max_per_class, seed=seed
    )
    frames_used = 0
    resolved_scenes = []
    for scene_path_value in scene_paths:
        scene_path = os.path.abspath(os.fspath(scene_path_value))
        if not has_ground_truth(scene_path):
            raise ValueError(f"Scene has no supported ground truth: {scene_path}")
        cameras = load_scene_calibration(
            scene_path, calibration_file=calibration_file, calib_mode=calib_mode
        )
        resolved_scenes.append(Path(scene_path).name)
        for _, frame_annotations in iter_gt_frames(
            scene_path, frame_stride, max_frames_per_scene
        ):
            if not frame_annotations:
                continue
            annotation_indices, class_ids, boxes = parse_objects(
                frame_annotations, name_to_id, anno_version
            )
            if not annotation_indices:
                continue
            frames_used += 1
            per_camera = defaultdict(list)
            for local_index, annotation_index in enumerate(annotation_indices):
                annotation = frame_annotations[annotation_index]
                amodal_boxes = annotation.get("2d bounding box", {}) or {}
                visible_boxes = annotation.get("2d bounding box visible", {}) or {}
                for camera_name, amodal_value in amodal_boxes.items():
                    if camera_name not in cameras:
                        continue
                    amodal = np.asarray(amodal_value, dtype=np.float64).reshape(4)
                    amodal_area = bbox_area(amodal)
                    if amodal_area <= 1.0:
                        continue
                    visible = visible_boxes.get(camera_name)
                    visibility = (
                        np.clip(
                            bbox_area(np.asarray(visible).reshape(4)) / amodal_area,
                            0.0,
                            1.0,
                        )
                        if visible is not None
                        else 0.0
                    )
                    if visibility >= min_visibility:
                        per_camera[camera_name].append(
                            (local_index, amodal, float(visibility))
                        )

            for camera_name, items in per_camera.items():
                camera = cameras[camera_name]
                image_wh = image_size_override or camera["wh"]
                local_indices = np.asarray([item[0] for item in items], dtype=np.int64)
                subset = boxes[local_indices]
                theta, phi, psi, distance = camera_view_geometry(subset, camera["w2c"])
                valid_indices, loose_boxes = [], []
                for item_index, box in enumerate(subset):
                    loose = ltt_data.project_cuboid_aabb(
                        box, camera["w2c"], camera["K"], img_wh=image_wh
                    )
                    if loose is not None and bbox_area(loose) > 1.0:
                        valid_indices.append(item_index)
                        loose_boxes.append(loose)
                if not valid_indices:
                    continue
                valid = np.asarray(valid_indices, dtype=np.int64)
                packed = ltt_data.pack(
                    subset[valid, 3:6],
                    theta[valid],
                    phi[valid],
                    psi[valid],
                    distance[valid],
                    np.stack(loose_boxes),
                    np.stack([items[index][1] for index in valid]),
                    np.broadcast_to(
                        np.asarray(image_wh, dtype=np.float32), (len(valid), 2)
                    ),
                    np.asarray([items[index][2] for index in valid], dtype=np.float32),
                )
                reservoir.add_batch(
                    class_ids[local_indices][valid],
                    packed,
                    np.full(len(packed), frames_used - 1, dtype=np.int64),
                )

    packed, class_id, group_id = reservoir.to_arrays()
    if not len(class_id):
        raise ValueError("No Loose-to-Tight samples were extracted")
    metadata = {
        "class_names": list(class_names),
        "scenes": resolved_scenes,
        "box_wiring": ("input=cuboid_box1,target=amodal_box2,weight=visible/amodal"),
        "frame_stride": int(frame_stride),
        "min_visibility": float(min_visibility),
        "max_per_class": int(max_per_class),
        "anno_version": anno_version,
        "calib_mode": calib_mode,
        "img_wh_override": image_size_override,
        "seen_per_class": dict(zip(class_names, reservoir.seen_per_class())),
        "kept_per_class": dict(zip(class_names, reservoir.kept_per_class())),
        "num_samples": int(len(class_id)),
        "num_frames_used": int(frames_used),
    }
    metadata.update(extra_meta or {})
    ltt_data.save_cache(output_path, packed, class_id, group_id, metadata)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--scenes-file")
    parser.add_argument("--scene-dir", nargs="*")
    parser.add_argument("--train-split")
    parser.add_argument("--dedup-regex", default=r"^CT[\w.]+?__")
    parser.add_argument("--class-config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-mode", choices=["aic24", "aic25"], default="aic25")
    parser.add_argument("--calibration-file")
    parser.add_argument("--anno-version", choices=["v0.0", "v0.1"], default="v0.1")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--max-per-class", type=int, default=50000)
    parser.add_argument("--img-width", type=float)
    parser.add_argument("--img-height", type=float)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    output_path = ltt_data.normalize_npz_path(args.out)
    if (args.img_width is None) != (args.img_height is None):
        raise ValueError("--img-width and --img-height must be provided together")
    image_size = (
        (args.img_width, args.img_height) if args.img_width is not None else None
    )
    scene_paths = resolve_scene_paths(
        args.data_root,
        args.scenes,
        args.scenes_file,
        args.scene_dir,
        args.train_split,
        args.dedup_regex,
    )
    class_names, name_to_id = ltt_data.load_class_taxonomy(args.class_config)
    metadata = extract_cache(
        scene_paths,
        output_path,
        class_names,
        name_to_id,
        calibration_file=args.calibration_file,
        calib_mode=args.calib_mode,
        anno_version=args.anno_version,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        min_visibility=args.min_visibility,
        max_per_class=args.max_per_class,
        image_size_override=image_size,
        seed=args.seed,
        extra_meta={
            "train_split": args.train_split,
            "dedup_regex": args.dedup_regex,
        },
    )
    print(
        f"[saved] {output_path}: {metadata['num_samples']} samples from "
        f"{metadata['num_frames_used']} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
