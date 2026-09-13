# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-scene visible-2D-GT sidecars for Loose-to-Tight supervision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_data, ltt_extract


SCHEMA_VERSION = "ltt_2dgt/v1"


def build_scene(
    scene_dir: os.PathLike | str,
    name_to_id: Dict[str, int],
    frame_stride: int = 1,
    max_frames: int = 0,
) -> Tuple[dict, list]:
    """Collect numeric columns for every valid frame/object/camera tuple."""
    camera_index: Dict[str, int] = {}
    frame_ids, instance_ids, class_ids, cameras = [], [], [], []
    amodal_boxes, visible_boxes, occlusion_weights = [], [], []
    for frame_id, frame_annotations in ltt_extract.iter_gt_frames(
        scene_dir, frame_stride, max_frames
    ):
        for annotation in frame_annotations:
            class_id = name_to_id.get(str(annotation.get("object type", "")))
            if class_id is None:
                continue
            try:
                instance_id = int(annotation.get("object id", -1))
            except (TypeError, ValueError):
                instance_id = -1
            amodal_by_camera = annotation.get("2d bounding box", {}) or {}
            visible_by_camera = annotation.get("2d bounding box visible", {}) or {}
            for camera_name, amodal_value in amodal_by_camera.items():
                try:
                    amodal = np.asarray(amodal_value, dtype=np.float64).reshape(4)
                except ValueError:
                    continue
                amodal_area = ltt_extract.bbox_area(amodal)
                if amodal_area <= 1.0 or not np.isfinite(amodal).all():
                    continue
                visible_value = visible_by_camera.get(camera_name)
                if visible_value is None:
                    visible = amodal
                    occlusion = 0.0
                else:
                    try:
                        visible = np.asarray(visible_value, dtype=np.float64).reshape(4)
                    except ValueError:
                        continue
                    if not np.isfinite(visible).all():
                        continue
                    occlusion = float(
                        np.clip(
                            ltt_extract.bbox_area(visible) / amodal_area,
                            0.0,
                            1.0,
                        )
                    )
                camera_id = camera_index.setdefault(str(camera_name), len(camera_index))
                frame_ids.append(frame_id)
                instance_ids.append(instance_id)
                class_ids.append(class_id)
                cameras.append(camera_id)
                amodal_boxes.append(amodal)
                visible_boxes.append(visible)
                occlusion_weights.append(occlusion)

    camera_names = [
        name for name, _ in sorted(camera_index.items(), key=lambda item: item[1])
    ]
    columns = {
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "instance_id": np.asarray(instance_ids, dtype=np.int64),
        "class_id": np.asarray(class_ids, dtype=np.int16),
        "cam": np.asarray(cameras, dtype=np.int16),
        "box2": np.asarray(amodal_boxes, dtype=np.float32).reshape(-1, 4),
        "box3": np.asarray(visible_boxes, dtype=np.float32).reshape(-1, 4),
        "occ": np.asarray(occlusion_weights, dtype=np.float32),
    }
    return columns, camera_names


def save_sidecar(
    output_path: os.PathLike | str,
    columns: dict,
    metadata: dict,
) -> None:
    """Write a numeric-only compressed sidecar consumed with allow_pickle=False."""
    count = len(columns["frame_id"])
    expected = {
        "instance_id": (count,),
        "class_id": (count,),
        "cam": (count,),
        "box2": (count, 4),
        "box3": (count, 4),
        "occ": (count,),
    }
    for key, shape in expected.items():
        if np.asarray(columns[key]).shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, _meta=ltt_data.encode_meta(metadata), **columns)


def build_sidecars(
    scene_paths: Sequence[os.PathLike | str],
    output_dir: os.PathLike | str,
    class_names: Sequence[str],
    name_to_id: Dict[str, int],
    *,
    anno_version: str = "v0.1",
    frame_stride: int = 1,
    max_frames_per_scene: int = 0,
) -> list:
    """Build one ``<scene>__ltt2dgt.npz`` file for every input scene."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for scene_path in scene_paths:
        scene = Path(scene_path).resolve().name
        columns, camera_names = build_scene(
            scene_path,
            name_to_id,
            frame_stride=frame_stride,
            max_frames=max_frames_per_scene,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "cam_names": camera_names,
            "class_names": list(class_names),
            "num_rows": int(len(columns["frame_id"])),
            "frame_stride": int(frame_stride),
            "anno_version": anno_version,
        }
        output_path = output_dir / f"{scene}__ltt2dgt.npz"
        save_sidecar(output_path, columns, metadata)
        outputs.append(str(output_path))
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--scenes-file")
    parser.add_argument("--scene-dir", nargs="*")
    parser.add_argument("--train-split")
    parser.add_argument("--dedup-regex", default=r"^CT[\w.]+?__")
    parser.add_argument("--class-config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--anno-version", choices=["v0.0", "v0.1"], default="v0.1")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    scene_paths = ltt_extract.resolve_scene_paths(
        args.data_root,
        args.scenes,
        args.scenes_file,
        args.scene_dir,
        args.train_split,
        args.dedup_regex,
    )
    class_names, name_to_id = ltt_data.load_class_taxonomy(args.class_config)
    outputs = build_sidecars(
        scene_paths,
        args.out_dir,
        class_names,
        name_to_id,
        anno_version=args.anno_version,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    print(f"[saved] {len(outputs)} LTT 2D-GT sidecar(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
