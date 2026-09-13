# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TAO Sparse4D PKL + safe pseudo-label NPZ artifacts from SV2D COCO."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
from typing import Optional, Sequence

import numpy as np

from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_data, sv2d_common


def validate_suffix(suffix: str) -> None:
    """Reject suffixes that could redirect an artifact outside its output directory."""
    if not isinstance(suffix, str):
        raise ValueError("suffix must be a string")
    if any(separator in suffix for separator in ("/", "\\")):
        raise ValueError("suffix must not contain a path separator")
    if "\x00" in suffix:
        raise ValueError("suffix must not contain a NUL character")


def _scaled_box(bbox, width: float, height: float) -> Optional[list]:
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if (
        width <= 0 or
        height <= 0 or
        box_width <= 0 or
        box_height <= 0 or
        not np.isfinite([x, y, box_width, box_height]).all()
    ):
        return None
    scale_x = sv2d_common.CANON_W / width
    scale_y = sv2d_common.CANON_H / height
    x_min = float(np.clip(x * scale_x, 0.0, sv2d_common.CANON_W))
    y_min = float(np.clip(y * scale_y, 0.0, sv2d_common.CANON_H))
    x_max = float(np.clip((x + box_width) * scale_x, 0.0, sv2d_common.CANON_W))
    y_max = float(np.clip((y + box_height) * scale_y, 0.0, sv2d_common.CANON_H))
    if x_max - x_min < 1.0 or y_max - y_min < 1.0:
        return None
    return [x_min, y_min, x_max, y_max]


def build_one(
    dataset: dict,
    cache_dir: os.PathLike | str,
    pkl_dir: os.PathLike | str,
    *,
    class_names: Optional[Sequence[str]] = None,
    name_to_id: Optional[dict] = None,
    max_images: int = 0,
    keep_empty: bool = False,
    suffix: str = "",
) -> dict:
    """Build one calibration-free COCO dataset for the Sparse4D 2D route.

    The annotation PKL is a trusted local runtime artifact.  The separately
    consumed pseudo-label NPZ contains numeric arrays and JSON bytes only.
    """
    validate_suffix(suffix)
    scene_name = sv2d_common.validate_scene_name(dataset.get("scene_name"))
    class_names = list(sv2d_common.CLASS_NAMES if class_names is None else class_names)
    if (
        not class_names or
        any(not isinstance(name, str) or not name for name in class_names) or
        len(class_names) != len(set(class_names))
    ):
        raise ValueError("SV2D class_names must be non-empty, unique strings")
    resolved_name_to_id = (
        {name: index for index, name in enumerate(class_names)}
        if name_to_id is None
        else {str(name): int(class_id) for name, class_id in name_to_id.items()}
    )
    if any(
        class_id < 0 or class_id >= len(class_names)
        for class_id in resolved_name_to_id.values()
    ):
        raise ValueError("SV2D name_to_id contains an ID outside class_names")
    with open(dataset["coco"], "r", encoding="utf-8") as stream:
        coco = json.load(stream)
    images = coco.get("images", [])
    categories = {
        category["id"]: str(category["name"]) for category in coco.get("categories", [])
    }
    annotations_by_image = {}
    for annotation in coco.get("annotations", []):
        class_id = sv2d_common.map_coco_name(
            categories.get(annotation.get("category_id"), ""),
            resolved_name_to_id,
        )
        if class_id is not None:
            annotations_by_image.setdefault(annotation.get("image_id"), []).append(
                (class_id, annotation.get("bbox"))
            )

    intrinsic, world2cam = sv2d_common.build_virtual_camera()
    artifact_scene = f"{scene_name}{suffix}"
    infos = []
    frame_ids, cameras, class_ids, boxes, scores = [], [], [], [], []
    for image in sorted(images, key=lambda value: value["id"]):
        try:
            image_width = float(image["width"])
            image_height = float(image["height"])
        except (KeyError, TypeError, ValueError):
            continue
        mapped_boxes = []
        for class_id, bbox in annotations_by_image.get(image["id"], []):
            box = _scaled_box(bbox, image_width, image_height)
            if box is not None:
                mapped_boxes.append((class_id, box))
        if not mapped_boxes and not keep_empty:
            continue

        frame_id = len(infos)
        for class_id, box in mapped_boxes:
            frame_ids.append(frame_id)
            cameras.append(0)
            class_ids.append(class_id)
            boxes.append(box)
            scores.append(1.0)
        token = f"{artifact_scene}__{frame_id:09d}"
        infos.append(
            {
                "frame_idx": frame_id,
                "cams": {
                    "cam0": {
                        "data_path": sv2d_common.resolve_image_ref(
                            dataset, image["file_name"]
                        ),
                        "sample_data_token": f"{token}+cam0",
                        "cam_intrinsic": intrinsic.copy(),
                        # TAO retains the historical field name, but consumes it
                        # as world-to-camera in projection_mat construction.
                        "sensor2world_transform": world2cam.copy(),
                        "group_info_dict": {
                            "origin": np.asarray([0.0, 0.0], dtype=np.float32),
                            "dimensions": np.asarray(
                                [-50.0, -50.0, 50.0, 50.0], dtype=np.float32
                            ),
                        },
                    }
                },
                "scene_name": artifact_scene,
                "timestamp": float(frame_id) / 30.0,
                "token": token,
                "group_name": artifact_scene,
                "gt_boxes": None,
            }
        )
        if 0 < max_images <= len(infos):
            break

    columns = {
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "cam": np.asarray(cameras, dtype=np.int16),
        "class_id": np.asarray(class_ids, dtype=np.int16),
        "box": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "score": np.asarray(scores, dtype=np.float32),
        # Preserve frames retained by --keep-empty even though they have no
        # detection row. All SV2D samples use the single virtual camera.
        "valid_frame_id": np.arange(len(infos), dtype=np.int32),
        "valid_cam": np.zeros(len(infos), dtype=np.int16),
    }
    metadata = {
        "schema_version": "ltt_rtdetr2d/v1",
        "scene": artifact_scene,
        "cam_names": ["cam0"],
        "class_names": class_names,
        "conf_thr": 0.0,
        "frame_stride": 1,
        "num_rows": int(len(frame_ids)),
        "num_valid_frame_cameras": int(len(infos)),
        "source": "SV2D COCO GT (bare pallet dropped); canonical 1920x1080",
        "class_map": {
            source_name: class_names[class_id]
            for source_name, class_id in resolved_name_to_id.items()
            if source_name not in sv2d_common.DROP_NAMES
        },
        "virtual_camera": sv2d_common.VCAM,
    }
    cache_dir = Path(cache_dir).expanduser()
    pkl_dir = Path(pkl_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{artifact_scene}__rtdetr2d.npz"
    pkl_path = pkl_dir / f"{artifact_scene}_infos_train.pkl"
    np.savez_compressed(npz_path, _meta=ltt_data.encode_meta(metadata), **columns)
    with open(pkl_path, "wb") as stream:
        pickle.dump(
            {"infos": infos, "metadata": {"version": "sv2d_2d_only"}},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    class_counts = Counter(class_ids)
    return {
        "dataset": dataset["name"],
        "scene": artifact_scene,
        "num_frames": len(infos),
        "num_detections": len(frame_ids),
        "per_class": {
            class_names[class_id]: count
            for class_id, count in sorted(class_counts.items())
        },
        "npz_path": str(npz_path),
        "pkl_path": str(pkl_path),
    }


def write_split_artifacts(
    results: Sequence[dict],
    datasets: Sequence[dict],
    output_path: os.PathLike | str,
) -> dict:
    """Write a sequence-safe split and an advisory per-scene weight map.

    PKL paths must remain unique: repeating a sequence in a Sparse4D split
    creates misleading sequence flags after the dataset sorts its frames.  The
    source manifest's relative weights are therefore preserved separately for
    future/custom samplers, not applied by duplicating data.
    """
    by_name = {dataset["name"]: dataset for dataset in datasets}
    lines = []
    scene_weights = {}
    for result in results:
        weight_value = by_name[result["dataset"]].get("weight", 1)
        try:
            weight = float(weight_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"SV2D weight for {result['dataset']!r} must be finite and positive"
            ) from error
        if weight <= 0 or not np.isfinite(weight):
            raise ValueError(
                f"SV2D weight for {result['dataset']!r} must be finite and positive"
            )
        pkl_path = str(Path(result["pkl_path"]).resolve())
        lines.append(pkl_path)
        scene_weights[result["scene"]] = weight
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        stream.write("".join(f"{line}\n" for line in lines))
    weights_path = output.with_suffix(".sv2d_weights.json")
    with open(weights_path, "w", encoding="utf-8") as stream:
        json.dump(scene_weights, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"split_path": str(output), "weights_path": str(weights_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest describing the file/HDF5 COCO sources to convert.",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        required=True,
        help="Output directory for numeric pseudo-label NPZ artifacts.",
    )
    parser.add_argument(
        "--pkl-dir",
        "--pkl_dir",
        required=True,
        help="Output directory for TAO Sparse4D annotation PKLs.",
    )
    parser.add_argument("--max-images", "--max_images", type=int, default=0)
    parser.add_argument("--keep-empty", "--keep_empty", action="store_true")
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--class-config",
        help=(
            "Python class config defining CLASS_LIST and optional SUB_CLASS_DICT. "
            "Defaults to the warehouse-v4 taxonomy."
        ),
    )
    parser.add_argument(
        "--split-out",
        help=(
            "TAO PKL list. With --dataset all, defaults to "
            "<pkl-dir>/sv2d_train_split<suffix>.txt."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    validate_suffix(args.suffix)
    datasets = sv2d_common.load_dataset_manifest(args.manifest)
    class_names, name_to_id = ltt_data.load_class_taxonomy(args.class_config)
    by_name = {dataset["name"]: dataset for dataset in datasets}
    if args.dataset == "all":
        selected = datasets
    elif args.dataset in by_name:
        selected = [by_name[args.dataset]]
    else:
        raise ValueError(
            f"Unknown dataset {args.dataset!r}; choose one of {sorted(by_name)} or 'all'"
        )
    results = []
    for dataset in selected:
        result = build_one(
            dataset,
            args.cache_dir,
            args.pkl_dir,
            class_names=class_names,
            name_to_id=name_to_id,
            max_images=args.max_images,
            keep_empty=args.keep_empty,
            suffix=args.suffix,
        )
        print(
            f"[{result['dataset']}] {result['num_frames']} frames, "
            f"{result['num_detections']} detections\n"
            f"  NPZ: {result['npz_path']}\n  PKL: {result['pkl_path']}"
        )
        results.append(result)
    split_output = args.split_out
    if split_output is None and args.dataset == "all":
        split_output = os.path.join(args.pkl_dir, f"sv2d_train_split{args.suffix}.txt")
    if split_output:
        split_artifacts = write_split_artifacts(results, selected, split_output)
        print(f"  split: {split_artifacts['split_path']}")
        print(f"  advisory weights: {split_artifacts['weights_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
