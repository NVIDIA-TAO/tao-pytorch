# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the on-disk index consumed by Sparse4D lazy annotation loading.

Annotation PKLs are Python pickle files and must come from trusted TAO dataset
generation workflows.  The generated camera-count mapping is embedded in the
index and, by default, also written as the legacy ``_pkl_cam_counts.pkl``
sidecar used by ``pkl_sample_size``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from os import path as osp
import pickle
import tempfile
from typing import Optional, Sequence


_GENERATED_FILENAMES = {"_lazy_index.pkl", "_pkl_cam_counts.pkl"}


def get_lazy_index_cache_path(ann_file: os.PathLike | str) -> str:
    """Return the cache path expected by ``Omniverse3DDetTrackDataset``."""
    ann_file = osp.expanduser(os.fspath(ann_file))
    if osp.isdir(ann_file):
        return osp.join(ann_file, "_lazy_index.pkl")
    if ann_file.endswith(".txt"):
        return f"{ann_file[:-len('.txt')]}_lazy_index.pkl"
    raise ValueError(f"Expected a split .txt file or annotation directory: {ann_file}")


def get_camera_counts_path(ann_file: os.PathLike | str) -> str:
    """Return the default camera-count sidecar path used by source tooling."""
    ann_file = osp.expanduser(os.fspath(ann_file))
    base_dir = ann_file if osp.isdir(ann_file) else osp.dirname(ann_file)
    return osp.join(base_dir, "_pkl_cam_counts.pkl")


def resolve_annotation_paths(ann_file: os.PathLike | str) -> list[str]:
    """Resolve the PKLs listed by the same directory/.txt inputs as the dataset."""
    ann_file = osp.expanduser(os.fspath(ann_file))
    if osp.isdir(ann_file):
        paths = [
            osp.join(ann_file, name)
            for name in sorted(os.listdir(ann_file))
            if name.endswith(".pkl") and name not in _GENERATED_FILENAMES
        ]
    elif ann_file.endswith(".txt"):
        if not osp.isfile(ann_file):
            raise FileNotFoundError(f"Annotation split not found: {ann_file}")
        with open(ann_file, "r", encoding="utf-8") as stream:
            paths = [line.split()[0] for line in stream if line.strip()]
    else:
        raise ValueError(
            f"Expected a split .txt file or annotation directory: {ann_file}"
        )

    paths = [osp.abspath(osp.expanduser(path)) for path in paths]
    if not paths:
        raise ValueError(f"No annotation PKLs found in: {ann_file}")
    missing = [path for path in paths if not osp.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} annotation PKL(s) do not exist; first: {missing[0]}"
        )
    return paths


def _index_one_pkl(pkl_path: str) -> tuple:
    """Return frame entries, metadata, mtime, and camera count for one PKL."""
    try:
        with open(pkl_path, "rb") as stream:
            data = pickle.load(stream)
        if not isinstance(data, dict) or not isinstance(data.get("infos"), list):
            raise ValueError("expected a dictionary containing an 'infos' list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("expected 'metadata' to be a dictionary")

        infos = data["infos"]
        entries = [
            {
                "pkl_path": pkl_path,
                "local_idx": local_idx,
                "scene_name": info["scene_name"],
                "timestamp": info["timestamp"],
                "frame_idx": info.get("frame_idx", local_idx),
            }
            for local_idx, info in enumerate(infos)
        ]
        camera_count = len(infos[0].get("cams", {})) if infos else 0
        return pkl_path, entries, metadata, osp.getmtime(pkl_path), camera_count
    except Exception as error:
        raise RuntimeError(f"Failed to index annotation PKL: {pkl_path}") from error


def _load_pickle_mapping(path: str) -> dict:
    """Load and validate a pickle mapping produced by this toolchain."""
    with open(path, "rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in cache: {path}")
    return value


def _atomic_pickle_dump(value, output_path: str) -> None:
    """Atomically serialize a pickle next to its final destination."""
    output_path = osp.abspath(osp.expanduser(output_path))
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{osp.basename(output_path)}.",
        suffix=".tmp",
        dir=osp.dirname(output_path),
    )
    os.close(descriptor)
    try:
        with open(temporary_path, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
    finally:
        if osp.exists(temporary_path):
            os.unlink(temporary_path)


def build_lazy_index(
    ann_file: os.PathLike | str,
    *,
    force: bool = False,
    num_workers: Optional[int] = None,
    write_camera_counts: bool = True,
    camera_counts_path: Optional[os.PathLike | str] = None,
) -> dict:
    """Build or incrementally update a Sparse4D lazy annotation index.

    The ``frame_index`` records have exactly the keys consumed by the runtime:
    ``pkl_path``, ``local_idx``, ``scene_name``, ``timestamp``, and
    ``frame_idx``. Camera counts are always embedded in the lazy index so
    balanced PKL sampling has no second required preprocessing step.
    """
    ann_file = osp.expanduser(os.fspath(ann_file))
    ann_paths = resolve_annotation_paths(ann_file)
    unique_paths = list(dict.fromkeys(ann_paths))
    cache_path = get_lazy_index_cache_path(ann_file)
    sidecar_path = (
        osp.expanduser(os.fspath(camera_counts_path))
        if camera_counts_path is not None
        else get_camera_counts_path(ann_file)
    )

    cached = {}
    existing_entries = {}
    previous_mtimes = {}
    previous_counts = {}
    if not force and osp.isfile(cache_path):
        cached = _load_pickle_mapping(cache_path)
        previous_mtimes = cached.get("mtimes", {})
        previous_counts = cached.get("pkl_cam_counts", {})
        for entry in cached.get("frame_index", []):
            pkl_path = entry.get("pkl_path")
            if pkl_path:
                existing_entries.setdefault(pkl_path, []).append(entry)

        # Older lazy indices did not embed counts. Reuse their matching legacy
        # sidecar when available; otherwise those PKLs are re-indexed below.
        if write_camera_counts and osp.isfile(sidecar_path):
            sidecar_counts = _load_pickle_mapping(sidecar_path)
            previous_counts = {**sidecar_counts, **previous_counts}

    reusable = set()
    to_index = []
    for pkl_path in unique_paths:
        if (
            pkl_path in existing_entries and
            previous_mtimes.get(pkl_path) == osp.getmtime(pkl_path) and
            pkl_path in previous_counts
        ):
            reusable.add(pkl_path)
        else:
            to_index.append(pkl_path)

    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 1)
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    if num_workers == 1:
        indexed_values = [_index_one_pkl(path) for path in to_index]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            indexed_values = list(executor.map(_index_one_pkl, to_index))
    indexed = {value[0]: value for value in indexed_values}

    frame_index = []
    mtimes = {}
    camera_counts = {}
    for pkl_path in ann_paths:
        if pkl_path in indexed:
            _, entries, _, mtime, camera_count = indexed[pkl_path]
        else:
            entries = existing_entries[pkl_path]
            mtime = previous_mtimes[pkl_path]
            camera_count = previous_counts[pkl_path]
        frame_index.extend(entries)
        mtimes[pkl_path] = mtime
        camera_counts[pkl_path] = int(camera_count)

    metadata = cached.get("metadata", {})
    if force or not metadata:
        metadata = {}
        for pkl_path in unique_paths:
            if pkl_path in indexed and indexed[pkl_path][2]:
                metadata = indexed[pkl_path][2]
                break

    index_data = {
        "frame_index": frame_index,
        "metadata": metadata,
        "mtimes": mtimes,
        "pkl_cam_counts": camera_counts,
    }
    _atomic_pickle_dump(index_data, cache_path)
    if write_camera_counts:
        _atomic_pickle_dump(camera_counts, sidecar_path)

    return {
        "cache_path": osp.abspath(cache_path),
        "camera_counts_path": (
            osp.abspath(sidecar_path) if write_camera_counts else None
        ),
        "num_frames": len(frame_index),
        "num_pkls": len(unique_paths),
        "num_reused_pkls": len(reusable),
        "num_indexed_pkls": len(to_index),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the lazy-index builder CLI."""
    parser = argparse.ArgumentParser(
        description="Build the index consumed by Sparse4D lazy annotation loading"
    )
    parser.add_argument("ann_file", help="Annotation split .txt file or directory")
    parser.add_argument(
        "--force", "-f", action="store_true", help="Ignore and rebuild existing data"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Worker processes (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--camera-counts-out",
        default=None,
        help="Override the default sibling _pkl_cam_counts.pkl path",
    )
    parser.add_argument(
        "--no-camera-counts-sidecar",
        action="store_true",
        help="Embed camera counts in the index without writing a legacy sidecar",
    )
    args = parser.parse_args(argv)
    result = build_lazy_index(
        args.ann_file,
        force=args.force,
        num_workers=args.workers,
        write_camera_counts=not args.no_camera_counts_sidecar,
        camera_counts_path=args.camera_counts_out,
    )
    print(
        f"Indexed {result['num_frames']} frames from {result['num_pkls']} PKLs "
        f"({result['num_reused_pkls']} reused): {result['cache_path']}"
    )
    if result["camera_counts_path"]:
        print(f"Camera-count sidecar: {result['camera_counts_path']}")


if __name__ == "__main__":
    main()
