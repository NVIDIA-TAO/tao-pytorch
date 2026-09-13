# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO-native Sparse4D lazy-index builder and runtime contract."""

import pickle
from pathlib import Path

import pytest

from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import (
    Omniverse3DDetTrackDataset,
)
from nvidia_tao_pytorch.cv.sparse4d.tools.build_lazy_index import (
    build_lazy_index,
    get_lazy_index_cache_path,
)


pytestmark = pytest.mark.cv_unit


def _write_annotation(path: Path, scene: str, camera_count: int) -> None:
    infos = []
    for frame_idx in range(2):
        infos.append(
            {
                "scene_name": scene,
                "timestamp": float(frame_idx),
                "frame_idx": frame_idx + 10,
                "cams": {f"cam{index}": {} for index in range(camera_count)},
            }
        )
    with open(path, "wb") as stream:
        pickle.dump(
            {"infos": infos, "metadata": {"version": "test-v1"}},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@pytest.mark.parametrize("input_kind", ["split", "directory"])
def test_builder_writes_runtime_schema_and_camera_counts(tmp_path, input_kind):
    """Both supported inputs produce exact frame records and both count caches."""
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    paths = [annotation_dir / "a.pkl", annotation_dir / "b.pkl"]
    _write_annotation(paths[0], "scene-a", 1)
    _write_annotation(paths[1], "scene-b", 3)

    if input_kind == "split":
        ann_file = tmp_path / "train.txt"
        ann_file.write_text("\n".join(str(path) for path in paths), encoding="utf-8")
    else:
        ann_file = annotation_dir

    result = build_lazy_index(ann_file, num_workers=1)
    with open(result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    with open(result["camera_counts_path"], "rb") as stream:
        sidecar = pickle.load(stream)

    expected_counts = {str(paths[0].resolve()): 1, str(paths[1].resolve()): 3}
    assert result["num_frames"] == 4
    assert result["num_pkls"] == 2
    assert index["metadata"] == {"version": "test-v1"}
    assert index["pkl_cam_counts"] == expected_counts
    assert sidecar == expected_counts
    assert all(
        set(entry) == {"pkl_path", "local_idx", "scene_name", "timestamp", "frame_idx"}
        for entry in index["frame_index"]
    )

    # A directory rerun must not index either generated cache as annotation data.
    rerun = build_lazy_index(ann_file, num_workers=1)
    assert rerun["num_pkls"] == 2
    assert rerun["num_reused_pkls"] == 2


def test_non_lazy_directory_discovery_ignores_generated_caches(tmp_path):
    """Generated index/count PKLs must not be parsed as frame annotations."""
    _write_annotation(tmp_path / "a.pkl", "scene-a", 1)
    _write_annotation(tmp_path / "b.pkl", "scene-b", 2)
    build_lazy_index(tmp_path, num_workers=1)
    dataset = object.__new__(Omniverse3DDetTrackDataset)

    paths = dataset._get_ann_paths(str(tmp_path))

    assert paths == [str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")]


def test_lazy_index_replaces_only_the_split_suffix(tmp_path):
    """A parent directory containing '.txt' is preserved in both code paths."""
    annotation_dir = tmp_path / "annotations.txt"
    annotation_dir.mkdir()
    annotation = annotation_dir / "scene.pkl"
    _write_annotation(annotation, "scene", 1)
    split = annotation_dir / "train.txt"
    split.write_text(str(annotation), encoding="utf-8")

    result = build_lazy_index(split, num_workers=1)
    expected = annotation_dir / "train_lazy_index.pkl"
    dataset = object.__new__(Omniverse3DDetTrackDataset)

    assert Path(result["cache_path"]) == expected
    assert get_lazy_index_cache_path(split) == str(expected)
    assert dataset._get_lazy_index_cache_path(str(split)) == str(expected)
    directory_cache = annotation_dir / "_lazy_index.pkl"
    assert get_lazy_index_cache_path(annotation_dir) == str(directory_cache)
    assert dataset._get_lazy_index_cache_path(str(annotation_dir)) == str(
        directory_cache
    )


def _lazy_loader(tmp_path, *, sidecar_path=None):
    annotation = tmp_path / "scene.pkl"
    _write_annotation(annotation, "scene", 2)
    split = tmp_path / "train.txt"
    split.write_text(str(annotation), encoding="utf-8")
    build_lazy_index(split, num_workers=1, write_camera_counts=False)

    dataset = object.__new__(Omniverse3DDetTrackDataset)
    dataset.max_frames = -1
    dataset.load_interval = 1
    dataset.pkl_sample_size = 1
    dataset.pkl_cam_counts_path = sidecar_path
    return dataset, split


def test_dataset_uses_camera_counts_embedded_in_lazy_index(tmp_path):
    """pkl_sample_size needs no separate sidecar for a newly built index."""
    dataset, split = _lazy_loader(tmp_path)

    data_infos = dataset.load_annotations_lazy(str(split))

    assert len(data_infos) == 2
    assert dataset._pkl_cam_counts == {
        str((tmp_path / "scene.pkl").resolve()): 2,
    }


def test_dataset_rejects_missing_explicit_camera_count_sidecar(tmp_path):
    """An explicitly configured missing sidecar fails before epoch sampling."""
    missing = tmp_path / "missing_counts.pkl"
    dataset, split = _lazy_loader(tmp_path, sidecar_path=str(missing))

    with pytest.raises(FileNotFoundError, match="pkl_cam_counts_path"):
        dataset.load_annotations_lazy(str(split))


def test_dataset_rejects_lazy_index_without_camera_counts(tmp_path):
    """A legacy index without counts explains how to rebuild the artifact."""
    dataset, split = _lazy_loader(tmp_path)
    cache_path = Path(get_lazy_index_cache_path(split))
    with open(cache_path, "rb") as stream:
        index = pickle.load(stream)
    index.pop("pkl_cam_counts")
    with open(cache_path, "wb") as stream:
        pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="tools.build_lazy_index"):
        dataset.load_annotations_lazy(str(split))


def test_pkl_sampling_rejects_non_lazy_dataset(tmp_path):
    """Sampling cannot silently become a no-op when lazy loading is disabled."""
    with pytest.raises(ValueError, match="requires lazy_load=True"):
        Omniverse3DDetTrackDataset(
            data_root="",
            anno_file=str(tmp_path / "unused.pkl"),
            classes=["person"],
            lazy_load=False,
            pkl_sample_size=1,
        )
