# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests for Sparse4D's portable artifact producers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import pytest

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    LooseToTightMLP,
    WAREHOUSE_V4_CLASSES,
)
from nvidia_tao_pytorch.cv.sparse4d.tools import (
    build_sv2d_dataset,
    ltt_build_2dgt,
    ltt_data,
    ltt_extract,
    ltt_train,
    rtdetr_pseudo_labels,
    sv2d_common,
)


pytestmark = pytest.mark.cv_unit


def _training_rows(count=16):
    extent = np.tile([[2.0, 3.0, 1.5]], (count, 1))
    loose = np.tile([[0.0, 0.0, 100.0, 100.0]], (count, 1))
    tight = np.tile([[10.0, 15.0, 85.0, 90.0]], (count, 1))
    return ltt_data.pack(
        extent,
        np.linspace(-0.5, 0.5, count),
        np.zeros(count),
        np.zeros(count),
        np.full(count, 10.0),
        loose,
        tight,
        np.tile([[200.0, 100.0]], (count, 1)),
        np.ones(count),
    )


def test_ltt_cache_trains_and_loads_runtime_checkpoint(tmp_path):
    """The numeric extraction cache can produce a runtime-loadable MLP."""
    cache_path = tmp_path / "ltt.npz"
    packed = _training_rows()
    class_ids = np.arange(len(packed), dtype=np.int16) % 2
    ltt_data.save_cache(
        cache_path,
        packed,
        class_ids,
        np.arange(len(packed), dtype=np.int64) // 2,
        {"class_names": ["person", "forklift"]},
    )

    data, class_names, _ = ltt_train.load_dataset([str(cache_path)])
    model, result = ltt_train.train_mlp(
        data,
        class_names,
        hidden_dim=8,
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        val_fraction=0.25,
        seed=3,
    )
    checkpoint_path = tmp_path / "ltt.pth"
    model.save(checkpoint_path, extra_meta=result)
    loaded = LooseToTightMLP.load(checkpoint_path)

    assert loaded.num_classes == 2
    assert loaded.hidden_dim == 8
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert np.isfinite(result["best_val_giou"])
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache["packed"].dtype == np.float32
        assert cache["_meta"].dtype == np.uint8
        assert cache["group_id"].dtype == np.int64


def test_ltt_validation_split_keeps_source_frames_disjoint():
    """Every camera/object row from one frame stays in one data partition."""
    group_ids = np.repeat(np.arange(6, dtype=np.int64), 3)
    train_indices, validation_indices = ltt_train.split_group_indices(
        group_ids, val_fraction=0.34, seed=7
    )

    train_groups = set(group_ids[train_indices.numpy()].tolist())
    validation_groups = set(group_ids[validation_indices.numpy()].tolist())
    assert train_groups
    assert validation_groups
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == set(range(6))


def test_ltt_class_config_must_match_cache_taxonomy_order(tmp_path):
    """A class override cannot silently reinterpret cached integer labels."""
    class_config = tmp_path / "classes.py"
    class_config.write_text(
        'CLASS_LIST = ["forklift", "person"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly match cache class_names"):
        ltt_train.resolve_class_names(["person", "forklift"], str(class_config))


def test_ltt_class_config_supplies_only_missing_cache_taxonomy(tmp_path):
    """An explicit taxonomy remains supported when every cache omits metadata."""
    class_config = tmp_path / "classes.py"
    class_config.write_text(
        'CLASS_LIST = ["forklift", "person"]\n',
        encoding="utf-8",
    )

    assert ltt_train.resolve_class_names(None, str(class_config)) == [
        "forklift",
        "person",
    ]
    assert ltt_train.resolve_class_names(["forklift", "person"], str(class_config)) == [
        "forklift",
        "person",
    ]


def test_npz_cli_outputs_are_normalized_and_reported(tmp_path, monkeypatch, capsys):
    """NPZ producers pass and report the actual suffixed output path."""
    seen = {}

    monkeypatch.setattr(ltt_extract, "resolve_scene_paths", lambda *args: ["scene"])

    def fake_extract(scene_paths, output_path, *args, **kwargs):
        seen["ltt"] = output_path
        return {"num_samples": 1, "num_frames_used": 1}

    monkeypatch.setattr(ltt_extract, "extract_cache", fake_extract)
    ltt_output = tmp_path / "ltt_cache"
    assert ltt_extract.main(["--scene-dir", "scene", "--out", str(ltt_output)]) == 0
    assert seen["ltt"] == ltt_output.with_suffix(".npz")
    assert str(ltt_output.with_suffix(".npz")) in capsys.readouterr().out

    def fake_rtdetr(rtdetr_dir, output_path, *args, **kwargs):
        seen["rtdetr"] = output_path
        return {"num_rows": 1, "cam_names": ["cam0"]}

    monkeypatch.setattr(rtdetr_pseudo_labels, "build_cache", fake_rtdetr)
    rtdetr_output = tmp_path / "rtdetr_cache"
    assert (
        rtdetr_pseudo_labels.main(
            ["--rtdetr-dir", "labels", "--out", str(rtdetr_output)]
        )
        == 0
    )
    assert seen["rtdetr"] == rtdetr_output.with_suffix(".npz")
    assert str(rtdetr_output.with_suffix(".npz")) in capsys.readouterr().out


def _write_scene(scene_path):
    scene_path.mkdir()
    calibration = {
        "sensors": [
            {
                "type": "camera",
                "id": "cam0",
                "intrinsicMatrix": [[100, 0, 100], [0, 100, 50], [0, 0, 1]],
                "extrinsicMatrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ],
                "attributes": [
                    {"name": "frameWidth", "value": "200"},
                    {"name": "frameHeight", "value": "100"},
                ],
            }
        ]
    }
    (scene_path / "calibration.json").write_text(json.dumps(calibration))
    gt_dir = scene_path / "ground_truth_final"
    gt_dir.mkdir()
    annotations = [
        {
            "object type": "person",
            "object id": 7,
            "3d location": [0, 0, 10],
            "3d bounding box scale": [2, 2, 2],
            "3d bounding box rotation": [0, 0, 0],
            "2d bounding box": {"cam0": [85, 35, 115, 65]},
            "2d bounding box visible": {"cam0": [90, 40, 110, 60]},
        }
    ]
    (gt_dir / "ground_truth_000000.json").write_text(json.dumps(annotations))


def test_ltt_extractor_and_2dgt_sidecar_share_scene_contract(tmp_path):
    """Raw GT produces both the MLP cache and runtime sidecar without legacy libs."""
    scene_path = tmp_path / "SceneA"
    _write_scene(scene_path)
    class_names = list(WAREHOUSE_V4_CLASSES)
    name_to_id = {name: index for index, name in enumerate(class_names)}
    cache_path = tmp_path / "training.npz"

    extraction_meta = ltt_extract.extract_cache(
        [scene_path],
        cache_path,
        class_names,
        name_to_id,
        frame_stride=1,
    )
    outputs = ltt_build_2dgt.build_sidecars(
        [scene_path], tmp_path / "sidecars", class_names, name_to_id
    )

    packed, class_id, group_id, _ = ltt_data.load_cache(cache_path)
    assert extraction_meta["num_samples"] == 1
    assert class_id.tolist() == [0]
    assert group_id.tolist() == [0]
    np.testing.assert_allclose(packed[:, ltt_data.I_VIS], [4 / 9])
    with np.load(outputs[0], allow_pickle=False) as sidecar:
        assert sidecar["instance_id"].tolist() == [7]
        assert sidecar["cam"].tolist() == [0]
        np.testing.assert_allclose(sidecar["box3"], [[90, 40, 110, 60]])
        assert ltt_data.decode_meta(sidecar["_meta"])["scene"] == "SceneA"


def _add_tar_member(archive, name, text):
    payload = text.encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_rtdetr_builder_drops_pallet_and_writes_safe_npz(tmp_path):
    """KITTI archives map classes, filter scores, and never use object arrays."""
    camera_dir = tmp_path / "SceneA" / "rt-detr" / "cam_a"
    camera_dir.mkdir(parents=True)
    with tarfile.open(camera_dir / "labels.tar.gz", "w:gz") as archive:
        _add_tar_member(
            archive,
            "labels/rgb_000042.txt",
            "person 0 0 0 1 2 20 30 0 0 0 0 0 0 0 0.9\n"
            "pallet 0 0 0 3 4 40 50 0 0 0 0 0 0 0 0.99\n"
            "forklift 0 0 0 5 6 50 60 0 0 0 0 0 0 0 0.2\n",
        )
        _add_tar_member(archive, "labels/rgb_000043.txt", "")
    class_names = list(WAREHOUSE_V4_CLASSES)
    output_path = tmp_path / "SceneA__rtdetr2d.npz"
    metadata = rtdetr_pseudo_labels.build_cache(
        camera_dir.parent,
        output_path,
        class_names,
        {name: index for index, name in enumerate(class_names)},
        camera_map={"cam_a": "CameraA"},
        confidence_threshold=0.4,
    )

    assert metadata["cam_names"] == ["CameraA"]
    assert metadata["raw_class_counts"] == {
        "person": 1,
        "pallet": 1,
        "forklift": 1,
    }
    with np.load(output_path, allow_pickle=False) as cache:
        assert cache["frame_id"].tolist() == [42]
        assert cache["class_id"].tolist() == [0]
        assert cache["box"].shape == (1, 4)
        assert cache["valid_frame_id"].tolist() == [42, 43]
        assert cache["valid_cam"].tolist() == [0, 0]
        assert all(cache[key].dtype.kind != "O" for key in cache.files)


def test_sv2d_builder_applies_suffix_to_both_artifacts_and_fixes_h5_uri(tmp_path):
    """Smoke variants stay isolated and HDF5 keys omit scheme/tag prefixes."""
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "h5://fixture:image_1.jpg",
                        "width": 100,
                        "height": 50,
                    },
                    {
                        "id": 2,
                        "file_name": "h5://fixture:image_2.jpg",
                        "width": 100,
                        "height": 50,
                    },
                ],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "pallet"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
                    {"image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5]},
                ],
            }
        )
    )
    dataset = {
        "name": "fixture",
        "scene_name": "SV2D__fixture",
        "coco": str(coco_path),
        "kind": "h5",
        "h5_path": str(tmp_path / "images.h5"),
        "weight": 2,
    }
    result = build_sv2d_dataset.build_one(
        dataset,
        tmp_path / "cache",
        tmp_path / "pkls",
        keep_empty=True,
        suffix="_smoke",
    )

    assert result["scene"] == "SV2D__fixture_smoke"
    assert result["num_frames"] == 2
    assert result["num_detections"] == 1
    assert result["npz_path"].endswith("SV2D__fixture_smoke__rtdetr2d.npz")
    assert result["pkl_path"].endswith("SV2D__fixture_smoke_infos_train.pkl")
    assert not (tmp_path / "cache" / "SV2D__fixture__rtdetr2d.npz").exists()
    with np.load(result["npz_path"], allow_pickle=False) as cache:
        assert cache["class_id"].tolist() == [0]
        assert cache["valid_frame_id"].tolist() == [0, 1]
        assert cache["valid_cam"].tolist() == [0, 0]
        assert ltt_data.decode_meta(cache["_meta"])["scene"].endswith("_smoke")
    with open(result["pkl_path"], "rb") as stream:
        info = pickle.load(stream)["infos"][0]
    assert info["cams"]["cam0"]["data_path"] == (
        str(tmp_path / "images.h5"),
        "rgb/image_1.jpg",
    )
    assert "depth_map_path" not in info["cams"]["cam0"]
    assert sv2d_common.strip_h5_uri("h5://tag:folder/image.jpg") == ("folder/image.jpg")

    split_artifacts = build_sv2d_dataset.write_split_artifacts(
        [result], [dataset], tmp_path / "sv2d_train_split.txt"
    )
    assert (tmp_path / "sv2d_train_split.txt").read_text().splitlines() == [
        str(Path(result["pkl_path"]).resolve()),
    ]
    weights_path = tmp_path / "sv2d_train_split.sv2d_weights.json"
    assert json.loads(weights_path.read_text()) == {"SV2D__fixture_smoke": 2.0}
    assert split_artifacts == {
        "split_path": str(tmp_path / "sv2d_train_split.txt"),
        "weights_path": str(weights_path),
    }


def test_sv2d_cli_records_custom_taxonomy_and_class_id_order(tmp_path):
    """SV2D artifacts can match a non-default dataset.classes order exactly."""
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 100, "height": 50}
                ],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "forklift"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]},
                    {"image_id": 1, "category_id": 2, "bbox": [20, 2, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "fixture",
                    "scene_name": "SV2D__fixture",
                    "coco": str(coco_path),
                    "kind": "file",
                    "image_root": str(tmp_path / "images"),
                }
            ]
        ),
        encoding="utf-8",
    )
    class_config = tmp_path / "classes.py"
    class_config.write_text(
        'CLASS_LIST = ["forklift", "person"]\n',
        encoding="utf-8",
    )

    assert (
        build_sv2d_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pkl-dir",
                str(tmp_path / "pkls"),
                "--class-config",
                str(class_config),
            ]
        )
        == 0
    )

    with np.load(
        tmp_path / "cache" / "SV2D__fixture__rtdetr2d.npz",
        allow_pickle=False,
    ) as cache:
        assert cache["class_id"].tolist() == [1, 0]
        metadata = ltt_data.decode_meta(cache["_meta"])
    assert metadata["class_names"] == ["forklift", "person"]
    assert metadata["class_map"] == {
        "forklift": "forklift",
        "person": "person",
    }


@pytest.mark.parametrize(
    "suffix,expected_name",
    [("", "sv2d_train_split.txt"), ("_smoke", "sv2d_train_split_smoke.txt")],
)
def test_sv2d_cli_suffix_isolates_default_split_artifacts(
    tmp_path, monkeypatch, suffix, expected_name
):
    """The default name stays compatible and inherits a non-empty suffix."""
    dataset = {"name": "fixture"}
    result = {
        "dataset": "fixture",
        "scene": "SV2D__fixture_smoke",
        "num_frames": 1,
        "num_detections": 1,
        "pkl_path": str(tmp_path / "fixture.pkl"),
        "npz_path": str(tmp_path / "fixture.npz"),
    }
    seen = {}
    monkeypatch.setattr(
        build_sv2d_dataset.sv2d_common,
        "load_dataset_manifest",
        lambda manifest: [dataset],
    )
    monkeypatch.setattr(build_sv2d_dataset, "build_one", lambda *args, **kwargs: result)

    def fake_write_split(results, datasets, output_path):
        seen["split"] = output_path
        return {
            "split_path": output_path,
            "weights_path": str(Path(output_path).with_suffix(".sv2d_weights.json")),
        }

    monkeypatch.setattr(build_sv2d_dataset, "write_split_artifacts", fake_write_split)
    assert (
        build_sv2d_dataset.main(
            [
                "--manifest",
                str(tmp_path / "datasets.json"),
                "--dataset",
                "all",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pkl-dir",
                str(tmp_path),
                "--suffix",
                suffix,
            ]
        )
        == 0
    )
    assert seen["split"] == str(tmp_path / expected_name)


@pytest.mark.parametrize("suffix", ["nested/_smoke", r"nested\_smoke"])
def test_sv2d_cli_rejects_path_suffix_with_empty_manifest(tmp_path, suffix):
    """Suffix validation runs even when an all-dataset manifest has no entries."""
    manifest = tmp_path / "empty.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="path separator"):
        build_sv2d_dataset.main(
            [
                "--dataset",
                "all",
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pkl-dir",
                str(tmp_path),
                "--suffix",
                suffix,
            ]
        )

    assert not (tmp_path / "nested").exists()


def _sv2d_manifest_entry(name, scene_name):
    """Return a minimal file-backed SV2D manifest entry."""
    return {
        "name": name,
        "scene_name": scene_name,
        "coco": "annotations.json",
        "kind": "file",
        "image_root": "images",
    }


@pytest.mark.parametrize(
    "scene_name",
    ["", " ", ".", "..", "/", "\\", "../escape", "nested/escape", r"nested\escape"],
)
def test_sv2d_manifest_rejects_unsafe_scene_name_before_writes(
    tmp_path,
    scene_name,
):
    """Scene names cannot redirect artifacts outside explicit output roots."""
    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps([_sv2d_manifest_entry("fixture", scene_name)]),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    pkl_dir = tmp_path / "pkls"

    with pytest.raises(ValueError, match="scene_name"):
        build_sv2d_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(cache_dir),
                "--pkl-dir",
                str(pkl_dir),
            ]
        )

    assert not cache_dir.exists()
    assert not pkl_dir.exists()


def test_sv2d_build_one_revalidates_scene_name_before_io(tmp_path):
    """Direct callers receive the same artifact-name safety boundary."""
    dataset = _sv2d_manifest_entry("fixture", "../escape")

    with pytest.raises(ValueError, match="scene_name"):
        build_sv2d_dataset.build_one(
            dataset,
            tmp_path / "cache",
            tmp_path / "pkls",
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "pkls").exists()


def test_sv2d_manifest_rejects_duplicate_scene_names_before_writes(tmp_path):
    """Two dataset aliases cannot overwrite the same artifact filenames."""
    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps(
            [
                _sv2d_manifest_entry("first", "SV2D__shared"),
                _sv2d_manifest_entry("second", "SV2D__shared"),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scene_name values must be unique"):
        build_sv2d_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pkl-dir",
                str(tmp_path / "pkls"),
            ]
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "pkls").exists()


def test_sv2d_manifest_rejects_empty_dataset_list(tmp_path):
    """An empty manifest cannot report a successful no-op artifact build."""
    manifest = tmp_path / "empty.json"
    manifest.write_text('{"datasets": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="at least one dataset"):
        build_sv2d_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pkl-dir",
                str(tmp_path / "pkls"),
            ]
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "pkls").exists()


def test_sv2d_cli_requires_manifest_and_explicit_output_directories():
    """The portable CLI must not fall back to private input or output paths."""
    parser = build_sv2d_dataset._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "datasets.json"])


def test_sv2d_manifest_rejects_missing_path_and_unknown_kind(tmp_path):
    """Manifest validation fails early for missing or unsupported sources."""
    with pytest.raises(ValueError, match="manifest path is required"):
        sv2d_common.load_dataset_manifest(None)

    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "fixture",
                    "scene_name": "SV2D__fixture",
                    "coco": "annotations.json",
                    "kind": "remote",
                    "image_root": "images",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported kind"):
        sv2d_common.load_dataset_manifest(manifest)
