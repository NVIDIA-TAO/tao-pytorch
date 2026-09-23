# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for contracts shared with data-services and core."""

import io
import json
import tarfile

from omegaconf import OmegaConf
import pytest

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import Omniverse3DDetTrackDataset
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule
from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_extract, rtdetr_pseudo_labels, sv2d_common
from nvidia_tao_pytorch.cv.sparse4d.tools.build_lazy_index import resolve_annotation_paths


pytestmark = pytest.mark.cv_unit


@pytest.mark.parametrize("probability", [-2.0, -0.5, 1.1, float("nan"), float("inf")])
def test_invalid_route_probability_fails_in_dataset_and_datamodule(probability):
    """Exactly -1 is the auto sentinel; fractional negatives are configuration errors."""
    with pytest.raises(ValueError, match="real_block_prob"):
        Omniverse3DDetTrackDataset("", "unused.pkl", ["person"], real_block_prob=probability)
    config = OmegaConf.structured(ExperimentConfig())
    config.dataset.data_root = "."
    config.dataset.real_block_prob = probability
    with pytest.raises(ValueError, match="real_block_prob"):
        Sparse4DDataModule(config)


@pytest.mark.parametrize("probability,expected", [(-1.0, None), (0.0, 0.0), (1.0, 1.0)])
def test_supported_route_probability_reaches_datamodule(probability, expected):
    """The schema-visible sentinel and endpoints retain their runtime meaning."""
    config = OmegaConf.structured(ExperimentConfig())
    config.dataset.data_root = "."
    for section in ("train_dataset", "val_dataset", "test_dataset"):
        config.dataset[section].ann_file = "unused.pkl"
    config.dataset.real_block_prob = probability
    assert Sparse4DDataModule(config).real_block_prob == expected


def test_split_path_identity_matches_runtime_and_native_builder(tmp_path, monkeypatch):
    """CWD-relative split rows retain symlink mount names through both consumers."""
    monkeypatch.chdir(tmp_path)
    physical = tmp_path / "physical"
    physical.mkdir()
    (physical / "scene.pkl").touch()
    alias = tmp_path / "mounted"
    alias.symlink_to(physical, target_is_directory=True)
    splits = tmp_path / "splits"
    splits.mkdir()
    split = splits / "train.txt"
    split.write_text("# fixture\nmounted/scene.pkl weight=2\n", encoding="utf-8")
    dataset = object.__new__(Omniverse3DDetTrackDataset)
    expected = [str(alias / "scene.pkl")]
    assert dataset._get_ann_paths(str(split)) == expected
    assert resolve_annotation_paths(split) == expected
    split.write_text("mounted/scene.pkl\nmounted/scene.pkl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        dataset._get_ann_paths(str(split))
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_annotation_paths(split)


def test_detector_taxonomy_requires_alias_or_explicit_drop(tmp_path):
    """Unknown labels must not be silently interpreted as background."""
    archive_path = tmp_path / "labels.tar.gz"
    payload = b"Human 0 0 0 1 2 20 30 0 0 0 0 0 0 0 0.9\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("labels/frame_000001.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unmapped detector class"):
        rtdetr_pseudo_labels.parse_camera_archive(archive_path, 0, {"person": 0})
    kept, _ = rtdetr_pseudo_labels.parse_camera_archive(
        archive_path, 0, {"person": 0}, class_name_map={"Human": "person"},
    )
    dropped, _ = rtdetr_pseudo_labels.parse_camera_archive(
        archive_path, 0, {"person": 0}, class_name_map={"Human": None},
    )
    assert kept["class_id"].tolist() == [0]
    assert dropped["class_id"].size == 0
    assert dropped["valid_frame_id"].tolist() == [1]


def test_projection_only_calibration_is_rejected(tmp_path):
    """Projective transforms cannot supply camera-space metric LTT features."""
    calibration = {"sensors": [{
        "id": "cam0", "type": "camera",
        "cameraMatrix": [[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]],
    }]}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="separate intrinsics"):
        ltt_extract.load_scene_calibration(tmp_path)


@pytest.mark.parametrize("per_frame", [False, True])
def test_both_gt_layouts_reject_ambiguous_frame_ids(tmp_path, per_frame):
    """Sampling limits must not conceal frame-ID aliases."""
    if per_frame:
        frames = tmp_path / "ground_truth_final"
        frames.mkdir()
        for name in ["ground_truth_1.json", "ground_truth_0001.json"]:
            (frames / name).write_text("[]", encoding="utf-8")
    else:
        (tmp_path / "ground_truth.json").write_text(
            '{"metadata": {}, "1": [], "01": []}', encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Duplicate normalized frame ID"):
        list(ltt_extract.iter_gt_frames(tmp_path, max_frames=1))


def test_scene_names_cannot_contain_runtime_group_separator():
    """Producer cache identity must survive runtime BEV-group stripping."""
    with pytest.raises(ValueError, match="BEV"):
        sv2d_common.validate_scene_name("SV2D__fixture+variant")
