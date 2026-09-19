# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVPanoptix3Dv2 dataloaders."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic import (
    augmentations,
    pl_data_module as panoptic_data_module,
    scannetpp,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.augmentations import (
    GeometrySafePhotometricAugmentation,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
    NVPanoptix3Dv2PanopticDataModule,
    resolve_panoptic_paths,
    resolve_vocabulary,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.scannetpp import (
    ScanNetppCollator,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.reasoning import (
    pl_data_module as reasoning_data_module,
    reasoning_dataset,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.reasoning.pl_data_module import (
    NVPanoptix3Dv2ReasoningDataModule,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.reasoning.reasoning_dataset import (
    ReasoningSegDataset,
    resolve_manifest_path,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.utils import (
    center_crop_and_resize,
    get_config_value,
    normalize_resolution_arg,
    rgb2id,
)


class _TinyDataset(Dataset):
    """One-record dataset used to inspect DataLoader settings."""

    classes = ["chair"]
    categories = [{"id": 0, "name": "chair", "isthing": True}]

    def __len__(self):
        """Return one synthetic record."""
        return 1

    def __getitem__(self, index):
        """Return the requested index."""
        return index


def _make_panoptic_roots(tmp_path):
    """Create the minimum valid ScanNet++ preprocessed roots."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots = {}
    for split, directory in (
        ("train", "pre_scannetpp_v2"),
        ("val", "pre_scannetpp_v2_val"),
    ):
        root = tmp_path / directory
        root.mkdir()
        (root / "all_metadata.npz").touch()
        (root / "categories.json").write_text("[]", encoding="utf-8")
        roots[split] = root
    return roots


def _load_manifest(tmp_path, records):
    """Write and load a reasoning manifest without image decoding."""
    manifest = tmp_path / "reasoning.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    dataset = object.__new__(ReasoningSegDataset)
    dataset.require_seg_token = True
    dataset.depth_scale = 1000.0
    return dataset.load_manifest(str(manifest))


@pytest.mark.cv_unit
def test_variants_reuse_shared_helpers():
    """Panoptic and reasoning must not duplicate dataloader utilities."""
    for consumer in (scannetpp, reasoning_dataset):
        assert consumer.center_crop_and_resize is center_crop_and_resize
        assert consumer.normalize_resolution_arg is normalize_resolution_arg
        assert consumer.rgb2id is rgb2id
    assert augmentations.get_config_value is get_config_value
    assert reasoning_data_module.get_config_value is get_config_value


@pytest.mark.cv_unit
def test_shared_image_utilities():
    """Shared helpers decode IDs and keep transformed modalities aligned."""
    encoded = np.asarray([[[3, 2, 1], [255, 255, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(rgb2id(encoded), [[66051, 16777215]])
    assert normalize_resolution_arg((8, 8)) == [(8, 8)]
    assert normalize_resolution_arg([[8, 8], [8, 6]]) == [(8, 8), (8, 6)]

    image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8))
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    panoptic = np.arange(24, dtype=np.int32).reshape(4, 6)
    intrinsics = np.asarray(
        [[6.0, 0.0, 3.0], [0.0, 6.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    output = center_crop_and_resize(
        image,
        intrinsics,
        depth,
        panoptic,
        (8, 8),
    )
    output_image, output_intrinsics, output_depth, output_panoptic = output

    assert output_image.size == (8, 8)
    assert output_intrinsics is intrinsics
    assert output_depth.shape == output_panoptic.shape == (8, 8)
    assert output_depth.dtype == np.float32
    assert output_panoptic.dtype == np.int32


@pytest.mark.cv_unit
def test_photometric_augmentation():
    """Augmentation is optional, deterministic, and geometry preserving."""
    assert GeometrySafePhotometricAugmentation.from_config(None) is None
    config = {
        "enabled": True,
        "color_jitter_prob": 1.0,
        "gamma_exposure_prob": 1.0,
        "grayscale_prob": 1.0,
    }
    augmentation = GeometrySafePhotometricAugmentation.from_config(config)
    recipe = augmentation.sample(random.Random(17))
    assert recipe == augmentation.sample(random.Random(17))

    image = Image.fromarray(
        np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    )
    first = augmentation.apply(image, recipe)
    second = augmentation.apply(image, recipe)
    assert first.size == image.size
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "config,error",
    (
        ({"brightness": -0.1}, "brightness must be non-negative"),
        ({"gamma": 1.0}, "gamma must be in"),
        ({"grayscale_prob": 1.1}, "grayscale_prob must be in"),
    ),
)
def test_invalid_augmentation(config, error):
    """Invalid augmentation ranges must fail before dataset iteration."""
    with pytest.raises(ValueError, match=error):
        GeometrySafePhotometricAugmentation(config)


@pytest.mark.cv_unit
def test_panoptic_paths(tmp_path):
    """Configured roots map directly to the two ScanNet++ splits."""
    roots = _make_panoptic_roots(tmp_path)
    config = SimpleNamespace(
        train_preprocessed_root=str(roots["train"]),
        val_preprocessed_root=str(roots["val"]),
    )

    assert resolve_panoptic_paths(config) == {
        "train": str(roots["train"]),
        "val": str(roots["val"]),
    }


@pytest.mark.cv_unit
def test_panoptic_path_validation(tmp_path):
    """Missing metadata and inconsistent taxonomies must fail early."""
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_root.mkdir()
    val_root.mkdir()
    config = SimpleNamespace(
        train_preprocessed_root=str(train_root),
        val_preprocessed_root=str(val_root),
    )
    with pytest.raises(FileNotFoundError, match="all_metadata.npz"):
        resolve_panoptic_paths(config)

    roots = _make_panoptic_roots(tmp_path / "valid")
    (roots["train"] / "categories.json").write_text(
        '[{"id": 0, "name": "chair", "isthing": true}]',
        encoding="utf-8",
    )
    (roots["val"] / "categories.json").write_text(
        '[{"id": 0, "name": "table", "isthing": true}]',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        train_preprocessed_root=str(roots["train"]),
        val_preprocessed_root=str(roots["val"]),
    )
    with pytest.raises(ValueError, match="same categories.json taxonomy"):
        resolve_vocabulary(config)


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "num_workers,timeout,persistent",
    ((0, 0, False), (1, 600, True)),
)
def test_panoptic_loader_workers(
    monkeypatch,
    num_workers,
    timeout,
    persistent,
):
    """One worker setting must apply consistently to both splits."""
    data_module = object.__new__(NVPanoptix3Dv2PanopticDataModule)
    data_module.data_cfg = SimpleNamespace(
        batch_size=1,
        num_workers=num_workers,
    )
    monkeypatch.setattr(
        data_module,
        "build_dataset",
        lambda split: _TinyDataset(),
    )

    for split, shuffle in (("train", True), ("val", False)):
        loader = data_module.common_dataloader(split, shuffle)
        assert loader.num_workers == num_workers
        assert loader.timeout == timeout
        assert loader.persistent_workers is persistent


@pytest.mark.cv_unit
def test_panoptic_validation_split(monkeypatch, tmp_path):
    """Validation uses its configured root, both cameras, and the train seed."""
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return _TinyDataset()

    monkeypatch.setattr(
        panoptic_data_module,
        "ScanNetppDataset",
        fake_dataset,
    )
    roots = _make_panoptic_roots(tmp_path)
    data_module = object.__new__(NVPanoptix3Dv2PanopticDataModule)
    data_module.data_cfg = SimpleNamespace(
        train_preprocessed_root=str(roots["train"]),
        val_preprocessed_root=str(roots["val"]),
        resolution=[[518, 518]],
        num_views=5,
        pairs_per_scene=1,
        randomize_view_order=True,
        photometric_augmentation=object(),
    )
    data_module.seed = 17

    data_module.build_dataset("val")

    assert captured["preprocessed_root"] == str(roots["val"])
    assert captured["split"] == "val"
    assert captured["seed"] == 17
    assert captured["photometric_augmentation"] is None
    assert "camera_filter" not in captured
    assert "pairs_root" not in captured


@pytest.mark.cv_unit
def test_panoptic_collator_vocabulary():
    """ScanNet++ batches carry only their native class vocabulary."""
    sample = [[{"img": torch.zeros(3, 2, 2), "dataset": "ScanNet++"}]]
    collated = ScanNetppCollator(["chair"])(sample)

    assert collated[0]["vocab"] == ["chair"]
    assert "vocab_ignore" not in collated[0]


@pytest.mark.cv_unit
def test_reasoning_manifest_contract(tmp_path):
    """Manifest assets remain portable and answers are not rewritten."""
    manifest_dir = str(tmp_path / "reasoning")
    assert resolve_manifest_path("scene/v0.jpg", manifest_dir) == str(
        tmp_path / "reasoning" / "scene" / "v0.jpg"
    )
    assert resolve_manifest_path("/data/scene/v0.jpg", manifest_dir) == (
        "/data/scene/v0.jpg"
    )
    records = [
        {
            "images": ["scene/v0.jpg"],
            "depth": ["scene/v0_depth.png"],
            "instruction": "Segment the chair.",
            "answer": "Original [SEG] answer.",
            "target_inst_id": 1,
            "depth_scale": 500.0,
        },
        {
            "images": ["scene/v1.jpg"],
            "instruction": "Describe the room.",
            "answer": "Original text-only answer.",
            "target_inst_id": 0,
        },
    ]

    loaded = _load_manifest(tmp_path, records)

    assert [record["answer"] for record in loaded] == [
        "Original [SEG] answer.",
        "Original text-only answer.",
    ]
    assert loaded[0]["depth"] == [str(tmp_path / "scene" / "v0_depth.png")]
    assert loaded[1]["depth"] == [None]
    assert [record["_depth_scale"] for record in loaded] == [500.0, 1000.0]


@pytest.mark.cv_unit
@pytest.mark.parametrize("num_workers", (None, 3))
def test_reasoning_loader_workers(monkeypatch, num_workers):
    """The reasoning loader uses one worker setting for train and validation."""
    captured = []

    def fake_dataset(**kwargs):
        captured.append(kwargs)
        return _TinyDataset()

    monkeypatch.setattr(
        reasoning_data_module,
        "ReasoningSegDataset",
        fake_dataset,
    )
    values = {
        "train_manifest": "train.jsonl",
        "val_manifest": "val.jsonl",
    }
    if num_workers is not None:
        values["num_workers"] = num_workers
    data_module = NVPanoptix3Dv2ReasoningDataModule(
        SimpleNamespace(**values)
    )
    expected = 4 if num_workers is None else num_workers

    assert data_module.train_dataloader().num_workers == expected
    assert data_module.val_dataloader().num_workers == expected
    assert captured[0]["num_views"] == 5
    assert captured[0]["depth_scale"] == 1000.0
    assert captured[0]["require_seg_token"] is True


@pytest.mark.cv_unit
def test_reasoning_manifest_required(monkeypatch):
    """Reasoning cannot silently fall back to a directory-style dataset."""
    monkeypatch.setattr(
        reasoning_data_module,
        "ReasoningSegDataset",
        lambda **kwargs: _TinyDataset(),
    )

    with pytest.raises(ValueError, match="train_manifest is required"):
        NVPanoptix3Dv2ReasoningDataModule(
            SimpleNamespace(train_manifest=None)
        )
