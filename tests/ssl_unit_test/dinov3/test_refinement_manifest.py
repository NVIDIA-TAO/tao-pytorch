# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DINOv3-owned manifest loading and checkpoint retention."""

from pathlib import Path
import os
import pickle
import tarfile
import zipfile

import pandas as pd
from PIL import Image
import pytest
from torch.utils.data import DataLoader

from nvidia_tao_pytorch.ssl.dinov3.dataloader.dataset import (
    DinoV3Dataset,
    ShardAwareDistributedSampler,
)
from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint_retention import (
    prune_periodic_ssl_checkpoints,
)


def _transform(image):
    return {"global_crops": [image.getpixel((0, 0))], "local_crops": []}


def _manifest(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_directory_mode_preserves_base_dataset_behavior(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4), "red").save(tmp_path / "image.png")
    dataset = DinoV3Dataset(root=tmp_path, transform=_transform)
    assert len(dataset) == 1
    assert dataset[0]["global_crops"] == [(255, 0, 0)]


def test_dataset_reads_exact_manifest_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)
    manifest = _manifest(tmp_path / "train.parquet", [
        {"sample_id": "b", "storage_type": "file", "path": str(second)},
        {"sample_id": "a", "storage_type": "file", "path": str(first)},
    ])
    dataset = DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)
    assert dataset[0]["global_crops"] == [(0, 0, 255)]
    assert dataset[1]["global_crops"] == [(255, 0, 0)]


@pytest.mark.parametrize("missing", ["sample_id", "storage_type", "path"])
def test_manifest_requires_canonical_columns(tmp_path: Path, missing: str) -> None:
    row = {"sample_id": "a", "storage_type": "file", "path": "image.png"}
    row.pop(missing)
    manifest = _manifest(tmp_path / "invalid.parquet", [row])
    with pytest.raises(ValueError, match="missing columns"):
        DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)


def test_manifest_requires_unique_sample_ids(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "duplicate.parquet", [
        {"sample_id": "a", "storage_type": "file", "path": "first.png"},
        {"sample_id": "a", "storage_type": "file", "path": "second.png"},
    ])
    with pytest.raises(ValueError, match="unique and non-empty"):
        DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)


def test_manifest_accepts_ds_balanced_replay_contract(tmp_path: Path) -> None:
    """Balanced views repeat locators only through replay_repeat identities."""
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)
    manifest = _manifest(tmp_path / "balanced.parquet", [
        {
            "sample_id": "minority",
            "storage_type": "file",
            "path": str(first),
            "replay_repeat": 0,
        },
        {
            "sample_id": "minority",
            "storage_type": "file",
            "path": str(first),
            "replay_repeat": 1,
        },
        {
            "sample_id": "majority",
            "storage_type": "file",
            "path": str(second),
            "replay_repeat": 0,
        },
    ])

    dataset = DinoV3Dataset(
        root=tmp_path, manifest_path=manifest, transform=_transform
    )

    assert len(dataset) == 3
    assert [record["replay_repeat"] for record in dataset.all_images] == [0, 1, 0]
    assert dataset[0]["global_crops"] == dataset[1]["global_crops"]


@pytest.mark.parametrize(("repeats", "message"), [
    ([0, 0], "replay identities must be unique"),
    ([1], "contiguous from zero"),
    ([0, 2], "contiguous from zero"),
    ([0, -1], "nonnegative integer"),
    ([0.0, 1.5], "nonnegative integer"),
])
def test_manifest_rejects_invalid_replay_identities(
    tmp_path: Path, repeats, message
) -> None:
    manifest = _manifest(tmp_path / "invalid-replay.parquet", [
        {
            "sample_id": "replayed",
            "storage_type": "file",
            "path": "same.png",
            "replay_repeat": repeat,
        }
        for repeat in repeats
    ])
    with pytest.raises(ValueError, match=message):
        DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)


def test_manifest_replay_requires_invariant_locator(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "changed-locator.parquet", [
        {
            "sample_id": "replayed", "storage_type": "file",
            "path": "first.png", "replay_repeat": 0,
        },
        {
            "sample_id": "replayed", "storage_type": "file",
            "path": "second.png", "replay_repeat": 1,
        },
    ])
    with pytest.raises(ValueError, match="one locator"):
        DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)


@pytest.mark.parametrize("column", ["sample_id", "storage_type", "path"])
def test_manifest_rejects_null_identity_or_locator(tmp_path: Path, column: str) -> None:
    row = {"sample_id": "a", "storage_type": "file", "path": "image.png"}
    row[column] = None
    manifest = _manifest(tmp_path / "null.parquet", [row])
    with pytest.raises(ValueError, match="cannot be null"):
        DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)


@pytest.mark.parametrize("storage_type", ["tar", "zip"])
def test_dataset_reads_archive_member(tmp_path: Path, storage_type: str) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image_path)
    member = "images/image.png"
    archive_path = tmp_path / f"images.{storage_type}"
    if storage_type == "tar":
        with tarfile.open(archive_path, "w") as archive:
            archive.add(image_path, arcname=member)
    else:
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(image_path, arcname=member)
    manifest = _manifest(tmp_path / "archive.parquet", [{
        "sample_id": "archive-image",
        "storage_type": storage_type,
        "path": archive_path.as_uri(),
        "member": member,
    }])
    dataset = DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)
    assert dataset[0]["global_crops"] == [(0, 128, 0)]


def test_tar_member_index_is_bounded_with_handle_eviction(tmp_path: Path, monkeypatch) -> None:
    paths = []
    for name, color in (("a", "red"), ("b", "blue")):
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (4, 4), color).save(image)
        archive_path = tmp_path / f"{name}.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(image, arcname="image.png")
        paths.append(archive_path)
    manifest = _manifest(tmp_path / "archives.parquet", [
        {"sample_id": str(index), "storage_type": "tar", "path": str(path), "member": "image.png"}
        for index, path in enumerate(paths)
    ])
    calls = 0
    original = tarfile.TarFile.getmembers

    def counted(archive):
        nonlocal calls
        calls += 1
        return original(archive)

    monkeypatch.setattr(tarfile.TarFile, "getmembers", counted)
    dataset = DinoV3Dataset(
        root=tmp_path, manifest_path=manifest, transform=_transform, archive_cache_size=1
    )
    dataset[0]
    dataset[1]
    dataset[0]
    assert calls == 3
    assert len(dataset._reader._handles) == 1
    assert len(dataset._reader._tar_members) == 1


def test_archive_descriptors_reopen_after_fork(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image)
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(image, arcname="image.png")
    manifest = _manifest(tmp_path / "archive.parquet", [{
        "sample_id": "image", "storage_type": "tar",
        "path": str(archive_path), "member": "image.png",
    }])
    dataset = DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)
    dataset[0]
    parent_handle = next(iter(dataset._reader._handles.values()))
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    dataset[0]
    assert next(iter(dataset._reader._handles.values())) is not parent_handle


def test_archive_reader_spawn_serialization_drops_process_state(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image)
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(image, arcname="image.png")
    manifest = _manifest(tmp_path / "archive.parquet", [{
        "sample_id": "image", "storage_type": "tar",
        "path": str(archive_path), "member": "image.png",
    }])
    dataset = DinoV3Dataset(
        root=tmp_path, manifest_path=manifest, transform=_transform
    )
    dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._reader._handles
    assert not restored._reader._tar_members
    assert restored[0]["global_crops"] == [(0, 128, 0)]


def test_archive_reader_runs_in_spawned_dataloader_worker(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image)
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(image, arcname="image.png")
    manifest = _manifest(tmp_path / "archive.parquet", [{
        "sample_id": "image", "storage_type": "tar",
        "path": str(archive_path), "member": "image.png",
    }])
    dataset = DinoV3Dataset(
        root=tmp_path, manifest_path=manifest, transform=_transform
    )
    assert dataset[0]["global_crops"] == [(0, 128, 0)]
    assert dataset._reader._handles
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    iterator = iter(loader)
    try:
        assert next(iterator)["global_crops"] == [[0, 128, 0]]
    finally:
        iterator._shutdown_workers()


def test_dataset_rejects_compressed_random_access_tar(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image)
    archive_path = tmp_path / "images.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(image, arcname="image.png")
    manifest = _manifest(tmp_path / "archive.parquet", [{
        "sample_id": "image", "storage_type": "tar",
        "path": str(archive_path), "member": "image.png",
    }])
    dataset = DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)
    with pytest.raises(ValueError, match="uncompressed"):
        dataset[0]


def test_dataset_rejects_remote_file_uri_authority(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "remote.parquet", [{
        "sample_id": "remote", "storage_type": "file",
        "path": "file://remote-host/data/image.png",
    }])
    dataset = DinoV3Dataset(root=tmp_path, manifest_path=manifest, transform=_transform)
    with pytest.raises(ValueError, match="Only local file locators"):
        dataset[0]


def test_shard_aware_sampler_preserves_windows_and_equalizes_ranks() -> None:
    class Dataset:
        groups = ["tar:a"] * 5 + ["tar:b"] * 5 + ["file:c"]

        def __len__(self):
            return len(self.groups)

        def sampling_group(self, index):
            return self.groups[index]

    partitions = [list(ShardAwareDistributedSampler(
        Dataset(), num_replicas=3, rank=rank, shuffle=False, shuffle_window=2
    )) for rank in range(3)]
    assert {len(values) for values in partitions} == {4}
    assert set().union(*map(set, partitions)) == set(range(11))


def test_shard_aware_sampler_pads_ranks_when_dataset_is_smaller() -> None:
    class Dataset:
        def __len__(self):
            return 2

        @staticmethod
        def sampling_group(index):
            return f"file:{index}"

    partitions = [list(ShardAwareDistributedSampler(
        Dataset(), num_replicas=4, rank=rank, shuffle=False
    )) for rank in range(4)]
    assert {len(values) for values in partitions} == {1}
    assert all(value[0] in {0, 1} for value in partitions)


def test_periodic_checkpoint_retention_keeps_newest_families(tmp_path: Path) -> None:
    for step in (1, 2, 3):
        for prefix in ("model_epoch", "student_epoch", "teacher_epoch"):
            (tmp_path / f"{prefix}_{step:03d}_step_{step:05d}.pth").write_text(
                str(step), encoding="utf-8"
            )
    prune_periodic_ssl_checkpoints(str(tmp_path), keep_last_n=2)
    for prefix in ("model_epoch", "student_epoch", "teacher_epoch"):
        assert [path.read_text(encoding="utf-8") for path in sorted(
            tmp_path.glob(f"{prefix}_*.pth")
        )] == ["2", "3"]


def test_checkpoint_retention_ignores_similarly_named_files(tmp_path: Path) -> None:
    valid = tmp_path / "teacher_epoch_001_step_00002.pth"
    versioned = tmp_path / "teacher_epoch_001_step_00002-v1.pth"
    valid.write_text("valid", encoding="utf-8")
    versioned.write_text("unrelated", encoding="utf-8")
    prune_periodic_ssl_checkpoints(str(tmp_path), keep_last_n=1)
    assert valid.read_text(encoding="utf-8") == "valid"
    assert versioned.read_text(encoding="utf-8") == "unrelated"
