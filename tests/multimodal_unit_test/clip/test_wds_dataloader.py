# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP WebDataset dataloader."""

import io
import tarfile

import pytest
from PIL import Image

from nvidia_tao_pytorch.multimodal.clip.dataloader.wds import (
    ResumableShardList,
    get_train_dataloader,
    group_by_keys_nothrow,
)


def _jpg_bytes(color):
    """Create a small JPEG payload."""
    image = Image.new("RGB", (16, 16), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_bytes(tar, name, data):
    """Add an in-memory file to a tar archive."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_wds_shard(path, sample_ids):
    """Create a minimal image-caption WebDataset shard."""
    with tarfile.open(path, "w") as tar:
        for sample_id in sample_ids:
            key = f"{sample_id:06d}"
            _add_bytes(tar, f"{key}.jpg", _jpg_bytes((sample_id, 40, 80)))
            _add_bytes(tar, f"{key}.txt", f"caption {key}".encode("utf-8"))


@pytest.fixture
def wds_dataset(tmp_path):
    """Create a small WebDataset directory with shard list files."""
    root = tmp_path / "wds"
    root.mkdir()

    _make_wds_shard(root / "shard_000.tar", [0, 1])
    _make_wds_shard(root / "shard_001.tar", [2, 3])

    root_shard_list = root / "shards.txt"
    root_shard_list.write_text("shard_000.tar\nshard_001.tar\n")

    external_dir = tmp_path / "lists"
    external_dir.mkdir()
    external_shard_list = external_dir / "shards.txt"
    external_shard_list.write_text("shard_000.tar\nshard_001.tar\n")

    return {
        "root": root,
        "root_shard_list": root_shard_list,
        "external_shard_list": external_shard_list,
    }


@pytest.mark.multimodal_unit
class TestResumableShardList:
    """Test CLIP WebDataset shard path resolution."""

    def test_discovers_root_dir_and_shard_lists(self, wds_dataset):
        """Test root and shard-list WebDataset path resolution."""
        root = wds_dataset["root"]

        root_urls = ResumableShardList(root=str(root)).urls
        parent_urls = ResumableShardList(
            urls=str(wds_dataset["root_shard_list"])
        ).urls
        explicit_root_urls = ResumableShardList(
            urls=str(wds_dataset["external_shard_list"]),
            root=str(root),
        ).urls

        expected = sorted([str(root / "shard_000.tar"), str(root / "shard_001.tar")])
        assert sorted(root_urls) == expected
        assert sorted(parent_urls) == expected
        assert sorted(explicit_root_urls) == expected

    def test_validates_missing_and_empty_inputs(self, tmp_path):
        """Test invalid WebDataset path configuration errors."""
        with pytest.raises(ValueError, match="Either 'urls' or 'root' must be provided"):
            ResumableShardList()

        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        with pytest.raises(ValueError, match="No valid shard URLs found"):
            ResumableShardList(root=str(empty_root))


@pytest.mark.multimodal_unit
class TestClipWDSDataloader:
    """Test CLIP WebDataset sample grouping and loading."""

    def test_group_by_keys_ignores_empty_shard_boundaries(self):
        """Test empty WebDataset shard boundary records do not fail grouping."""
        records = [
            {"fname": "000001.jpg", "data": b"image-a", "__url__": "a.tar"},
            {"fname": "000001.txt", "data": b"caption-a", "__url__": "a.tar"},
            {},
            {"fname": "000002.jpg", "data": b"image-b", "__url__": "b.tar"},
            {"fname": "000002.txt", "data": b"caption-b", "__url__": "b.tar"},
        ]

        samples = list(group_by_keys_nothrow(records))

        assert len(samples) == 2
        assert samples[0]["__key__"] == "000001"
        assert samples[0]["jpg"] == b"image-a"
        assert samples[0]["txt"] == b"caption-a"
        assert samples[1]["__key__"] == "000002"
        assert samples[1]["jpg"] == b"image-b"
        assert samples[1]["txt"] == b"caption-b"

    def test_train_dataloader_yields_batch_across_shard_boundary(self, wds_dataset):
        """Test CLIP WebDataset train dataloader yields samples across shards."""
        def transform(sample):
            image, text = sample
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            return image.size, text

        dataloader = get_train_dataloader(
            root=str(wds_dataset["root"]),
            samples_per_file=2,
            batch_size=2,
            transform=transform,
            num_workers=0,
            pin_memory=False,
        )

        image_sizes, captions = next(iter(dataloader))

        assert len(image_sizes) == 2
        assert len(captions) == 2
        assert all(size == [16, 16] for size in image_sizes)
        assert all(caption.startswith("caption ") for caption in captions)
