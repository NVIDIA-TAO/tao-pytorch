# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash and competing-writer tests for DINOv3 artifact publication."""

import pytest

from nvidia_tao_pytorch.ssl.dinov3.utils.atomic import atomic_path


def test_overlapping_writers_use_private_temporary_files(tmp_path):
    """One writer must not overwrite or unlink another writer's temporary."""
    destination = tmp_path / "checkpoint.pth"
    with atomic_path(destination) as first:
        first.write_bytes(b"first")
        with atomic_path(destination) as second:
            assert first != second
            second.write_bytes(b"second")
        assert destination.read_bytes() == b"second"
        assert first.read_bytes() == b"first"
    assert destination.read_bytes() == b"first"
    assert list(tmp_path.iterdir()) == [destination]


def test_failed_writer_preserves_previous_publication(tmp_path):
    """A serialization error must leave the old complete artifact intact."""
    destination = tmp_path / "checkpoint.pth"
    destination.write_bytes(b"old")
    with pytest.raises(ValueError, match="serialization failed"):
        with atomic_path(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise ValueError("serialization failed")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]
