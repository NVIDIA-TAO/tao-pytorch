# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, same-filesystem publication for DINOv3 artifacts."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_path(destination):
    """Yield a private path, then sync and replace; clean up only our own file."""
    destination = Path(destination)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                        dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(destination, value):
    """Publish UTF-8 text durably."""
    with atomic_path(destination) as temporary:
        temporary.write_text(value, encoding="utf-8")


def atomic_json(destination, value):
    """Publish canonical, human-readable JSON durably."""
    atomic_text(destination, json.dumps(value, indent=2, sort_keys=True) + "\n")
