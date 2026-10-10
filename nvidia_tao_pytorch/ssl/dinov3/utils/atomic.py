# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, same-filesystem publication for DINOv3 artifacts."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets


@contextmanager
def atomic_path(destination):
    """Yield a fresh same-directory path, then sync and replace; clean up only our own file."""
    destination = Path(destination)
    while True:
        # Unlike mkstemp's fixed 0600, mode 0666 honours the umask like any other
        # output, so a platform reading the shared results directory can open it.
        temporary = destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
        try:
            os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            break
        except FileExistsError:
            continue
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
