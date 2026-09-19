# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-safe local file and archive image access for DINOv3."""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import os
from pathlib import Path
import tarfile
from urllib.parse import unquote, urlparse
import zipfile

from PIL import Image


class LocalImageReader:
    """Read local files and archive members with process-local handle caching."""

    def __init__(self, root: str | Path = ".", archive_cache_size: int = 8):
        if archive_cache_size <= 0:
            raise ValueError("archive_cache_size must be positive")
        self.root = Path(root)
        self.archive_cache_size = int(archive_cache_size)
        self._handles: OrderedDict[tuple[str, str], object] = OrderedDict()
        self._tar_members: dict[tuple[str, str], dict[str, tarfile.TarInfo]] = {}
        self._owner_pid = os.getpid()

    def __getstate__(self):
        """Do not pickle process-owned archive descriptors into workers."""
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        state["_tar_members"] = {}
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        """Close process-owned archive descriptors."""
        for archive in getattr(self, "_handles", {}).values():
            try:
                archive.close()
            except Exception:  # pragma: no cover - interpreter shutdown safety
                pass
        if hasattr(self, "_handles"):
            self._handles.clear()
        if hasattr(self, "_tar_members"):
            self._tar_members.clear()

    def __del__(self):
        """Release process-owned archive handles during worker teardown."""
        self.close()

    def resolve(self, configured: str) -> Path:
        """Resolve an absolute/file URI or a path relative to the dataset root."""
        parsed = urlparse(configured)
        if parsed.scheme:
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise ValueError(f"Only local file locators are supported: {configured}")
            if parsed.params or parsed.query or parsed.fragment:
                raise ValueError(f"File URI modifiers are unsupported: {configured}")
            return Path(unquote(parsed.path))
        path = Path(configured)
        return path if path.is_absolute() else self.root / path

    def _ensure_process_owner(self) -> None:
        owner_pid = os.getpid()
        if self._owner_pid == owner_pid:
            return
        self.close()
        self._owner_pid = owner_pid

    def _archive(self, storage_type: str, path: Path):
        self._ensure_process_owner()
        key = (storage_type, str(path))
        archive = self._handles.pop(key, None)
        if archive is None:
            if storage_type == "tar":
                if path.suffix.lower() != ".tar":
                    raise ValueError(
                        "Random archive-member access requires uncompressed .tar "
                        f"shards; stage or reshard {path}"
                    )
                archive = tarfile.open(path, mode="r:")
                if key not in self._tar_members:
                    self._tar_members[key] = {
                        member.name: member for member in archive.getmembers()
                    }
            elif storage_type == "zip":
                archive = zipfile.ZipFile(path)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            while len(self._handles) >= self.archive_cache_size:
                stale_key, stale = self._handles.popitem(last=False)
                stale.close()
                self._tar_members.pop(stale_key, None)
        self._handles[key] = archive
        return archive

    def image(self, storage_type: str, configured: str, member: str | None) -> Image.Image:
        """Load one RGB image from a file, uncompressed tar, or zip locator."""
        path = self.resolve(configured)
        if storage_type == "file":
            return Image.open(path, mode="r").convert("RGB")
        if member is None:
            raise ValueError("Archive-backed rows require a member")
        archive = self._archive(storage_type, path)
        if storage_type == "zip":
            return Image.open(BytesIO(archive.read(member))).convert("RGB")
        member_info = self._tar_members[(storage_type, str(path))].get(member)
        stream = None if member_info is None else archive.extractfile(member_info)
        if stream is None:
            raise FileNotFoundError(f"Tar member does not exist: {path}::{member}")
        return Image.open(BytesIO(stream.read())).convert("RGB")
