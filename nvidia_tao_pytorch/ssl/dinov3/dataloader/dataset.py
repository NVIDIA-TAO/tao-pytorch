# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 manifest dataset"""
from io import BytesIO
from collections import defaultdict, OrderedDict
import logging
import os
from pathlib import Path
import tarfile
from typing import Iterable, Optional, Union
from urllib.parse import unquote, urlparse
import zipfile

from PIL import Image
import torch
from torch.utils.data import Dataset, distributed

logger = logging.getLogger(__name__)

ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".zip")


class ShardAwareDistributedSampler(distributed.DistributedSampler):
    """Shuffle shards and rows while retaining archive locality per rank."""

    def _rank_assignments(self):
        cached = getattr(self, "_shard_rank_assignments", None)
        if cached is not None:
            return cached
        grouped = defaultdict(list)
        for index in range(len(self.dataset)):
            grouped[self.dataset.sampling_group(index)].append(index)
        assignments = [[] for _ in range(self.num_replicas)]
        if self.drop_last:
            epoch_size = len(self.dataset) // self.num_replicas
        else:
            epoch_size = (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
        remaining_capacity = [epoch_size] * self.num_replicas
        for name, rows in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            offset = 0
            while offset < len(rows):
                remaining_rows = len(rows) - offset
                whole_fit = [
                    rank
                    for rank, capacity in enumerate(remaining_capacity)
                    if capacity >= remaining_rows
                ]
                if whole_fit:
                    rank = min(
                        whole_fit,
                        key=lambda value, required=remaining_rows: (
                            remaining_capacity[value] - required,
                            value,
                        ),
                    )
                    count = remaining_rows
                else:
                    rank = max(
                        range(self.num_replicas),
                        key=lambda value: (remaining_capacity[value], -value),
                    )
                    count = min(remaining_rows, remaining_capacity[rank])
                if count == 0:
                    break
                assignments[rank].append(
                    (f"{name}#{offset}", rows[offset:offset + count])
                )
                remaining_capacity[rank] -= count
                offset += count
        for rank, capacity in enumerate(remaining_capacity):
            if capacity == epoch_size and epoch_size:
                donor = max(
                    range(self.num_replicas),
                    key=lambda value: epoch_size - remaining_capacity[value],
                )
                _, donor_rows = assignments[donor][0]
                assignments[rank].append((f"padding:{rank}", donor_rows[:1]))
                remaining_capacity[rank] -= 1
        loads = [epoch_size - capacity for capacity in remaining_capacity]
        if epoch_size and not all(loads):
            raise RuntimeError("Shard-aware sampling could not populate every rank")
        result = (assignments, epoch_size)
        self._shard_rank_assignments = result
        return result

    def __len__(self):
        """Return the equalized number of rows consumed by every rank."""
        return self._rank_assignments()[1]

    def __iter__(self):
        """Yield one deterministic, shard-local partition for this epoch."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        assignments, epoch_size = self._rank_assignments()
        if epoch_size == 0:
            return iter(())
        groups = assignments[self.rank]
        group_order = (
            torch.randperm(len(groups), generator=generator).tolist()
            if self.shuffle else range(len(groups))
        )
        shuffle_window = int(os.environ.get("TAO_SHARD_SHUFFLE_WINDOW", "256"))
        if shuffle_window <= 0:
            raise ValueError("TAO_SHARD_SHUFFLE_WINDOW must be positive")
        indices = []
        for group_index in group_order:
            _, rows = groups[group_index]
            for start in range(0, len(rows), shuffle_window):
                stop = min(len(rows), start + shuffle_window)
                row_order = (
                    torch.randperm(stop - start, generator=generator).tolist()
                    if self.shuffle else range(stop - start)
                )
                indices.extend(rows[start + position] for position in row_order)
        if self.drop_last:
            indices = indices[:epoch_size]
        else:
            padding = epoch_size - len(indices)
            indices += (indices * ((padding // len(indices)) + 1))[:padding]
        if len(indices) != epoch_size:
            raise RuntimeError("Shard-aware sampler produced an invalid partition")
        return iter(indices)


class DinoV3Dataset(Dataset):
    """Dataset for NVDINOv2 to manage and transform image data for training, with support for various image formats."""

    def __init__(
        self,
        *,
        root: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
        transform: Optional[callable] = None,
        train: bool = True,
        extensions: Iterable[str] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
    ):
        """Initializes the dataset with the root directory, optional transformations, and valid image extensions.

        Args:
            root (Union[str, Path]): The root directory containing the image files
            transform (Optional[callable], optional): A transformation function to apply to the images. Required for data processing. Defaults to None.
            extensions (Iterable[str], optional): A list of valid image file extensions. Defaults include common image formats. Defaults to ( ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ).
        """
        self.root = Path(root)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.extensions = extensions
        self.transform = transform
        self.train = train
        self._archive_handles = OrderedDict()
        self._archive_member_indices = {}
        self._archive_owner_pid = os.getpid()
        self._archive_cache_size = int(os.environ.get("TAO_ARCHIVE_HANDLE_CACHE", "8"))
        if self._archive_cache_size <= 0:
            raise ValueError("TAO_ARCHIVE_HANDLE_CACHE must be positive")

        assert self.transform is not None, "Transform must be specified."

        if self.manifest_path is not None:
            if not self.manifest_path.is_file():
                raise FileNotFoundError(
                    f"train_manifest does not exist: {self.manifest_path}"
                )
            self.all_images = self._read_manifest()
        elif not self.root.exists():
            raise FileNotFoundError(f"images_dir does not exist: {self.root}")
        elif self.root.is_file():
            if self.root.name.lower().endswith(ARCHIVE_SUFFIXES):
                raise ValueError(
                    f"images_dir must be an extracted image directory, not an archive ({self.root}). "
                    "Extract it first (e.g. `tar -xzf`) and point images_dir at the resulting folder."
                )
            raise ValueError(f"images_dir must be a directory, but got a file: {self.root}")

        else:
            self.all_images = self._list_images()

        if not self.all_images:
            raise ValueError(f"No images found in {self.root}.")

    def _list_images(self):
        """Lists all image paths in the specified root directory.

        Returns:
            List: List of all image paths relative to the root directory.
        """
        return [
            str(f.relative_to(self.root))
            for f in self.root.rglob("*")
            if f.suffix.lower() in self.extensions
        ]

    def _read_manifest(self):
        """Read ordered canonical local-file or archive-member records."""
        import pyarrow.parquet as pq

        # Keep compute threading, but avoid asynchronous prefetch teardown.
        table = pq.read_table(self.manifest_path, pre_buffer=False)
        columns = set(table.column_names)
        required = {"storage_type", "path"}
        if required.issubset(columns):
            storage_types = table.column("storage_type").to_pylist()
            paths = table.column("path").to_pylist()
            members = (
                table.column("member").to_pylist()
                if "member" in columns
                else [None] * len(paths)
            )
            records = []
            for storage_type, path, member in zip(storage_types, paths, members):
                storage_type = str(storage_type)
                if storage_type not in {"file", "tar", "zip"}:
                    raise ValueError(
                        f"Unsupported train_manifest storage_type: {storage_type}"
                    )
                if storage_type != "file" and member is None:
                    raise ValueError("Archive-backed manifest rows require member")
                records.append(
                    {
                        "storage_type": storage_type,
                        "path": str(path),
                        "member": None if member is None else str(member),
                    }
                )
            return records
        raise ValueError(
            "train_manifest must contain canonical storage_type/path/member columns"
        )

    def __getstate__(self):
        """Do not pickle process-owned archive descriptors into workers."""
        value = dict(self.__dict__)
        value["_archive_handles"] = OrderedDict()
        value["_archive_member_indices"] = {}
        return value

    def __del__(self):
        """Close any process-owned archive descriptors."""
        for archive in getattr(self, "_archive_handles", {}).values():
            try:
                archive.close()
            except Exception:  # pragma: no cover - interpreter shutdown safety
                pass

    def _resolved_path(self, configured: str) -> Path:
        parsed = urlparse(configured)
        if parsed.scheme:
            if parsed.scheme != "file":
                raise ValueError(f"Only local file locators are supported: {configured}")
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError(
                    f"Remote file URI authorities are unsupported: {configured}"
                )
            if parsed.params or parsed.query or parsed.fragment:
                raise ValueError(f"File URI modifiers are unsupported: {configured}")
            return Path(unquote(parsed.path))
        path = Path(configured)
        return path if path.is_absolute() else self.root / path

    def _archive(self, storage_type: str, path: Path):
        # Linux DataLoader workers fork by default, bypassing __getstate__.
        # Reopen inherited descriptors so workers never share a seek offset.
        owner_pid = os.getpid()
        if self._archive_owner_pid != owner_pid:
            for inherited in self._archive_handles.values():
                inherited.close()
            self._archive_handles.clear()
            self._archive_member_indices.clear()
            self._archive_owner_pid = owner_pid
        key = (storage_type, str(path))
        archive = self._archive_handles.pop(key, None)
        if archive is None:
            if storage_type == "tar":
                if path.suffix.lower() != ".tar":
                    raise ValueError(
                        "Random archive-member training requires uncompressed .tar "
                        f"shards; stage or reshard {path}"
                    )
                archive = tarfile.open(path, mode="r:")
                self._archive_member_indices[key] = {
                    member.name: member for member in archive.getmembers()
                }
            else:
                archive = zipfile.ZipFile(path)
            while len(self._archive_handles) >= self._archive_cache_size:
                stale_key, stale = self._archive_handles.popitem(last=False)
                self._archive_member_indices.pop(stale_key, None)
                stale.close()
        self._archive_handles[key] = archive
        return archive

    def __len__(self):
        """Returns the number of images in the dataset.

        Returns:
            Int: The total count of images in the dataset.
        """
        return len(self.all_images)

    def sampling_group(self, index: int) -> str:
        """Return a locality group for shard-aware manifest sampling."""
        record = self.all_images[index]
        if isinstance(record, dict) and record["storage_type"] in {"tar", "zip"}:
            return f"{record['storage_type']}:{record['path']}"
        return f"row:{index}"

    def _get_item_internal_(self, idx):
        """Retrieves and transforms an image at a given index.

        Args:
            idx (Int): Index of the image to retrieve.

        Returns:
            Dict: A dictionary containing global and local crops of the image.
        """
        record = self.all_images[idx]
        if isinstance(record, str):
            img_path = self.root / record
            input_path = record
            image = Image.open(img_path, mode="r").convert("RGB")
        elif record["storage_type"] == "file":
            configured = str(record["path"])
            img_path = self._resolved_path(configured)
            input_path = configured
            image = Image.open(img_path, mode="r").convert("RGB")
        else:
            configured = str(record["path"])
            archive_path = self._resolved_path(configured)
            input_path = f"{configured}::{record['member']}"
            if record["storage_type"] == "tar":
                archive = self._archive("tar", archive_path)
                member = self._archive_member_indices[
                    ("tar", str(archive_path))
                ].get(record["member"])
                stream = None if member is None else archive.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(
                        f"Tar member does not exist: {archive_path}::{record['member']}"
                    )
                image = Image.open(BytesIO(stream.read())).convert("RGB")
            else:
                archive = self._archive("zip", archive_path)
                image = Image.open(
                    BytesIO(archive.read(record["member"]))
                ).convert("RGB")
        images = self.transform(image)
        if self.train:
            return {
                "global_crops": images["global_crops"],
                "local_crops": images["local_crops"],
            }
        return {
            "images": images,
            "input_path": input_path
        }

    def __getitem__(self, idx):
        """Retrieves an item (image) at a given index with error handling.

        Args:
            idx (Int): Index of the image to retrieve.

        Returns:
            Dict: A dictionary containing global and local crops of the image.
        """
        try:
            return self._get_item_internal_(idx)
        except Exception as e:
            logger.error(f"Error retrieving image record {self.all_images[idx]}: {e}")
            raise
