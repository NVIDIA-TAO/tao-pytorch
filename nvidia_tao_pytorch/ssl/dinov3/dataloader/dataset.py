# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 datasets for directories and canonical DEFT manifests."""

from collections import defaultdict
import logging
from pathlib import Path

import torch
from torch.utils.data import distributed

from nvidia_tao_pytorch.ssl.dinov3.dataloader.local_image import LocalImageReader
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.dataset import DinoV2Dataset

logger = logging.getLogger(__name__)


class ShardAwareDistributedSampler(distributed.DistributedSampler):
    """Shuffle shard-local windows while producing equal partitions per rank."""

    def __init__(self, *args, shuffle_window: int = 256, batch_size: int = 1, **kwargs):
        if shuffle_window <= 0:
            raise ValueError("shuffle_window must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        super().__init__(*args, **kwargs)
        if self.drop_last:
            raise ValueError("Manifest sampling must preserve all rows; use drop_last=False")
        # Pad each rank to full batches: a singleton tail is invalid for KoLeo.
        self.num_samples = ((self.num_samples + batch_size - 1) // batch_size) * batch_size
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle_window = int(shuffle_window)

    def _rank_assignments(self):
        cached = getattr(self, "_shard_rank_assignments", None)
        if cached is not None:
            return cached
        grouped = defaultdict(list)
        for index in range(len(self.dataset)):
            grouped[self.dataset.sampling_group(index)].append(index)
        assignments = [[] for _ in range(self.num_replicas)]
        epoch_size = self.num_samples
        capacity = [epoch_size] * self.num_replicas
        for name, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
            offset = 0
            while offset < len(rows):
                remaining = len(rows) - offset
                whole_fit = [rank for rank, free in enumerate(capacity) if free >= remaining]
                if whole_fit:
                    rank = min(
                        whole_fit,
                        key=lambda value, required=remaining: (
                            capacity[value] - required,
                            value,
                        ),
                    )
                    count = remaining
                else:
                    rank = max(range(self.num_replicas), key=lambda value: (capacity[value], -value))
                    count = min(remaining, capacity[rank])
                if count == 0:
                    break
                assignments[rank].append((f"{name}#{offset}", rows[offset:offset + count]))
                capacity[rank] -= count
                offset += count
        self._shard_rank_assignments = (assignments, epoch_size)
        return self._shard_rank_assignments

    def __iter__(self):
        """Yield a deterministic shard-local partition for the current epoch."""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        assignments, epoch_size = self._rank_assignments()
        groups = assignments[self.rank]
        group_order = (
            torch.randperm(len(groups), generator=generator).tolist()
            if self.shuffle else range(len(groups))
        )
        indices = []
        for group_index in group_order:
            _, rows = groups[group_index]
            for start in range(0, len(rows), self.shuffle_window):
                window = rows[start:start + self.shuffle_window]
                order = (
                    torch.randperm(len(window), generator=generator).tolist()
                    if self.shuffle else range(len(window))
                )
                indices.extend(window[position] for position in order)
        if self.drop_last:
            indices = indices[:epoch_size]
        else:
            padding = epoch_size - len(indices)
            if padding:
                padding_source = indices or list(range(len(self.dataset)))
                if not padding_source:
                    raise RuntimeError("Cannot sample an empty DINOv3 dataset")
                if self.shuffle:
                    order = torch.randperm(
                        len(padding_source), generator=generator
                    ).tolist()
                    padding_source = [padding_source[position] for position in order]
                indices += (
                    padding_source * ((padding // len(padding_source)) + 1)
                )[:padding]
        if len(indices) != epoch_size:
            raise RuntimeError("Shard-aware sampler produced an invalid partition")
        return iter(indices)


class DinoV3Dataset(DinoV2Dataset):
    """Use legacy directory loading or an ordered canonical DEFT manifest."""

    def __init__(self, *, root, manifest_path=None, transform=None, train=True,
                 archive_cache_size=8, **kwargs):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        if self.manifest_path is None:
            super().__init__(root=root, transform=transform, train=train, **kwargs)
            self._reader = None
            return
        if transform is None:
            raise ValueError("Transform must be specified")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"train_manifest does not exist: {self.manifest_path}")
        # Relative locators are rooted at the manifest, never the launcher's cwd.
        self.root = Path(root) if root else self.manifest_path.resolve().parent
        self.transform = transform
        self.train = train
        self._reader = LocalImageReader(self.root, archive_cache_size)
        self.all_images = self._read_manifest()
        if not self.all_images:
            raise ValueError(f"No images found in {self.manifest_path}")

    def _read_manifest(self):
        """Read ordered canonical file or archive-member records."""
        import pyarrow.parquet as pq

        table = pq.read_table(self.manifest_path, pre_buffer=False)
        required = {"sample_id", "storage_type", "path"}
        missing = required.difference(table.column_names)
        if missing:
            raise ValueError(f"train_manifest is missing columns: {sorted(missing)}")
        values = table.select(["sample_id", "storage_type", "path"]).to_pydict()
        members = (table.column("member").to_pylist()
                   if "member" in table.column_names else [None] * table.num_rows)
        replay_mode = "replay_repeat" in table.column_names
        replay_repeats = (
            table.column("replay_repeat").to_pylist()
            if replay_mode else [None] * table.num_rows
        )
        records = []
        seen = set()
        replay_locators = {}
        repeats_by_sample = {}
        for sample_id, storage_type, path, member, replay_repeat in zip(
                values["sample_id"], values["storage_type"], values["path"],
                members, replay_repeats):
            if sample_id is None or storage_type is None or path is None:
                raise ValueError("train_manifest identity and locator fields cannot be null")
            sample_id = str(sample_id)
            storage_type = str(storage_type)
            path = str(path)
            if not sample_id.strip():
                raise ValueError("train_manifest sample_id must be non-empty")
            if replay_mode and (
                isinstance(replay_repeat, bool) or
                not isinstance(replay_repeat, int) or
                replay_repeat < 0
            ):
                raise ValueError(
                    "train_manifest replay_repeat must be a nonnegative integer"
                )
            identity = (
                (sample_id, replay_repeat) if replay_mode else sample_id
            )
            if identity in seen:
                if replay_mode:
                    raise ValueError(
                        "train_manifest replay identities must be unique: "
                        f"{identity!r}"
                    )
                raise ValueError(
                    "train_manifest sample_id must be unique and non-empty: "
                    f"{sample_id!r}"
                )
            if storage_type not in {"file", "tar", "zip"}:
                raise ValueError(f"Unsupported train_manifest storage_type: {storage_type}")
            if not path.strip():
                raise ValueError("train_manifest path must be non-empty")
            if storage_type != "file" and (
                member is None or not str(member).strip()
            ):
                raise ValueError("Archive-backed manifest rows require member")
            normalized_member = None if member is None else str(member)
            locator = (storage_type, path, normalized_member)
            if replay_mode:
                previous_locator = replay_locators.setdefault(sample_id, locator)
                if previous_locator != locator:
                    raise ValueError(
                        "train_manifest replay rows must retain one locator per "
                        f"sample_id: {sample_id!r}"
                    )
                repeats_by_sample.setdefault(sample_id, set()).add(replay_repeat)
            seen.add(identity)
            record = {
                "sample_id": sample_id,
                "storage_type": storage_type,
                "path": path,
                "member": normalized_member,
            }
            if replay_mode:
                record["replay_repeat"] = replay_repeat
            records.append(record)
        for sample_id, repeats in repeats_by_sample.items():
            expected = set(range(max(repeats) + 1))
            if repeats != expected:
                raise ValueError(
                    "train_manifest replay_repeat must be contiguous from zero "
                    f"for sample_id {sample_id!r}"
                )
        return records

    def sampling_group(self, index):
        """Return the backing file/shard used to group sampler reads."""
        record = self.all_images[index]
        return f"{record['storage_type']}:{record['path']}"

    def _get_item_internal_(self, idx):
        if self.manifest_path is None:
            return super()._get_item_internal_(idx)
        record = self.all_images[idx]
        image = self._reader.image(record["storage_type"], record["path"], record["member"])
        images = self.transform(image)
        if self.train:
            return {"global_crops": images["global_crops"], "local_crops": images["local_crops"]}
        return {"images": images, "input_path": record["sample_id"]}

    def __getitem__(self, idx):
        """Return one transformed sample while logging record-level failures."""
        try:
            return self._get_item_internal_(idx)
        except Exception as error:
            logger.error("Error retrieving DINOv3 image %s: %s", self.all_images[idx], error)
            raise
