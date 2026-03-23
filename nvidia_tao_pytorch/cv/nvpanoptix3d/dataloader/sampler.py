# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Samplers to provide batch-level deterministic metadata.

For DDP with num_workers>0, per-sample randomness computed inside Dataset workers
cannot guarantee batch-consistent behavior, because a single batch is assembled
from multiple workers.

This sampler runs in the main process and attaches a per-batch seed to each index.
All indices that end up in the same batch (within a rank) will carry the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterator, Optional, TypeVar

import torch
from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class IndexWithSeed:
    """Index wrapper passed into ``Dataset.__getitem__``.

    Attributes:
        idx: Dataset index.
        seed: Per-batch seed attached by :class:`BatchSeededSampler`.
    """

    idx: int
    seed: int


class BatchSeededSampler(Sampler[IndexWithSeed]):
    """Wrap an existing sampler and attach a per-batch seed to each yielded index.

    The DataLoader will batch these yielded items using its own BatchSampler
    (when `batch_size` is provided). Since this sampler yields items sequentially,
    every consecutive `batch_size` items will share one seed.
    """

    def __init__(self, base_sampler: Sampler[int], *, batch_size: int, base_seed: Optional[int] = None):
        """Create a sampler that attaches a deterministic per-batch seed.

        Args:
            base_sampler: Sampler that yields integer dataset indices (e.g. a
                :class:`~torch.utils.data.DistributedSampler` or a random sampler).
            batch_size: Batch size used by the DataLoader. Every consecutive
                ``batch_size`` indices yielded by this wrapper will share the same seed.
            base_seed: Optional base seed used to derive the per-epoch RNG stream.
                If None, defaults to ``torch.initial_seed()`` (which is stable per-rank
                when global seeding is configured).

        Raises:
            ValueError: If ``batch_size <= 0``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.base_sampler = base_sampler
        self.batch_size = int(batch_size)
        # Use torch.initial_seed() as a stable default (per-rank) when a global seed is set.
        self.base_seed = int(torch.initial_seed()) if base_seed is None else int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic seeding and forward to wrapped sampler if supported.

        Args:
            epoch: Current epoch number.
        """
        # Forward epoch to the wrapped sampler if supported (DistributedSampler pattern).
        self.epoch = int(epoch)
        if hasattr(self.base_sampler, "set_epoch"):
            try:
                getattr(self.base_sampler, "set_epoch")(epoch)
            except Exception:
                pass

    def __iter__(self) -> Iterator[IndexWithSeed]:
        """Yield indices wrapped with a per-batch seed.

        The seeding scheme is:
        - Create an epoch-specific RNG seed from ``(base_seed, epoch)``.
        - Use that RNG to draw a new integer seed at the start of each batch.
        - Attach that seed to every item in the batch.

        Returns:
            Iterator of :class:`IndexWithSeed` values.
        """
        # Derive a deterministic per-epoch RNG from base_seed.
        seed = hash((self.base_seed, self.epoch)) % (2**31)
        rng = random.Random(int(seed))

        cur_batch_seed = rng.randrange(0, 2**31)
        i = 0
        for idx in self.base_sampler:
            if i % self.batch_size == 0:
                cur_batch_seed = rng.randrange(0, 2**31)
            yield IndexWithSeed(int(idx), int(cur_batch_seed))
            i += 1

    def __len__(self) -> int:
        """Return the number of samples produced by the wrapped sampler."""
        return len(self.base_sampler)

    # Optional state passthrough for stateful training/resume.
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing/resume.

        Returns:
            A dict containing ``epoch`` and ``base_seed`` and, if supported by the
            wrapped sampler, a nested ``base_sampler`` state dict.
        """
        out: dict[str, Any] = {"epoch": self.epoch, "base_seed": self.base_seed}
        if hasattr(self.base_sampler, "state_dict"):
            try:
                out["base_sampler"] = getattr(self.base_sampler, "state_dict")()
            except Exception:
                pass
        return out

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from :meth:`state_dict`.

        Args:
            state_dict: State dict produced by :meth:`state_dict`.
        """
        self.epoch = int(state_dict.get("epoch", 0))
        self.base_seed = int(state_dict.get("base_seed", self.base_seed))
        inner = state_dict.get("base_sampler")
        if inner is not None and hasattr(self.base_sampler, "load_state_dict"):
            try:
                getattr(self.base_sampler, "load_state_dict")(inner)
            except Exception:
                pass
