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

"""Balanced samplers for CLIP custom dataloaders."""

import random
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


# Default order for quota indexing in BalancedUniqueCaptionBatchSampler
# when using full NVIDIA PAS preparation output.
_DEFAULT_QUERY_TYPES_ORDER = (
    'easy',
    'medium',
    'hard',
    'natural_caption',
    'original_captions',
)


class BalancedByQueryTypeSampler(Sampler[int]):
    """Samples indices so that each query type is seen with equal probability per epoch.

    The sampler uses inverse-frequency weights, where sample ``i`` receives
    ``1 / count(query_type[i])``. When used with DDP, indices are first
    partitioned across ranks, then each rank samples from its own partition.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        query_type_per_index (List[str]): Query type metadata aligned to dataset indices.
        num_replicas (int): Number of distributed ranks participating in training.
        rank (int): Current distributed rank.
        seed (int): Base seed used for deterministic epoch-level sampling.
        replacement (bool): Whether to sample with replacement.
        epoch (int): Epoch value set by the Lightning data-fetching loop.
    """

    def __init__(
        self,
        num_samples: int,
        query_type_per_index: List[str],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
    ):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank ({rank}) must be in [0, {num_replicas})."
            )
        if len(query_type_per_index) != num_samples:
            raise ValueError(
                "query_type_per_index length "
                f"({len(query_type_per_index)}) must match num_samples "
                f"({num_samples})."
            )
        self.num_samples = num_samples
        self.query_type_per_index = query_type_per_index
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.replacement = replacement
        self.epoch = 0

        # Count per type and assign weight = 1 / count
        counts = Counter(query_type_per_index)
        self._weights = torch.tensor(
            [1.0 / max(counts[t], 1) for t in query_type_per_index],
            dtype=torch.double,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic per-epoch reshuffling."""
        self.epoch = epoch

    def _rank_indices(self) -> List[int]:
        """Partition indices across ranks (deterministic shuffle then split)."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.num_samples, generator=g).tolist()
        per_rank = (self.num_samples + self.num_replicas - 1) // self.num_replicas
        start = self.rank * per_rank
        end = min(start + per_rank, self.num_samples)
        return indices[start:end]

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + 7919)
        if self.num_replicas > 1:
            rank_indices = self._rank_indices()
            if not rank_indices:
                return iter([])
            weights = self._weights[rank_indices]
            # num_samples so that one epoch on this rank has len(rank_indices) samples
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(rank_indices),
                replacement=self.replacement,
                generator=generator,
            )
            return iter(rank_indices[i] for i in sampler)
        else:
            sampler = WeightedRandomSampler(
                weights=self._weights,
                num_samples=self.num_samples,
                replacement=self.replacement,
                generator=generator,
            )
            return iter(sampler)

    def __len__(self) -> int:
        if self.num_replicas > 1:
            rank_indices = self._rank_indices()
            return len(rank_indices)
        return self.num_samples


def query_types_order_from_pairs(pairs: List[dict]) -> Tuple[str, ...]:
    """Stable order: preferred NVIDIA_PAS order, then any other query_type values found."""
    return query_types_order_from_types(
        [p.get('query_type', 'easy') for p in pairs]
    )


def query_types_order_from_types(query_types: List[str]) -> Tuple[str, ...]:
    """Stable order: preferred NVIDIA_PAS order, then first-seen extra query types."""
    seen = set()
    first_seen: List[str] = []
    for t in query_types:
        if t not in seen:
            seen.add(t)
            first_seen.append(t)
    ordered = [t for t in _DEFAULT_QUERY_TYPES_ORDER if t in seen]
    for t in first_seen:
        if t not in ordered:
            ordered.append(t)
    return tuple(ordered)


class BalancedUniqueCaptionBatchSampler(Sampler[List[int]]):
    """Yields batches with (1) fixed quotas per query type and (2) unique captions per batch.

    Each batch splits ``batch_size`` across ``query_types_order`` (equal base quota per type,
    remainder to the first types in order). No two rows share the same caption string, so
    CLIP/SigLIP losses do not treat other valid image-text pairs in the batch as negatives.

    DDP: all ranks use the same ``seed + epoch`` for greedy batch construction, trim to a
    multiple of ``world_size``, then each rank yields every ``world_size``-th batch.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        query_type_per_index (List[str]): Query type metadata aligned to dataset indices.
        caption_per_index (List[str]): Caption text aligned to dataset indices.
        batch_size (int): Number of samples yielded per batch.
        query_types_order (Tuple[str, ...]): Stable query-type order used for batch quotas.
        num_replicas (int): Number of distributed ranks participating in training.
        rank (int): Current distributed rank.
        seed (int): Base seed used for deterministic epoch-level batch construction.
        epoch (int): Epoch value set by the Lightning data-fetching loop.
    """

    def __init__(
        self,
        num_samples: int,
        query_type_per_index: List[str],
        caption_per_index: List[str],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        query_types_order: Optional[Sequence[str]] = None,
    ):
        q_order = tuple(query_types_order) if query_types_order else _DEFAULT_QUERY_TYPES_ORDER
        if not q_order:
            raise ValueError("query_types_order must be non-empty.")
        if batch_size < len(q_order):
            raise ValueError(
                f"batch_size ({batch_size}) must be >= {len(q_order)} "
                "to balance all query types present."
            )
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank ({rank}) must be in [0, {num_replicas})."
            )
        if len(query_type_per_index) != num_samples or len(caption_per_index) != num_samples:
            raise ValueError("query_type and caption lists must match num_samples.")
        unknown = set(query_type_per_index) - set(q_order)
        if unknown:
            raise ValueError(
                f"train_pairs.json contains query_type values not in query_types_order {q_order}: "
                f"{sorted(unknown)}"
            )
        self.num_samples = num_samples
        self.query_type_per_index = query_type_per_index
        self.caption_per_index = caption_per_index
        self.batch_size = batch_size
        self.query_types_order = q_order
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        # __len__ uses epoch-0 batch count; later epochs may differ slightly (seed+epoch).
        rng0 = random.Random(seed)
        batches0 = self._generate_all_batches(rng0)
        usable0 = (len(batches0) // self.num_replicas) * self.num_replicas
        batches0 = batches0[:usable0]
        # First __iter__ uses the same RNG as rng0 (seed + epoch with epoch==0). Rebuilding
        # here would duplicate multi-hour CPU work for multi-million-sample datasets.
        self._epoch0_batches_trimmed = batches0
        self._len_per_rank = (
            len(batches0) // self.num_replicas
            if self.num_replicas else 0
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic per-epoch batch construction."""
        self.epoch = epoch

    def _quotas(self) -> List[int]:
        n = len(self.query_types_order)
        base = self.batch_size // n
        r = self.batch_size % n
        return [base + (1 if i < r else 0) for i in range(n)]

    def _initial_pools(self) -> Dict[Tuple[str, str], List[int]]:
        pools: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx in range(self.num_samples):
            qt = self.query_type_per_index[idx]
            cap = self.caption_per_index[idx]
            pools[(qt, cap)].append(idx)
        return pools

    def _try_form_batch(
        self,
        pools: Dict[Tuple[str, str], List[int]],
        quotas: List[int],
        rng: random.Random,
    ) -> Optional[List[int]]:
        used_captions = set()
        batch: List[int] = []
        type_order = list(range(len(self.query_types_order)))
        rng.shuffle(type_order)
        for ti in type_order:
            t = self.query_types_order[ti]
            k = quotas[ti]
            candidates = []
            for (qt, cap), indices in list(pools.items()):
                if qt != t or not indices or cap in used_captions:
                    continue
                candidates.append((qt, cap, indices))
            rng.shuffle(candidates)
            if len(candidates) < k:
                return None
            for qt, cap, indices in candidates[:k]:
                j = rng.randrange(len(indices))
                idx = indices[j]
                if len(indices) > 1:
                    indices[j] = indices[-1]
                    indices.pop()
                else:
                    del pools[(qt, cap)]
                batch.append(idx)
                used_captions.add(cap)
        if len(batch) != self.batch_size:
            return None
        return batch

    def _generate_all_batches(self, rng: random.Random) -> List[List[int]]:
        pools = self._initial_pools()
        quotas = self._quotas()
        batches: List[List[int]] = []
        while True:
            batch = self._try_form_batch(pools, quotas, rng)
            if batch is None:
                break
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # Different random greedy order each epoch so coverage varies; __len__
        # is from epoch 0.
        rng = random.Random(self.seed + self.epoch)
        if self.epoch == 0 and self._epoch0_batches_trimmed is not None:
            # Same plan as __init__ (rng0 == Random(seed+0)); avoid a second full rebuild.
            batches = [list(b) for b in self._epoch0_batches_trimmed]
            self._epoch0_batches_trimmed = None
        else:
            batches = self._generate_all_batches(rng)
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:usable]
        rng_order = random.Random(self.seed + self.epoch + 7919)
        rng_order.shuffle(batches)
        my_batches = batches[self.rank:: self.num_replicas]
        for b in my_batches:
            yield b

    def __len__(self) -> int:
        return max(0, self._len_per_rank)
