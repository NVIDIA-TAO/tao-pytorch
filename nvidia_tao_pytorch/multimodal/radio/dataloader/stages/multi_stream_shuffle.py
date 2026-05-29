# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shard-level shuffling with visit counters.

Distributes tar URLs across workers, tracks which shards have been visited
via a shared-memory counter, and always picks the least-visited shard next.
This gives reproducible, balanced coverage of all shards across epochs.
"""

import logging
import random
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset

import webdataset as wds

from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.utils import (
    SharedEpoch,
    expand_urls,
    pytorch_worker_seed,
    iterator_exhauster,
    ignore_and_log,
    seed_from_tuple,
)

logger = logging.getLogger(__name__)


class MultiStreamShuffle(IterableDataset):
    """Iterable dataset that streams samples from shards in visit-count order.

    Maintains ``num_streams`` parallel shard iterators.  On each step the
    next shard is chosen as the least-visited one among the URLs assigned
    to this worker, ensuring balanced coverage.  Visit counts are kept in
    shared memory so they persist across DataLoader workers and can be
    checkpointed/restored for resumption.

    Args:
        urls: List of shard URLs (or a brace-expand pattern string).
        seed: Base seed for deterministic shuffling.
        num_streams: Number of parallel shard iterators to maintain.
        epoch: Current epoch (int or SharedEpoch).
        deterministic: If True, seed RNG from worker/epoch for reproducibility.
        reduce_urls_fn: Optional callable ``(urls, rng) -> [(idx, url), ...]``
            that partitions URLs for the current worker/rank.
    """

    def __init__(
        self,
        urls: List[str],
        seed: int,
        num_streams: int = 4,
        epoch: Union[int, SharedEpoch] = -1,
        deterministic: bool = True,
        reduce_urls_fn: Optional[Callable[[List[str], random.Random], List[Tuple[int, str]]]] = None,
    ) -> None:
        super().__init__()

        self.urls = urls if isinstance(urls, (list, tuple)) else expand_urls(urls)[0]
        self.num_streams = num_streams
        self.epoch = epoch
        self.deterministic = deterministic
        self.reduce_urls_fn = reduce_urls_fn
        self.seed = seed

        self.rng = random.Random(seed)
        self.urls.sort()

        self.visit_counter = torch.zeros(len(urls), dtype=torch.int64).share_memory_()

    def __iter__(self):
        """Iterate samples by round-robin drawing from per-worker streams."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            increment = seed_from_tuple(self.seed, epoch, 877)
            seed = pytorch_worker_seed(increment)
            logging.getLogger("parity_debug").info(
                "MultiStreamShuffle seed debug: self.seed=%s epoch=%s "
                "increment=%s worker_seed=%s num_urls=%d",
                self.seed, epoch, increment, seed, len(self.urls),
            )
            self.rng.seed(seed)

        urls = self._reduce_urls()
        if len(urls) == 0:
            raise RuntimeError(
                f"Reduced to no valid urls! Original url count: {len(self.urls)}. "
                f"First: {self.urls[0]}"
            )

        streams = [
            iterator_exhauster(self._get_url_sampler(urls))
            for _ in range(self.num_streams)
        ]

        while True:
            for stream in streams:
                try:
                    yield next(stream)
                except StopIteration:
                    return

    def load_state(self, visit_counter: torch.Tensor):
        """Load per-URL visit counters from a previously-saved checkpoint."""
        if visit_counter.shape != self.visit_counter.shape:
            warnings.warn(
                "The number of urls in the state does not match "
                "the number of urls being managed!"
            )
            return False
        self.visit_counter.copy_(visit_counter)
        return True

    def get_state(self):
        """Return a snapshot of the per-URL visit counters."""
        return self.visit_counter.clone()

    def _get_url_sampler(self, urls: List[Tuple[int, str]]):
        t_url_idxs = torch.tensor([url_idx for url_idx, _ in urls], dtype=torch.int64)
        my_valid_mask = torch.zeros(len(self.urls), dtype=torch.bool)
        my_valid_mask[t_url_idxs] = True

        my_global_to_local = torch.zeros_like(self.visit_counter)
        my_global_to_local[t_url_idxs] = torch.arange(len(t_url_idxs), dtype=my_global_to_local.dtype)

        handler = ignore_and_log(logger)

        def sample_fn():
            cp_visit_counter = self.visit_counter.clone()
            min_val = torch.where(my_valid_mask, cp_visit_counter, 10000000000).amin()
            valid_mask = (cp_visit_counter == min_val) & my_valid_mask
            valid_idxs = torch.nonzero(valid_mask).flatten()
            if valid_idxs.shape[0] == 0:
                raise RuntimeError(
                    f"Reduced to no valid urls! Original url count: {len(self.urls)}. "
                    f"First: {self.urls[0]}"
                )

            g_visit_idx = self.rng.randint(0, valid_idxs.shape[0] - 1) if valid_idxs.shape[0] > 1 else 0
            g_visit_idx = valid_idxs[g_visit_idx].item()

            l_visit_idx = my_global_to_local[g_visit_idx].item()

            real_idx, real_url = urls[l_visit_idx]

            self.visit_counter[real_idx] += 1

            url = dict(url=real_url)
            yield from wds.tarfile_samples([url], handler=handler)
        return sample_fn

    def _reduce_urls(self):
        if self.reduce_urls_fn is not None:
            return self.reduce_urls_fn(self.urls, self.rng)
        return list(enumerate(self.urls))
