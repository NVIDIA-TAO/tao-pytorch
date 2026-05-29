# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted multi-source data sampler.

Combines multiple data streams (typically one per shard group or data source)
into a single iterable, drawing from each source according to configurable
weights.  Uses deficit-round-robin scheduling (MixtureEmitter) for source
selection and weighted reservoir sampling (WeightedWSWoRReservoir) for
buffered, without-replacement sample selection within each source.
"""

import math
import random
from typing import Iterable, List, Union

from torch.utils.data import IterableDataset

from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.utils import (
    SharedEpoch,
    pytorch_worker_seed,
    iterator_exhauster,
    seed_from_tuple,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.misc.weighted_streaming_sampler import (
    WeightedWSWoRReservoir,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.misc.mixture_emitter import (
    MixtureEmitter,
)


class MultiPipeSampler(IterableDataset):
    """Iterable dataset that merges multiple weighted data pipes.

    Each pipe feeds a reservoir buffer whose size is proportional to its
    rate.  On each step, MixtureEmitter picks the next source (deficit
    round-robin), the corresponding reservoir yields a sample while
    ingesting a fresh one from the stream.

    Args:
        pipes: List of iterables (one per data source).
        rates: Unnormalised weights controlling draw frequency.
        epoch: Current epoch (int or SharedEpoch).
        seed: Base seed for deterministic behaviour.
        deterministic: If True, seed RNG from worker/epoch.
        bufsize: Total reservoir capacity (split proportionally).
    """

    def __init__(
        self,
        pipes: List[Iterable],
        rates: List[float],
        epoch: Union[int, SharedEpoch],
        seed: int,
        deterministic: bool = True,
        bufsize: int = 1000,
    ):
        super().__init__()

        total_rate = sum(rates)
        self.pipes = pipes
        self.rates = [r / total_rate for r in rates]
        self.epoch = epoch
        self.deterministic = deterministic
        self.seed = seed
        self.bufsize = bufsize

        self.rng = random.Random(seed)

        self.reservoirs: List[WeightedWSWoRReservoir] = []
        self.rv_sizes: List[int] = []
        for r in self.rates:
            r_bufsize = max(int(math.ceil(r * bufsize)), 200)
            rv = WeightedWSWoRReservoir(r_bufsize, self.rng)
            self.reservoirs.append(rv)
            self.rv_sizes.append(r_bufsize)

        self.ds_selector = MixtureEmitter(self.rates)

    def __iter__(self):
        """Iterate by drawing samples from multiple pipelines via weighted reservoirs."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            seed = pytorch_worker_seed(seed_from_tuple(self.seed, epoch, 137))
            self.rng.seed(seed)

        streams = [
            iterator_exhauster(self.get_pipe_iterator(p))
            for p in self.pipes
        ]

        for rv_size, rv, stream in zip(self.rv_sizes, self.reservoirs, streams):
            num_to_prefill = rv_size - rv.live_count
            pf_weights, pf_samples = [], []
            for _ in range(num_to_prefill):
                try:
                    sample = next(stream)
                except StopIteration:
                    return
                weight = sample.get('__weight__', 1.0)
                pf_weights.append(weight)
                pf_samples.append(sample)
            rv.fast_build(pf_weights, pf_samples)

        while True:
            stream_idx = self.ds_selector.pick()
            stream = streams[stream_idx]

            def gen():
                sample = next(stream)
                weight = sample.get('__weight__', 1.0)
                return weight, sample
            sample_to_yield = self.reservoirs[stream_idx].get_next_and_refresh(gen)

            yield sample_to_yield

    def get_pipe_iterator(self, pipe):
        """Return a zero-arg callable that yields samples from ``pipe``."""
        def fn():
            iterator = iter(pipe)
            yield from iterator
        return fn
