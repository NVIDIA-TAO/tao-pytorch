# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-sample data loading for DINO-style SSL training.

``DataLoader(batch_sampler=...)`` hands a *whole batch* of indices to a single worker. DINO
augmentation is expensive on CPU (tens of milliseconds per image), so at a large batch size one
worker needs seconds to produce one batch and the remaining workers cannot help: the GPU idles
between steps even when many workers are configured.

``PerSampleBatchLoader`` instead distributes work one *sample* at a time across all workers and
assembles (collates, pins) the batch in a background thread of the main process. Sample order,
``drop_last`` behaviour and sampler ownership are unchanged.
"""

import math
import queue
import threading
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.pin_memory import pin_memory as _pin_memory

_END = object()


def _identity(sample: Any) -> Any:
    """Return the sample untouched (batching happens after the workers)."""
    return sample


class PerSampleBatchLoader:
    """Batch iterable that spreads per-sample work across all DataLoader workers.

    The object mimics the subset of the ``DataLoader`` API used by Lightning and TAO
    (``dataset``, ``sampler``, ``batch_size``, ``num_workers``, ``len`` and iteration).
    ``sampler`` is the exact object passed in, so ``set_epoch`` calls made by Lightning on
    ``loader.sampler`` reach the real sampler.
    """

    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        batch_size: int,
        num_workers: int,
        collate_fn: Callable[[list], Any],
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_batches: int = 2,
    ):
        """Initialize the loader.

        Args:
            dataset: Map-style dataset returning one sample per index.
            sampler: Sampler yielding dataset indices (one rank's share of the epoch).
            batch_size: Samples per batch. A trailing partial batch is dropped.
            num_workers: Number of worker processes (must be > 0).
            collate_fn: Callable turning a list of ``batch_size`` samples into a batch.
            pin_memory: Pin the collated batch (only when CUDA is available).
            persistent_workers: Keep worker processes alive across epochs.
            prefetch_batches: Number of assembled batches the background thread keeps ready.
        """
        if num_workers < 1:
            raise ValueError("PerSampleBatchLoader requires num_workers >= 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.prefetch_batches = max(1, prefetch_batches)
        # Keep roughly `prefetch_batches` batches worth of samples in flight across the workers.
        samples_in_flight = self.prefetch_batches * batch_size
        self._loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_identity,
            pin_memory=False,
            persistent_workers=persistent_workers,
            prefetch_factor=max(2, math.ceil(samples_in_flight / num_workers)),
        )

    def __len__(self) -> int:
        """Number of full batches per epoch (``drop_last=True``)."""
        return len(self.sampler) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        """Iterate over collated batches, assembled in a background thread."""
        out: "queue.Queue" = queue.Queue(maxsize=self.prefetch_batches)
        stop = threading.Event()

        def _put(item) -> bool:
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def _produce():
            try:
                samples = []
                for sample in self._loader:
                    samples.append(sample)
                    if len(samples) < self.batch_size:
                        continue
                    batch = self.collate_fn(samples)
                    samples = []
                    if self.pin_memory:
                        batch = _pin_memory(batch)
                    if not _put(batch):
                        return
                _put(_END)
            except BaseException as err:  # pylint: disable=broad-except
                _put(err)

        thread = threading.Thread(target=_produce, name="per-sample-batcher", daemon=True)
        thread.start()
        try:
            while True:
                item = out.get()
                if item is _END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            thread.join()
