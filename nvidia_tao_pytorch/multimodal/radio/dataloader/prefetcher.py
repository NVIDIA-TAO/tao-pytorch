# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetcher utilities for overlapping host-to-device copies with model compute."""
from collections.abc import Mapping
from typing import Iterator, Tuple, Union, List, Dict

import torch
from torch.utils.data import IterableDataset


DATA_T = Union[int, float, torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]


class Prefetcher:
    """Follows same interface as Fetcher, but implements overlapped
    prefetching in a separate stream.
    """

    def __init__(self, loader: Iterator[Tuple[DATA_T, DATA_T]]):
        self.next_data = None

        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
        self.done = False

    def __len__(self):
        """Enables len(fetcher)"""
        return len(self.loader)

    def preload(self):
        """Fetch the next input/target pair on a separate stream.
        Trigger H2D copy
        """
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.done = True
            return
        with torch.cuda.stream(self.stream):
            self.next_data = _to_cuda(self.next_data)

    def next(self):
        """Return prefetched image/target and trigger next h2d.
        We first wait for the h2d op to be complete, which is precautionary
        since the compute stream is expected to be busy much longer.
        The call to record_stream marks the tensor as being used by a different
        stream than the one in which it was allocated (which is the prefetch stream),
        and prevents premature return to the mem pool.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.record_stream
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            _record_stream(data, torch.cuda.current_stream())
        self.preload()
        return data


class IterablePrefetcher(IterableDataset):
    """Iterable dataset that wraps another iterable and applies a Prefetcher."""

    def __init__(self, inner: IterableDataset):  # , batch_size: int = None, dataset_length: int = None):
        """Initialize the iterable prefetcher with an inner IterableDataset."""
        super().__init__()
        self.inner = inner
        # # Store batch_size for compatibility with TAO Lightning module
        # self.batch_size = batch_size if batch_size is not None else getattr(inner, 'batch_size', None)

        # # Add other attributes that TAO Lightning module might expect
        # self.batch_sampler = getattr(inner, 'batch_sampler', None)

        # # Create a dataset wrapper that provides length information
        # self.dataset = DatasetWrapper(inner, dataset_length)

    def __iter__(self):
        """Iterate over the inner dataset with prefetching enabled."""
        prefetcher = Prefetcher(iter(self.inner))

        while not prefetcher.done:
            yield prefetcher.next()

    # def __len__(self):
    #     """Return length of inner dataset if available."""
    #     if hasattr(self.inner, '__len__'):
    #         return len(self.inner)
    #     elif hasattr(self.dataset, '__len__'):
    #         return len(self.dataset)
    #     else:
    #         # For IterableDatasets that don't have a length, we can't provide one
    #         raise TypeError(f"object of type '{type(self).__name__}' has no len()")


class DatasetWrapper:
    """Wrapper class to provide dataset interface for IterablePrefetcher."""

    def __init__(self, inner_dataset, length: int = None):
        self.inner_dataset = inner_dataset
        self._length = length

    def __len__(self):
        """Return the length of the dataset."""
        if self._length is not None:
            return self._length
        elif hasattr(self.inner_dataset, '__len__'):
            return len(self.inner_dataset)
        elif hasattr(self.inner_dataset, 'dataset') and hasattr(self.inner_dataset.dataset, '__len__'):
            return len(self.inner_dataset.dataset)
        else:
            # For WebDataset pipelines, we need to estimate or use a default
            # This is a fallback - in practice, you should provide dataset_length
            return 10000  # Default fallback length

    def __getattr__(self, name):
        """Delegate attribute access to the inner dataset."""
        return getattr(self.inner_dataset, name)


def _record_stream(data: DATA_T, stream: torch.cuda.Stream):
    if torch.is_tensor(data):
        data.record_stream(stream)
    elif isinstance(data, (list, tuple)):
        for d in data:
            _record_stream(d, stream)
    elif isinstance(data, Mapping):
        for v in data.values():
            _record_stream(v, stream)


def _to_cuda(data: DATA_T):
    if torch.is_tensor(data):
        return data.cuda(non_blocking=True)
    if isinstance(data, (list, tuple)):
        return [_to_cuda(d) for d in data]
    if isinstance(data, Mapping):
        return {k: _to_cuda(v) for k, v in data.items()}
    return data
