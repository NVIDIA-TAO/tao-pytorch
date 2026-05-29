# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and utilities for the spatial transform pipeline.

Classes:
    Size            -- (height, width) named tuple that doubles as a callable.
    RandomSize      -- Callable that samples a random (height, width) per call.
    ScopedRNG       -- Worker-aware numpy RNG wrapper for deterministic
                       stochastic resolution across ranks.
    Identity        -- No-op transform.
    RandomTransformBase -- Mixin that advances numpy RNG state per worker.
    CompositeTransform  -- Applies a list of transforms in order.

Functions:
    linear_sample     -- Uniform sample in [a, b].
    log_linear_sample -- Uniform sample in log-space.
    clamped_normal     -- Normal sample clamped to [min_val, max_val].
"""

import logging
import math
from typing import Callable, Dict, List, Optional, Union

import numpy
import torch

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Size helpers
# -------------------------------------------------------------------------


class Size:
    """Fixed (height, width) pair that is also callable (returns itself)."""

    __slots__ = ("height", "width")

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self):
        """Return ``self`` so ``Size`` instances satisfy the ``RandomSize`` interface."""
        return self

    def __eq__(self, other):
        """Return True if ``other`` is a ``Size`` with the same dimensions."""
        if isinstance(other, Size):
            return self.height == other.height and self.width == other.width
        return NotImplemented

    def __hash__(self):
        """Hash over ``(height, width)`` so ``Size`` is usable as a dict key."""
        return hash((self.height, self.width))

    def __iter__(self):
        """Yield ``(height, width)`` for tuple-like unpacking."""
        return iter((self.height, self.width))

    def __repr__(self):
        """Return a human-readable repr of this size."""
        return f"Size(height={self.height}, width={self.width})"


class ScopedRNG:
    """Worker-aware RNG for deterministic stochastic resolution sampling.

    On first property access, the seed is derived from a base RNG plus the
    DataLoader worker id.  Every ``__getattr__`` call creates a fresh
    ``numpy.random.Generator`` from that seed — this guarantees that all
    workers on all ranks sample the same sequence *per batch*.

    Call ``reset_seed()`` between batches to resample.
    """

    def __init__(self, seed: int):
        self._rng_seed = None
        self._base_rng = numpy.random.default_rng(seed)
        self.reset_seed()

    def __getattr__(self, name: str):
        """Return a bound numpy Generator method, instantiated fresh from the seed."""
        rng = numpy.random.default_rng(self.seed)
        return lambda *args, **kwargs: getattr(rng, name)(*args, **kwargs)

    @property
    def seed(self):
        """Return the worker-aware lazily-derived RNG seed."""
        if self._rng_seed is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
            else:
                worker_id = 0

            base_rand = self._base_rng.bit_generator.random_raw()

            self._rng_seed = (base_rand + worker_id) % (2 ** 32 - 1)
        return self._rng_seed

    def reset_seed(self):
        """Clear the cached seed so the next access re-samples one."""
        self._rng_seed = None


class RandomSize:
    """Callable that returns a random ``Size`` per invocation.

    Supports both explicit resolution dictionaries (multinomial sampling)
    and uniform integer ranges quantised by ``step_size``.
    """

    def __init__(self, rng: ScopedRNG, base_size: int, min_size: int, max_size: int,
                 step_size: int, fixed_aspect: bool = True,
                 resolutions: Optional[Dict[int, float]] = None,
                 transform: Optional[Callable[[int], int]] = None):
        self.rng = rng
        self.base_size = base_size // step_size
        self.min_size = min_size // step_size if min_size is not None else self.base_size
        self.max_size = (max_size // step_size) + 1 if max_size is not None else self.base_size + 1
        self.step_size = step_size
        self.fixed_aspect = fixed_aspect
        self.transform = transform

        self.explicit = False
        if resolutions is not None:
            self.res_probs = numpy.array(list(resolutions.values()))
            self.res_values = numpy.array(list(resolutions.keys()))
            self.explicit = True

    def __call__(self):
        """Sample a ``Size`` according to the configured distribution."""
        width = self._generate()
        if self.fixed_aspect:
            height = width
        else:
            height = self._generate()

        return Size(height=height, width=width)

    def _generate(self):
        if self.explicit:
            v = self.rng.choice(a=self.res_values, p=self.res_probs)
        else:
            v = self.rng.integers(self.min_size, self.max_size)
            v = v * self.step_size

        if self.transform is not None:
            v = self.transform(v)
        return v


# -------------------------------------------------------------------------
# No-op transform
# -------------------------------------------------------------------------


class Identity:
    """No-op transform — does nothing when called."""

    def __call__(self, *items, **kwargs):
        """No-op: leave items unchanged."""
        pass


# -------------------------------------------------------------------------
# Random transform base class
# -------------------------------------------------------------------------


class RandomTransformBase:
    """Mixin that advances numpy RNG state per DataLoader worker.

    Ensures different workers produce different random sequences by
    consuming worker_id * 31 raw bits from the generator on first use.
    """

    def _prepare_rng(self, rngs: Union[numpy.random.Generator, List[numpy.random.Generator]]):
        is_prepared = getattr(self, '_is_rng_worker_prepared', False)

        if is_prepared:
            return

        if not isinstance(rngs, (list, tuple)):
            rngs = [rngs]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            for rng in rngs:
                if isinstance(rng, ScopedRNG):
                    crng = rng._base_rng
                elif isinstance(rng, numpy.random.Generator):
                    crng = rng
                else:
                    continue
                for _ in range(worker_id * 31):
                    crng.bit_generator.random_raw()

        setattr(self, '_is_rng_worker_prepared', True)


# -------------------------------------------------------------------------
# Composite transform
# -------------------------------------------------------------------------


class CompositeTransform:
    """Applies a sequence of sub-transforms in order.

    Before the sub-transforms run, any item with a ``coalesce_homogeneous``
    method is coalesced, and after they run ``mark_dirty`` is called.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, items, **kwargs):
        """Apply each sub-transform to ``items`` in order and mark them dirty."""
        mark_dirties = []
        for item in items:
            coal_fn = getattr(item, 'coalesce_homogeneous', None)
            if coal_fn is not None:
                coal_fn()
                mark_dirties.append(item)

        for tx in self.transforms:
            tx(*items, **kwargs)

        for item in mark_dirties:
            item.mark_dirty()
        if len(items) > 1:
            return items
        return items[0]

    def set_dataset_name(self, name):
        """Propagate ``name`` to any sub-transform that supports it."""
        for tx in self.transforms:
            st = getattr(tx, 'set_dataset_name', None)
            if st is not None:
                st(name)


# -------------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------------


def linear_sample(a, b, rng: numpy.random.Generator = None):
    """Uniform sample in [a, b]."""
    rval = rng.random() if rng is not None else numpy.random.random()
    return (b - a) * rval + a


def log_linear_sample(a, b, rng: numpy.random.Generator = None):
    """Uniform sample in log-space between a and b."""
    la = math.log(a)
    lb = math.log(b)
    s = linear_sample(la, lb, rng=rng)
    return math.exp(s)


def clamped_normal(std, min_val, max_val, rng: numpy.random.Generator = None, **kwargs):
    """Normal sample clamped to [min_val, max_val] via rejection."""
    def _sample():
        return rng.normal(**kwargs) if rng is not None else numpy.random.randn(**kwargs)

    if min_val == max_val:
        return std * _sample()

    s = std * _sample()
    invalid = numpy.logical_or(min_val > s, max_val < s)
    while numpy.any(invalid):
        s2 = std * _sample()
        s = numpy.where(invalid, s2, s)
        invalid = numpy.logical_or(min_val > s, max_val < s)
    return s
