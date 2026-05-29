# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frequency-based sample weight injection via SQLite hash database.

Queries a precomputed hash-counts SQLite database to look up how often
each sample appears in the dataset, then injects a ``__weight__`` field
into the sample dict.  Downstream consumers (e.g. ``WeightedWSWoRReservoir``
inside ``MultiPipeSampler``) use these weights for balanced sampling.
"""

import logging
import math
import sqlite3
from typing import Any, Callable, List, Union

import webdataset as wds

logger = logging.getLogger(__name__)


class SampleWeightInjector(wds.PipelineStage):
    """WebDataset pipeline stage that annotates samples with frequency weights.

    For every batch of ``batch_size`` samples drawn from the upstream
    pipeline, the injector extracts a key (typically an MD5 hash) from
    each sample's metadata, queries the SQLite database for occurrence
    counts, computes a weight via ``weight_fn(count)``, and stores it
    in ``sample['__weight__']``.

    Args:
        database: Path to the SQLite database or an existing connection.
        key_extractor: Callable that maps a sample dict to its database key.
        weight_mode: How to convert counts to weights.  One of
            ``'inv_frequency'``, ``'inv_sq_frequency'``, ``'inv_log_frequency'``,
            ``'none'``/``'uniform'``.
        batch_size: Number of samples to query at once (for efficiency).
        logger: Optional logger for progress messages during DB loading.
    """

    def __init__(
        self,
        database: Union[str, sqlite3.Connection],
        key_extractor: Callable[[Any], str],
        weight_mode: str = 'frequency',
        batch_size: int = 512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.key_extractor = key_extractor

        def _uniform(_v):
            return 1

        def _inv_frequency(v):
            return 1 / v

        def _inv_sq_frequency(v):
            return 1 / (v ** 2)

        def _inv_log_frequency(v):
            return 1 / math.log(v + 1)

        if not weight_mode or weight_mode in ('none', 'uniform'):
            weight_fn = _uniform
        elif weight_mode == 'inv_frequency':
            weight_fn = _inv_frequency
        elif weight_mode == 'inv_sq_frequency':
            weight_fn = _inv_sq_frequency
        elif weight_mode == 'inv_log_frequency':
            weight_fn = _inv_log_frequency
        else:
            raise ValueError(f'Unsupported weight function: {weight_mode}')
        self.weight_fn = weight_fn

        if isinstance(database, str):
            logger.info('Loading database...')
            self.conn = sqlite3.connect(
                database, isolation_level=None,
                check_same_thread=False, uri=True,
            )
            self.conn.execute('PRAGMA query_only = 1')
            self.conn.execute('PRAGMA read_uncommitted = true')
            logger.info('Done')
        else:
            self.conn = database

    def _get_weights(self, keys: List[Union[str, bytes]]):
        query = 'SELECT md5, count FROM md5_values WHERE md5 IN ({})'.format(
            ', '.join(['?'] * len(keys)),
        )

        self.cursor.execute(query, tuple(keys))
        results = self.cursor.fetchall()

        count_map = dict.fromkeys(keys, 1)
        for key, count in results:
            count_map[key] = count

        weights = []
        for key in keys:
            weight = self.weight_fn(count_map[key])
            weights.append(weight)

        return weights

    def run(self, src):
        """Yield samples annotated with per-sample frequency weights."""
        def get_next_samples(count: int):
            dataset = []
            for _ in range(count):
                try:
                    dataset.append(next(src))
                except StopIteration:
                    break
            if not dataset:
                return None, None, None
            keys = [self.key_extractor(d) for d in dataset]
            weights = self._get_weights(keys)
            return weights, keys, dataset

        self.cursor = self.conn.cursor()

        while True:
            weights, _keys, samples = get_next_samples(self.batch_size)
            if samples is None:
                break
            for w, s in zip(weights, samples):
                s['__weight__'] = w
                yield s

        self.cursor.close()
