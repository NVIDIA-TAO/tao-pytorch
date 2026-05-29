# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-count SQLite database discovery.

Walks up the directory tree from a dataset root looking for a
``hash_counts.db`` file.  This database stores per-sample MD5 occurrence
counts used by ``SampleWeightInjector`` for frequency-based rebalancing.
"""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

_CONN_MAP = dict()


def get_hash_database(dataset_root: str):
    """Find and open the nearest ``hash_counts.db`` above *dataset_root*.

    Searches up to 3 directory levels.  Returns a read-only
    ``sqlite3.Connection`` or ``None`` if no database is found.
    Connections are cached per path so repeated calls for the same
    dataset are free.
    """
    for _ in range(3):
        db_path = os.path.join(dataset_root, "hash_counts.db")
        if os.path.exists(db_path):
            break
        dataset_root = os.path.dirname(dataset_root)
    else:
        return None

    if db_path in _CONN_MAP:
        return _CONN_MAP[db_path]

    db = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, uri=True)
    db.execute('PRAGMA query_only = 1')
    db.execute('PRAGMA read_uncommitted = true')

    _CONN_MAP[db_path] = db
    return db


def get_hash_count_filter(db: sqlite3.Connection, max_count: int = 100):
    """Build a set of MD5 hashes that appear more than *max_count* times.

    The returned set can be used as a blocklist to drop over-represented
    samples from the pipeline.

    Args:
        db: Open SQLite connection (from ``get_hash_database``).
        max_count: Count threshold above which hashes are included.

    Returns:
        Set of hex-encoded MD5 strings.
    """
    query = 'SELECT md5, count FROM md5_values WHERE count > ? ORDER BY count DESC'
    results = db.execute(query, (max_count,)).fetchall()

    filter_set = {r[0].hex() for r in results}
    counts = list(r[1] for r in results)

    total = sum(counts)
    logger.info(
        'Filtering %d hashes. Num Images: %d. Top 10 Counts: %s',
        len(filter_set), total, counts[:10],
    )

    return filter_set
