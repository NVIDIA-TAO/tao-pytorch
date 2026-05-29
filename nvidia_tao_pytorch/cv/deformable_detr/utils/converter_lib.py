# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for converting datasets to json"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math


def _shard(partitions, num_shards):
    """Shard each partition."""
    num_shards = max(num_shards, 1)  # 0 means 1 shard.
    shards = []
    for partition in partitions:
        result = []
        if len(partition) == 0:
            continue
        shard_size = math.ceil(len(partition) / num_shards)

        for i in range(num_shards):
            begin = i * shard_size
            end = (i + 1) * shard_size
            if end > len(partition):
                pad_counter = end - len(partition)
                pad_samples = random.sample(partition, pad_counter)
                out_partition = partition[begin: len(partition)] + pad_samples
            else:
                out_partition = partition[begin:end]
            result.append(out_partition)
        shards.append(result)
    return shards


def _shuffle(partitions):
    """Shuffle each partition independently."""
    for partition in partitions:
        random.shuffle(partition)
