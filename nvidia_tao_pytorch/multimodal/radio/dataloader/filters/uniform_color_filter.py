# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter that drops images with near-uniform colour.

Uses PIL's ``Image.getextrema()`` to compute the per-channel min/max
range and rejects samples whose L2 norm of ranges falls below a
threshold.  This catches solid-colour placeholder images that add
noise to training.
"""

import logging
import math

import webdataset as wds

logger = logging.getLogger(__name__)


class UniformColorFilter(wds.PipelineStage):
    """Drop samples whose image is essentially a single colour.

    The image is expected at position ``image_tuple_idx`` in the sample
    tuple (after ``wds.to_tuple``).  ``getextrema()`` returns
    ``[(min, max), ...]`` per channel; the L2 norm of ``(max - min)``
    values is compared against ``threshold``.

    Args:
        image_tuple_idx: Position of the PIL image in the sample tuple.
        threshold: Minimum L2 norm of per-channel ranges to keep.
        verbose: Log every filtered image (instead of every 100th).
    """

    def __init__(
        self,
        image_tuple_idx: int = 0,
        threshold: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self.image_tuple_idx = image_tuple_idx
        self.threshold = threshold

        self.num_seen = 0
        self.num_filtered = 0
        self.verbose = verbose

    def run(self, src):
        """Yield samples whose image is not uniformly coloured."""
        for sample in src:
            im = sample[self.image_tuple_idx]

            extrema = im.getextrema()

            val_range = math.sqrt(
                sum((e[1] - e[0]) ** 2 for e in extrema)
            )

            self.num_seen += 1

            if val_range > self.threshold:
                yield sample
            else:
                self.num_filtered += 1

                if self.verbose or (self.num_filtered % 100) == 0:
                    pct_filtered = self.num_filtered / self.num_seen * 100
                    logger.info(
                        'Filtered uniform image. Value range: %s. '
                        'Percent Filtered: %.3f%%',
                        extrema, pct_filtered,
                    )
