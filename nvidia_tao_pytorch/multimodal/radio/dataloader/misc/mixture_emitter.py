# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic deficit-round-robin (WFQ) scheduler.

Pure Python, no external dependencies. Used by MultiPipeSampler to
decide which data source to draw from on each iteration.
"""

from typing import List


class MixtureEmitter:
    """Deterministic deficit-round-robin (WFQ) scheduler.

    Assumes all datasets are always non-empty.
    Produces picks with near-exact proportions *ks* in every short window.
    """

    def __init__(self, ks: List[float]):
        s = float(sum(ks))
        assert s > 0 and all(k >= 0 for k in ks)
        # normalize
        self.ks = [k / s for k in ks]
        self.n = len(self.ks)
        self.deficit = [0.0] * self.n
        # rotates to break ties deterministically
        self.cursor = 0
        self.pick_ct = [0] * self.n

    def pick(self):
        """Return the index of the component with the largest pending deficit."""
        # 1) accumulate target share
        for i in range(self.n):
            self.deficit[i] += self.ks[i]

        # 2) choose argmax deficit, breaking ties by rotating cursor
        eps = 1e-12
        best = self.cursor
        best_val = self.deficit[best]
        for step in range(1, self.n):
            i = (self.cursor + step) % self.n
            v = self.deficit[i]
            if v > best_val + eps:
                best, best_val = i, v

        # 3) consume one unit of service and advance tie-break cursor
        self.deficit[best] -= 1.0
        self.cursor = (best + 1) % self.n
        self.pick_ct[best] += 1
        return best
