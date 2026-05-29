# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted streaming without-replacement reservoir sampler.

Uses exponential keys (key = Exp(1)/weight) and a dual min/max heap to
maintain a reservoir of the R smallest keys seen so far.

Only runtime dependency beyond stdlib is torch (for get_worker_info logging).
"""

import heapq
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Node:
    """Reservoir entry storing a payload with its exponential key and weight."""

    key: float
    payload: Any
    weight: float
    alive: bool = True


class WeightedWSWoRReservoir:
    """Weighted streaming sampler (without replacement) using exponential keys.

    key = Exp(1) / weight
    Maintains the R smallest keys seen so far. Emitting pops the global
    minimum key.
    """

    def __init__(self, R: int, rng: Optional[random.Random] = None):
        assert R > 0
        self.R = R
        self.rng = rng or random.Random()
        self.hmin: List[Tuple[float, int, Node]] = []
        self.hmax: List[Tuple[float, int, Node]] = []
        self.live_count = 0
        self._ctr = 0
        self._rebuild_factor = 4.0

        self._next_ct = 0
        self._next_ct_full = 0
        self._num_samples = 0
        self._total_weight = 0
        self._total_weight_gen = 0
        self._min_weight_sampled = math.inf
        self._min_weight_gen = math.inf

    def _exp1(self) -> float:
        return self.rng.expovariate(1.0)

    def _prune_min(self):
        while self.hmin and not self.hmin[0][2].alive:
            heapq.heappop(self.hmin)

    def _prune_max(self):
        while self.hmax and not self.hmax[0][2].alive:
            heapq.heappop(self.hmax)

    def mink(self) -> float:
        """Min key among ALIVE items; +inf only if empty."""
        self._prune_min()
        return math.inf if self.live_count == 0 else self.hmin[0][0]

    def frontier(self) -> float:
        """Max key among ALIVE items; +inf only if empty."""
        self._prune_max()
        return math.inf if self.live_count == 0 else -self.hmax[0][0]

    def _evict_max(self):
        self._prune_max()
        if not self.hmax:
            return
        _, _, node = heapq.heappop(self.hmax)
        if node.alive:
            node.alive = False
            self.live_count -= 1

    def _push(self, node: Node):
        self._ctr += 1
        heapq.heappush(self.hmin, (node.key, self._ctr, node))
        heapq.heappush(self.hmax, (-node.key, self._ctr, node))
        self.live_count += 1
        total = len(self.hmin) + len(self.hmax)
        if total > self._rebuild_factor * max(1, 2 * self.live_count):
            self._rebuild()

    def _rebuild(self):
        new_min, new_max = [], []
        self._ctr = 0
        for _, _, node in self.hmin:
            if node.alive:
                self._ctr += 1
                heapq.heappush(new_min, (node.key, self._ctr, node))
                heapq.heappush(new_max, (-node.key, self._ctr, node))
        self.hmin, self.hmax = new_min, new_max

    def next_sample(self) -> Tuple[Optional[Any], Optional[float]]:
        """Emit the reservoir entry with the smallest key, or ``(None, None)`` if empty."""
        self._prune_min()
        while self.hmin:
            _, _, node = heapq.heappop(self.hmin)
            if not node.alive:
                continue
            node.alive = False
            self.live_count -= 1
            return node.payload, node.weight
        return None, None

    def get_next_and_refresh(self, generator: Callable[[], Tuple[float, Any]]) -> Optional[Any]:
        """Emit a sample, then pull from ``generator`` until the reservoir is refreshed."""
        if self.live_count == 0:
            raise ValueError("Reservoir is empty")

        out, out_weight = self.next_sample()
        num_samples = 0
        while True:
            num_samples += 1
            weight, payload = generator()
            k = self._exp1() / weight
            tau = self.frontier()

            self._total_weight_gen += weight
            self._min_weight_gen = min(self._min_weight_gen, weight)

            if k < tau or self.live_count < self.R:
                self._push(Node(key=k, payload=payload, weight=weight, alive=True))
                if self.live_count > self.R:
                    self._evict_max()
                    break
            if self.live_count > 0:
                mink = self.mink()
                tau = self.frontier()
                ratio = tau / (100 * mink)
                sampled = self._exp1()
                if sampled < ratio:
                    continue
            else:
                continue
            break

        self._next_ct += 1
        self._next_ct_full += 1
        self._num_samples += num_samples
        self._total_weight += out_weight
        self._min_weight_sampled = min(self._min_weight_sampled, out_weight)

        w_info = torch.utils.data.get_worker_info()
        w_id = w_info.id if w_info is not None else 0

        if self._next_ct % 10000 == 0 and w_id == 0:
            avg_samples_per = self._num_samples / self._next_ct
            avg_weight = self._total_weight / self._next_ct
            avg_weight_dist = self._total_weight_gen / self._num_samples
            min_exp = 1 / self._next_ct_full
            logger.info(
                "Avg samples per get_next_and_refresh: %.3f. "
                "Weight Sampled: %.3f, Weight Data: %.3f, "
                "Min-W Sampled: %.5f, Min-W Data: %.5f, "
                "Min-W Expected: %.5f, Min-k: %.3f, Frontier: %.3f",
                avg_samples_per, avg_weight, avg_weight_dist,
                self._min_weight_sampled, self._min_weight_gen,
                min_exp, self.mink(), self.frontier(),
            )
            self._next_ct = 0
            self._num_samples = 0
            self._total_weight = 0
            self._total_weight_gen = 0

        return out

    def fast_build(self, weights: List[float], payloads: List[Any]):
        """Bulk-populate the reservoir from parallel ``weights``/``payloads`` lists."""
        if len(weights) != len(payloads):
            raise ValueError("weights and payloads must have the same length")
        for w, p in zip(weights, payloads):
            k = self._exp1() / w
            node = Node(key=k, payload=p, weight=w, alive=True)
            ct = self._ctr
            self.hmin.append((k, ct, node))
            self.hmax.append((-k, ct, node))
            self._ctr += 1
        heapq.heapify(self.hmin)
        heapq.heapify(self.hmax)
        self.live_count += len(weights)
