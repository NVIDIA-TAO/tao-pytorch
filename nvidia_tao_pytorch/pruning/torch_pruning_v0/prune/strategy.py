# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strategy of pruning."""
import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random


def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    round_to = int(round_to)
    if round_to <= 1:
        return n_to_prune
    after_pruning = total_parameters - n_to_prune
    compensation = after_pruning % round_to
    # round to the nearest (round_to * N)
    # avoid negative n_to_prune
    if (compensation < round_to // 2 and after_pruning > round_to) or round_to > n_to_prune:
        n_to_prune = n_to_prune + compensation  # floor
    else:
        n_to_prune = n_to_prune - round_to + compensation  # ceiling
    return n_to_prune


class BaseStrategy(ABC):
    """Base Strategy class."""

    def __call__(self, *args, **kwargs):
        """Call method."""
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(cls, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0)
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError


class RandomStrategy(BaseStrategy):
    """Random Strategy class."""

    def apply(self, weights, amount=0.0, round_to=1) -> Sequence[int]:  # return index
        """Apply the strategy."""
        if amount <= 0:
            return []
        n = len(weights)
        n_to_prune = int(amount * n) if amount < 1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0:
            return []
        indices = random.sample(list(range(n)), k=n_to_prune)
        return indices


class LNStrategy(BaseStrategy):
    """LN magnitude based pruning strategy.

    Two mode of LN-magnitude-based (L1 or L2) pruning startegy are provided through this class:
    - "amount": The pruning algorithm in original Torch-pruning. "amount" means the ratio of
    number of filters to be pruned to the total number of filters. Suppose the total number of
    filters is N, then the number of filters to be pruned is N * amount. The filters are sorted
    along the LN-magnitude of each filter and the smallest N* amount filters will be pruned.
    - "thresh": The pruning algorithm in tao-keras. The filter with smaller LN-magnitude than
    a threshold will be pruned.

    Common tricks:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def __init__(self, p, mode="amount"):
        """Constructor for LNS strategy."""
        self.p = p
        self.mode = mode
        if self.mode not in ["amount", "thresh"]:
            raise ValueError("Only support \"amount\" and \"thresh\" mode")

    def apply(self, weights, amount=0.0, round_to=1, scores=None) -> Sequence[int]:  # return index
        """Apply the pruning."""
        if amount <= 0:
            return []
        n = len(weights)
        if scores is None:
            l1_norm = torch.norm(weights.view(n, -1), p=self.p, dim=1)
        else:
            l1_norm = scores

        if self.mode == "amount":
            n_to_prune = int(amount * n) if amount < 1.0 else amount
            n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
            if n_to_prune == 0:
                return []
            threshold = torch.kthvalue(l1_norm, k=n_to_prune).values
            indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        elif self.mode == "thresh":
            # Thresh is the strategy in tao-tf
            l1_norm /= torch.max(l1_norm)
            remained_idx = torch.nonzero(l1_norm > amount).view(-1).tolist()
            num_remained = len(remained_idx)
            # Granularity
            if num_remained % round_to > 0:
                num_remained += round_to - (num_remained % round_to)
            num_remained = min(num_remained, n)
            if num_remained == n:
                return []
            sorted_idx = torch.argsort(-l1_norm)
            indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist()

        return indices


class CustomScoreStrategy(BaseStrategy):
    """Custom Score Strategy.

    A helper class to execute sorting and filtering with any pruning score.

    common trick:
    - granularity. The pruned number of filters will be divisible by the granularity number.
    """

    def apply(self, scores, thresh=0.0, round_to=1) -> Sequence[int]:
        """Apply the pruning."""
        if thresh <= 0:
            return []
        n = len(scores)
        remained_idx = torch.nonzero(scores > thresh).view(-1).tolist()
        num_remained = len(remained_idx)
        # Granularity
        if num_remained % round_to > 0:
            num_remained += round_to - (num_remained % round_to)
        # keep the min idxs
        num_remained = max(num_remained, round_to)
        num_remained = min(num_remained, n)
        if num_remained == n:
            return []
        sorted_idx = torch.argsort(-scores)
        indices = torch.sort(sorted_idx[num_remained:])[0].view(-1).tolist()

        return indices


class L1Strategy(LNStrategy):
    """L1 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L1Strategy, self).__init__(p=1)


class L2Strategy(LNStrategy):
    """L2 Strategy class."""

    def __init__(self):
        """Initialize."""
        super(L2Strategy, self).__init__(p=2)
