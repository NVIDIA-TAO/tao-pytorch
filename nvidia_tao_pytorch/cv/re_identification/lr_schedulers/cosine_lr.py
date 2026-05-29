# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosine LR Scheduler Module for Re-Identification."""
import math
import torch
from typing import Dict, Any


def create_cosine_scheduler(cfg, optimizer):
    """
    Create a cosine learning rate scheduler with the given configuration and optimizer.

    Args:
        cfg (object): A configuration object with attributes for scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer for which the learning rate schedule will be set.

    Returns:
        CosineLRScheduler: A Cosine learning rate scheduler with specified attributes.
    """
    num_epochs = cfg.train.num_epochs
    lr_min = 0.002 * cfg.train.optim.base_lr
    warmup_lr_init = 0.01 * cfg.train.optim.base_lr

    warmup_t = cfg.train.optim.warmup_epochs
    noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )

    return lr_scheduler


class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements cosine learning rate decay with restarts, as described in the paper:
    https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 last_epoch=-1) -> None:
        """Initialize the CosineLRScheduler module.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            t_initial (int): Number of iterations in the first cycle.
            t_mul (float, optional): Factor to grow the cycle length after each restart. Default: 1.
            lr_min (float, optional): Minimum learning rate value. Default: 0.
            decay_rate (float, optional): Decay rate applied at each cycle restart. Default: 1.
            warmup_t (int, optional): Number of warm-up iterations. Default: 0.
            warmup_lr_init (float, optional): Initial learning rate for warm-up. Default: 0.
            warmup_prefix (bool, optional): Whether to include warm-up in t. Default: False.
            cycle_limit (int, optional): Maximum number of cycles. Default: 0.
            t_in_epochs (bool, optional): If True, treat t as epochs. Otherwise as iterations. Default: True.
            noise_range_t (tuple or None, optional): Interval to add noise, or None for no noise. Default: None.
            noise_type (str, optional): Type of noise injection ('normal' or 'uniform'). Default: 'normal'.
            noise_pct (float, optional): Noise level. Default: 0.67.
            noise_std (float, optional): Noise standard deviation (only if noise_type is 'normal'). Default: 1.0.
            noise_seed (int, optional): Seed for random noise. Default: 42.
            initialize (bool, optional): Whether to initialize the learning rate to its original value. Default: True.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        assert t_initial > 0, "The number of iterations in the first cycle must be larger than 0."
        assert lr_min >= 0, "The minimum learning rate value cannot be negative."
        self.optimizer = optimizer
        self.initialize = initialize
        self.param_group_field = "lr"
        self._initial_param_group_field = f"initial_{self.param_group_field}"
        if self.initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if self.param_group_field not in group:
                    raise KeyError(f"{self.param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[self.param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                  "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.t = 0
        super(CosineLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculate and return the learning rate for the current step.

        Returns:
            List[float]: List of learning rates for each parameter group.
        """
        if self.t < self.warmup_t:
            lrs = [self.warmup_lr_init + self.t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                self.t = self.t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - self.t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = self.t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = self.t // self.t_initial
                t_i = self.t_initial
                t_curr = self.t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        """Returns the learning rate values for the given epoch.

        Args:
            epoch (int): The current epoch.

        Returns:
            List[float] or None: Learning rates for the current epoch or None if `t_in_epochs` is False.
        """
        if self.t_in_epochs:
            return self.get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        """Returns the learning rate values for the given number of updates.

        Args:
            num_updates (int): The current number of updates.

        Returns:
            List[float] or None: Learning rates for the current update count or None if `t_in_epochs` is True.
        """
        if not self.t_in_epochs:
            return self.get_lr(num_updates)
        return None

    def get_cycle_length(self, cycles=0):
        """Returns the total length of the learning rate schedule.

        Args:
            cycles (int, optional): Number of cycles. Defaults to 0 which uses the `cycle_limit`.

        Returns:
            int: Total length of the learning rate schedule.
        """
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))

    def _add_noise(self, lrs, t):
        """Adds noise to the learning rates for exploration.

        Args:
            lrs (List[float]): Original learning rates.
            t (int): Current step or epoch.

        Returns:
            List[float]: Learning rates with or without added noise.
        """
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing scheduler state except the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the scheduler state from a dictionary.

        Args:
            state_dict (Dict[str, Any]): Scheduler state.
        """
        self.__dict__.update(state_dict)

    def step(self, metric: float = None) -> None:
        """Updates the learning rate based on the cosine schedule.

        Args:
            metric (float, optional): Current value of the metric if any. Defaults to None.
        """
        self.metric = metric
        values = self.get_lr()
        if values is not None:
            values = self._add_noise(values, self.last_epoch)
            self.update_groups(values)
            self.t += 1

    def step_update(self, num_updates: int, metric: float = None):
        """Updates the learning rate based on the number of updates.

        Args:
            num_updates (int): Current number of updates.
            metric (float, optional): Current value of the metric if any. Defaults to None.
        """
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        """Updates the learning rate of each parameter group in the optimizer.

        Args:
            values (List[float] or float): New learning rate values for each group or a single value for all.
        """
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value
