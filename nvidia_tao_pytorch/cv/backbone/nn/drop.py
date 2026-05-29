# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EfficientViT drop functions."""
import numpy as np
import torch
import torch.nn as nn

from nvidia_tao_pytorch.cv.backbone.nn.ops import IdentityLayer, ResidualBlock
from nvidia_tao_pytorch.cv.backbone.efficientvit_utils import build_kwargs_from_config

__all__ = ["apply_drop_func"]


class Scheduler:
    PROGRESS = 0


def apply_drop_func(network: nn.Module, drop_config: dict[str, any] or None) -> None:
    """Apply drop func."""
    if drop_config is None:
        return

    drop_lookup_table = {
        "droppath": apply_droppath,
    }

    drop_func = drop_lookup_table[drop_config["name"]]
    drop_kwargs = build_kwargs_from_config(drop_config, drop_func)

    drop_func(network, **drop_kwargs)


def apply_droppath(
    network: nn.Module,
    drop_prob: float,
    linear_decay=True,
    scheduled=True,
    skip=0,
) -> None:
    """Apply droppath."""
    all_valid_blocks = []
    for m in network.modules():
        for name, sub_module in m.named_children():
            if isinstance(sub_module, ResidualBlock) and isinstance(sub_module.shortcut, IdentityLayer):
                all_valid_blocks.append((m, name, sub_module))
    all_valid_blocks = all_valid_blocks[skip:]
    for i, (m, name, sub_module) in enumerate(all_valid_blocks):
        prob = drop_prob * (i + 1) / len(all_valid_blocks) if linear_decay else drop_prob
        new_module = DropPathResidualBlock(
            sub_module.main,
            sub_module.shortcut,
            sub_module.post_act,
            sub_module.pre_norm,
            prob,
            scheduled,
        )
        m._modules[name] = new_module


class DropPathResidualBlock(ResidualBlock):
    """Residual Block with DropPath."""

    def __init__(
        self,
        main: nn.Module,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
        ######################################
        drop_prob: float = 0,
        scheduled=True,
    ):
        super().__init__(main, shortcut, post_act, pre_norm)

        self.drop_prob = drop_prob
        self.scheduled = scheduled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if not self.training or self.drop_prob == 0 or not isinstance(self.shortcut, IdentityLayer):
            return ResidualBlock.forward(self, x)
        else:
            drop_prob = self.drop_prob
            if self.scheduled:
                drop_prob *= np.clip(Scheduler.PROGRESS, 0, 1)
            keep_prob = 1 - drop_prob

            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize

            res = self.forward_main(x) / keep_prob * random_tensor + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
            return res
