# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

"""EfficientViT utils."""

from inspect import signature

import torch
import torch.nn.functional as F


def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    """Build kwargs from config."""
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def list_sum(x: list) -> any:
    """Compute list sum."""
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> any:
    """Compute list mean."""
    return list_sum(x) / len(x)


def weighted_list_sum(x: list, weights: list) -> any:
    """Weighted list sum."""
    assert len(x) == len(weights), "Mismatch between inputs and weights."
    return x[0] * weights[0] if len(x) == 1 else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])


def list_join(x: list, sep="\t", format_str="%s") -> str:
    """List join."""
    return sep.join([format_str % val for val in x])


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    """Value to list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """Value to tuple."""
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def squeeze_list(x: list or None) -> list or any:
    """Squeeze list."""
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    """Get same padding."""
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
) -> torch.Tensor:
    """Resize tensor."""
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")
