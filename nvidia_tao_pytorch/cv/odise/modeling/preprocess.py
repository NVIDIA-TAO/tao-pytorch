# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import torch


def batched_input_to_device(batched_inputs, device, exclude=()):

    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(batched_inputs, torch.Tensor):
        batch = batched_inputs.to(device, non_blocking=True)
        return batch
    elif isinstance(batched_inputs, collections.abc.Mapping):
        batch = {}
        for k in batched_inputs:
            if k not in exclude:
                batched_inputs[k] = batched_input_to_device(batched_inputs[k], device)
        return batched_inputs

    elif isinstance(batched_inputs, collections.abc.Sequence) and not isinstance(
        batched_inputs, str
    ):
        return [batched_input_to_device(d, device) for d in batched_inputs]
    elif isinstance(batched_inputs, str):
        return batched_inputs
    else:
        raise TypeError(f"Unsupported type {type(batched_inputs)}")
