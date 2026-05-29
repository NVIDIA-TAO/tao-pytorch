# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File containing the config for early-stopping."""


from dataclasses import dataclass


@dataclass
class EarlyStoppingConfig:
    """EarlyStopping config."""

    monitor: str = ""
    patience: int = 3
    min_delta: float = 0.0
