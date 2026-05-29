# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet Classification loss Init"""

from .contrastive_loss import ContrastiveLoss

__all__ = [
    "ContrastiveLoss"
]
