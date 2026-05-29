# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core functionality module for TAO Toolkit PyTorch implementation."""

import os


TAO_PYT_CACHE = os.getenv("TAO_TOOLKIT_CACHE", "/.cache")
