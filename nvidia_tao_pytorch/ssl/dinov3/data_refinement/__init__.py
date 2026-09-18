# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-specific scoring primitives for SSL data refinement."""

from .grit import GRIT_FORMULA_VERSION, score_grit_frame

__all__ = ["GRIT_FORMULA_VERSION", "score_grit_frame"]
