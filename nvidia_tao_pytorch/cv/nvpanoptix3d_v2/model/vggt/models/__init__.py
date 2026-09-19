# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this code are based on the VGGT project by Facebook Research (Meta):
# https://github.com/facebookresearch/vggt

"""Model module for the VGGT backbone of the NVPanoptix3Dv2 model."""

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models.aggregator import (
    Aggregator,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models.vggt import VGGT

__all__ = ["VGGT", "Aggregator"]
