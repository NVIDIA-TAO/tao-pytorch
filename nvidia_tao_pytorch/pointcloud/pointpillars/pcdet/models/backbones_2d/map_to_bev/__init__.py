# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Voxels to BEV(2D feature map)."""
from .pointpillar_scatter import PointPillarScatter

__all__ = {
    'PointPillarScatter': PointPillarScatter
}
