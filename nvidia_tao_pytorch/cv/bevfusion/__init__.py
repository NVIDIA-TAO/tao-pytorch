# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion module."""

import os
from nvidia_tao_pytorch.core import TAO_PYT_CACHE


os.environ["XDG_CACHE_HOME"] = os.environ["HF_HOME"] = os.environ["MPLCONFIGDIR"] = TAO_PYT_CACHE
