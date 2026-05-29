# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINO module."""
# Temporarily override torch versioning from DLFW so that we disable warning from fairscale
# about torch version during ddp_sharded training. Fairscale doesn't handle commit versions well
# E.g. 1.13.0a0+d0d6b1f
import torch
import re


numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", torch.__version__)
torch.__version__ = ".".join([str(numbering.group(n)) for n in range(1, 4)])
