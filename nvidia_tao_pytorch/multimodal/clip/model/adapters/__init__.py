# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP-compatible model adapters.

This package contains model classes that wrap various vision-language models
to provide a unified interface for CLIP training.

Classes:
    BaseCLIPAdapter: Abstract base class defining the model interface
    CRADIO: C-RADIO model (NVIDIA, Commercial License)
    SigLIP2: SigLIP2 model (Google)
    OpenCLIP: OpenCLIP/NV-CLIP models
"""

from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.adapters.openclip import OpenCLIP
from nvidia_tao_pytorch.multimodal.clip.model.adapters.radio import CRADIO
from nvidia_tao_pytorch.multimodal.clip.model.adapters.siglip2 import SigLIP2

__all__ = [
    'BaseCLIPAdapter',
    'CRADIO',
    'SigLIP2',
    'OpenCLIP',
]
