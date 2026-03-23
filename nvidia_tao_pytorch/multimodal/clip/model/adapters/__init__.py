# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
