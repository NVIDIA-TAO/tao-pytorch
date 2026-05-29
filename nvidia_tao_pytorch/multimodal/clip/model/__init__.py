# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP model package.

This package provides CLIP-compatible model implementations for training
vision-language models.

Subpackages:
    adapters: Model classes (CRADIO, SigLIP2, OpenCLIP)
    evaluation: Zero-shot evaluation utilities

Modules:
    builders: Factory functions for building models
    tokenizers: Tokenizer utilities and wrappers
    transforms: Image preprocessing transforms
    clip: Main model builder entry point
    pl_clip_model: PyTorch Lightning module
"""

# Models
from nvidia_tao_pytorch.multimodal.clip.model.adapters import (
    BaseCLIPAdapter,
    CRADIO,
    SigLIP2,
    OpenCLIP,
)

# Tokenizers
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    canonicalize_text,
    SigLIP2WrappedTokenizer,
    CLIPCompatibleTokenizer,
)

# Transforms
from nvidia_tao_pytorch.multimodal.clip.model.transforms import (
    SigLIP2ImageTransform,
)

# Builders
from nvidia_tao_pytorch.multimodal.clip.model.builders import (
    build_radio_model,
    build_siglip2_model,
)

__all__ = [
    # Models
    'BaseCLIPAdapter',
    'CRADIO',
    'SigLIP2',
    'OpenCLIP',
    # Tokenizers
    'canonicalize_text',
    'SigLIP2WrappedTokenizer',
    'CLIPCompatibleTokenizer',
    # Transforms
    'SigLIP2ImageTransform',
    # Builders
    'build_radio_model',
    'build_siglip2_model',
]
