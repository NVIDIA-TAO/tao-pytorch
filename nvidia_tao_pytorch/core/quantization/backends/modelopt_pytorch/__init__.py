# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt PyTorch backend integration for TAO quantization framework.

Importing this package registers the ``modelopt.pytorch`` backend via the
``@register_backend`` decorator on ``ModelOptBackend``.
"""

from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import ModelOptBackend

__all__ = ["ModelOptBackend"]
