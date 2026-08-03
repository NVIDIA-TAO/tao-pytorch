# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rank-local CUDA device binding."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import torch


CUDA_DEVICE_MODULE = "nvidia_tao_pytorch.core.utils.cuda_device"


def test_cuda_binding_maps_original_rank_zero_to_first_tao_device(monkeypatch):
    """Import-time binding should map the original process to rank zero."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "2,5")
    cudart = MagicMock()
    cudart.cudaSetDevice.return_value = 0

    with patch("ctypes.CDLL", return_value=cudart):
        if CUDA_DEVICE_MODULE in sys.modules:
            importlib.reload(sys.modules[CUDA_DEVICE_MODULE])
        else:
            importlib.import_module(CUDA_DEVICE_MODULE)

    cudart.cudaSetDevice.assert_called_once_with(2)


def test_cuda_binding_uses_rank_mapped_tao_device(monkeypatch):
    """A distributed child should bind to its rank-mapped TAO device."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TAO_VISIBLE_DEVICES", raising=False)
    cuda_device = importlib.import_module(CUDA_DEVICE_MODULE)

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "2,5")
    cudart = MagicMock()
    cudart.cudaSetDevice.return_value = 0

    with patch.object(
        cuda_device.ctypes,
        "CDLL",
        return_value=cudart,
    ) as load_cudart:
        cuda_device.bind_rank_local_cuda_device()

    cuda_major = torch.version.cuda.split('.', maxsplit=1)[0]
    load_cudart.assert_called_once_with(f"libcudart.so.{cuda_major}")
    cudart.cudaSetDevice.assert_called_once_with(5)
