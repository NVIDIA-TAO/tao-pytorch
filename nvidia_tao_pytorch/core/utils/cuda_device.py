# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early rank-local CUDA device binding for TAO training."""

import ctypes
import os

import torch


def bind_rank_local_cuda_device():
    """Bind this process to its rank-local CUDA device."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configured_devices = [
        int(device.strip())
        for device in os.environ.get("TAO_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if configured_devices:
        if not 0 <= local_rank < len(configured_devices):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside TAO_VISIBLE_DEVICES="
                f"{configured_devices}."
            )
        device = configured_devices[local_rank]
    else:
        device = local_rank

    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError(
            "Rank-local CUDA binding requires a CUDA-enabled PyTorch build."
        )
    cudart = ctypes.CDLL(
        f"libcudart.so.{cuda_version.split('.', maxsplit=1)[0]}"
    )
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int
    error = cudart.cudaSetDevice(device)
    if error:
        raise RuntimeError(
            f"cudaSetDevice({device}) failed with CUDA error {error}."
        )


def _bind_configured_rank_on_import():
    """Bind when launched through TAO or as a distributed child process."""
    if "TAO_VISIBLE_DEVICES" in os.environ or "LOCAL_RANK" in os.environ:
        bind_rank_local_cuda_device()


_bind_configured_rank_on_import()
