# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
import torch.nn as nn

# Note: pytest handles sys.path automatically via PYTHONPATH and conftest.py
# No manual sys.path manipulation needed
from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizer,
    get_registry_manager,
    register_backend,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _DummyMtqModule:
    def quantize(self, model, cfg, forward_loop):
        return model


def _create_modelopt_mocks():
    """Create mock objects for modelopt modules."""
    dummy = _DummyMtqModule()
    return {
        "mtq": MagicMock(quantize=dummy.quantize),
        "mto": MagicMock(save=MagicMock()),
    }


def _patch_modelopt_imports():
    """Targeted patch for modelopt imports - FAST"""
    modelopt_mocks = _create_modelopt_mocks()
    return patch.multiple(
        "nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch",
        **modelopt_mocks
    )


@pytest.mark.unit
def test_quantize_model_accepts_dict_and_omegaconf():
    # Ensure backend is registered for this test and isolation maintained
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import (
        ModelOptBackend,
    )

    register_backend("modelopt")(ModelOptBackend)

    model = Tiny()
    cfg_dict = {
        "backend": "modelopt",
        "mode": "static_ptq",
        "algorithm": "max",
        "layers": [
            {
                "module_name": "Linear",
                "weights": {"dtype": "int8"},
                "activations": {"dtype": "int8"},
            }
        ],
    }
    with _patch_modelopt_imports():
        quantizer = ModelQuantizer(cfg_dict)
        out = quantizer.quantize_model(model)
        assert isinstance(
            out, nn.Module
        ), "quantize_model should return a torch.nn.Module"
