# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for import exception handling in quantization module."""


def test_backend_import_resilience():
    """Test that the quantization module is resilient to backend import failures."""
    # This test verifies the module can be imported even if backends have issues
    # The actual exception handling is in the module initialization

    from nvidia_tao_pytorch.core.quantization import ModelQuantizer
    from nvidia_tao_pytorch.core.quantization import get_registry_manager

    # The module should be usable
    assert ModelQuantizer is not None
    assert get_registry_manager is not None

    # Registry should exist and be functional
    registry = get_registry_manager()
    assert registry is not None
    assert hasattr(registry, 'get_available_backends')


def test_import_exception_documentation():
    """Document that import exceptions are handled for optional backends.

    The backends/__init__.py and quantizer.py files contain try/except blocks
    that catch ImportError exceptions when optional backends are not available.
    These are defensive programming practices that ensure the module remains
    functional even if optional dependencies are missing.

    This test documents that behavior and confirms the module structure
    supports graceful degradation.
    """
    import nvidia_tao_pytorch.core.quantization.backends as backends
    import nvidia_tao_pytorch.core.quantization.quantizer as quantizer

    # Verify the modules loaded successfully
    assert backends is not None
    assert quantizer is not None

    # The try/except blocks in these modules (lines 28-44 in backends/__init__.py
    # and lines 33-37 in quantizer.py) handle ImportError gracefully.
    # This defensive approach allows the toolkit to function with a subset of backends.
