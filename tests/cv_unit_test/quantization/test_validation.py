# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantization validation functions."""

import pytest
from nvidia_tao_pytorch.core.quantization.validation import get_valid_dtype_options, assert_supported_dtype
from nvidia_tao_pytorch.core.quantization.constants import SupportedDtype


def test_get_valid_dtype_options():
    """
    Tests that get_valid_dtype_options returns the correct list of supported data types.
    It verifies the type, content, and consistency with the SupportedDtype enum.
    """
    valid_dtypes = get_valid_dtype_options()

    # It should return a list...
    assert isinstance(
        valid_dtypes, list
    ), "get_valid_dtype_options should return a list"
    # ...of strings.
    assert all(
        isinstance(item, str) for item in valid_dtypes
    ), "All returned dtype options should be strings"

    # And the content should be exactly what's in our enum
    expected_dtypes = [e.value for e in SupportedDtype]
    assert sorted(valid_dtypes) == sorted(
        expected_dtypes
    ), "The dtype options should match values in SupportedDtype enum"


def test_assert_supported_dtype_with_valid_dtype():
    """Test that assert_supported_dtype accepts valid dtypes without raising."""
    assert_supported_dtype("int8")
    assert_supported_dtype("fp8_e4m3fn")
    assert_supported_dtype("fp8_e5m2")


def test_assert_supported_dtype_with_none():
    """Test that assert_supported_dtype raises TypeError for None."""
    with pytest.raises(TypeError, match="dtype cannot be None"):
        assert_supported_dtype(None)


def test_assert_supported_dtype_with_invalid_dtype():
    """Test that assert_supported_dtype raises ValueError for unsupported dtypes."""
    with pytest.raises(ValueError, match="Unsupported dtype 'int4'"):
        assert_supported_dtype("int4")

    with pytest.raises(ValueError, match="Unsupported dtype 'float32'"):
        assert_supported_dtype("float32")
