# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for state validation in ModelQuantizer."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizer,
    QuantizationState,
    get_registry_manager,
)
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.quantizer_base import QuantizerBase
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable


class DummyBackendForStateTests(QuantizerBase, Calibratable):
    """A minimal backend for testing state validation."""

    def __init__(self, backend_kwargs=None):
        self.calibrated = False

    def prepare(self, model, config):
        return model

    def quantize(self, model, config):
        return model

    def calibrate(self, model, data_loader):
        self.calibrated = True


class TestStateValidation:
    """Test suite for state validation in ModelQuantizer."""

    def setup_method(self):
        """Set up test environment with a clean registry and dummy backend."""
        get_registry_manager().clear_all()
        register_backend("dummy_state_test")(DummyBackendForStateTests)

    def teardown_method(self):
        """Clean up registry after tests."""
        get_registry_manager().clear_all()

    def test_initial_state_is_initialized(self):
        """Test that ModelQuantizer starts in INITIALIZED state."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        assert quantizer.state == QuantizationState.INITIALIZED

    def test_prepare_transitions_to_prepared(self):
        """Test that prepare() transitions to PREPARED state."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        quantizer.prepare(model)
        assert quantizer.state == QuantizationState.PREPARED

    def test_calibrate_transitions_to_calibrated(self):
        """Test that calibrate() transitions to CALIBRATED state."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        quantizer.prepare(model)

        # Create dummy dataloader
        data = torch.randn(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, labels), batch_size=2
        )

        quantizer.calibrate(model, loader)
        assert quantizer.state == QuantizationState.CALIBRATED

    def test_quantize_transitions_to_quantized(self):
        """Test that quantize() transitions to QUANTIZED state."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        prepared = quantizer.prepare(model)
        quantizer.quantize(prepared)
        assert quantizer.state == QuantizationState.QUANTIZED

    def test_prepare_called_twice_raises_error(self):
        """Test that calling prepare() twice raises RuntimeError."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        quantizer.prepare(model)

        with pytest.raises(RuntimeError, match="prepare\\(\\) called in invalid state"):
            quantizer.prepare(model)

    def test_calibrate_without_prepare_raises_error(self):
        """Test that calling calibrate() before prepare() raises RuntimeError."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        # Create dummy dataloader
        data = torch.randn(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, labels), batch_size=2
        )

        with pytest.raises(RuntimeError, match="calibrate\\(\\) called in invalid state"):
            quantizer.calibrate(model, loader)

    def test_quantize_without_prepare_raises_error(self):
        """Test that calling quantize() before prepare() raises RuntimeError."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        with pytest.raises(RuntimeError, match="quantize\\(\\) called in invalid state"):
            quantizer.quantize(model)

    def test_quantize_after_calibration_succeeds(self):
        """Test that quantize() can be called after calibration."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        prepared = quantizer.prepare(model)

        # Create dummy dataloader
        data = torch.randn(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, labels), batch_size=2
        )

        quantizer.calibrate(prepared, loader)
        result = quantizer.quantize(prepared)

        assert quantizer.state == QuantizationState.QUANTIZED
        assert isinstance(result, nn.Module)

    def test_save_model_without_quantize_raises_error(self):
        """Test that save_model() before quantize() raises RuntimeError."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        prepared = quantizer.prepare(model)

        with pytest.raises(RuntimeError, match="save_model\\(\\) called in invalid state"):
            quantizer.save_model(model=prepared, path="/tmp/model.pth")

    def test_quantize_model_enforces_initialized_state(self):
        """Test that quantize_model() only works from INITIALIZED state."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        # First call should succeed
        quantizer.quantize_model(model)
        assert quantizer.state == QuantizationState.QUANTIZED

        # Second call should fail since we're no longer in INITIALIZED state
        model2 = nn.Linear(10, 5)
        with pytest.raises(RuntimeError, match="quantize_model\\(\\) called in invalid state"):
            quantizer.quantize_model(model2)

    def test_state_property_is_readable(self):
        """Test that the state property can be read."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)

        # Should be able to read state
        state = quantizer.state
        assert isinstance(state, QuantizationState)
        assert state == QuantizationState.INITIALIZED

    def test_full_workflow_state_transitions(self):
        """Test complete workflow state transitions."""
        config = {"backend": "dummy_state_test", "mode": "static_ptq", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        # Initial state
        assert quantizer.state == QuantizationState.INITIALIZED

        # After prepare
        prepared = quantizer.prepare(model)
        assert quantizer.state == QuantizationState.PREPARED

        # After calibrate
        data = torch.randn(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, labels), batch_size=2
        )
        quantizer.calibrate(prepared, loader)
        assert quantizer.state == QuantizationState.CALIBRATED

        # After quantize
        quantizer.quantize(prepared)
        assert quantizer.state == QuantizationState.QUANTIZED


class TestBackendAvailability:
    """Test suite for backend availability checking."""

    def setup_method(self):
        """Set up test environment with a clean registry."""
        get_registry_manager().clear_all()

    def teardown_method(self):
        """Clean up registry after tests."""
        get_registry_manager().clear_all()

    def test_unavailable_backend_raises_clear_error(self):
        """Test that using an unavailable backend raises a clear error."""
        config = {"backend": "nonexistent_backend", "layers": []}

        with pytest.raises(ValueError) as exc_info:
            ModelQuantizer(config)

        error_msg = str(exc_info.value)
        assert "not available" in error_msg
        assert "Available backends" in error_msg
        assert "pip install" in error_msg

    def test_available_backend_succeeds(self):
        """Test that an available backend can be instantiated."""
        # Register a backend
        register_backend("test_available")(DummyBackendForStateTests)

        config = {"backend": "test_available", "layers": []}
        quantizer = ModelQuantizer(config)

        assert quantizer is not None
        assert quantizer.config.backend == "test_available"

    def test_backend_error_provides_context(self):
        """Test that backend loading errors provide helpful context."""
        # Register a backend that will be in the registry
        register_backend("test_backend")(DummyBackendForStateTests)

        # Now try with a backend that's not registered
        config = {"backend": "another_nonexistent", "layers": []}

        with pytest.raises(ValueError) as exc_info:
            ModelQuantizer(config)

        # Should mention available backends
        error_msg = str(exc_info.value)
        assert "Available backends" in error_msg or "not available" in error_msg


class TestStateValidationErrorMessages:
    """Test that state validation error messages are helpful."""

    def setup_method(self):
        """Set up test environment."""
        get_registry_manager().clear_all()
        register_backend("dummy_state_test")(DummyBackendForStateTests)

    def teardown_method(self):
        """Clean up registry."""
        get_registry_manager().clear_all()

    def test_prepare_error_message_includes_hint(self):
        """Test that prepare() error includes helpful hint."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        quantizer.prepare(model)

        with pytest.raises(RuntimeError) as exc_info:
            quantizer.prepare(model)

        error_msg = str(exc_info.value)
        assert "Hint:" in error_msg
        assert "new ModelQuantizer instance" in error_msg

    def test_calibrate_error_message_includes_hint(self):
        """Test that calibrate() error includes helpful hint."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        data = torch.randn(4, 10)
        labels = torch.zeros(4, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, labels), batch_size=2
        )

        with pytest.raises(RuntimeError) as exc_info:
            quantizer.calibrate(model, loader)

        error_msg = str(exc_info.value)
        assert "Hint:" in error_msg
        assert "prepare()" in error_msg

    def test_quantize_error_message_includes_hint(self):
        """Test that quantize() error includes helpful hint."""
        config = {"backend": "dummy_state_test", "layers": []}
        quantizer = ModelQuantizer(config)
        model = nn.Linear(10, 5)

        with pytest.raises(RuntimeError) as exc_info:
            quantizer.quantize(model)

        error_msg = str(exc_info.value)
        assert "Hint:" in error_msg
        assert "prepare()" in error_msg
