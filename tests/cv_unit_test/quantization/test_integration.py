# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the quantization framework."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nvidia_tao_pytorch.core.quantization import (
    QuantizerBase,
    Calibratable,
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
    get_backend_class,
    get_registry_manager,
)
from nvidia_tao_pytorch.core.quantization.registry import (
    register_observer,
    register_fake_quant,
    register_backend,
)


class DummyObserver(nn.Module):
    """A dummy observer module for integration testing."""

    def forward(self, x):
        """Return the input tensor unmodified."""
        return x


class DummyFakeQuant(nn.Module):
    """A dummy fake quantization module for integration testing."""

    def forward(self, x):
        """Return the input tensor unmodified."""
        return x


class QuantizedLinear(nn.Module):
    """A placeholder for a quantized Linear layer."""

    def __init__(self, original_linear):
        """Initialize with the original Linear layer."""
        super().__init__()
        self.original_linear = original_linear

    def forward(self, x):
        """Forward pass through the original layer."""
        return self.original_linear(x)


class DummyBackend(QuantizerBase, Calibratable):
    """A dummy backend for integration testing."""

    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Replace Linear layers with a placeholder for quantization."""
        # Create a new model with replaced layers instead of modifying in place
        # For Sequential models, create a new Sequential with replacements
        new_modules = []
        for module in model:
            if isinstance(module, nn.Linear):
                new_modules.append(QuantizedLinear(module))
            else:
                new_modules.append(module)
        return nn.Sequential(*new_modules)

    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Return the model unmodified as a no-op."""
        return model

    def calibrate(self, model: nn.Module, data_loader: DataLoader):
        """Simulate a calibration loop."""
        model.eval()
        with torch.no_grad():
            # Get first batch directly without iteration
            data, _ = next(iter(data_loader))
            model(data)


class TestQuantizationIntegration:
    """Test suite for a full dummy quantization workflow."""

    def setup_method(self):
        """Set up the test environment by registering dummy components."""
        get_registry_manager().clear_all()
        register_observer("dummy_observer")(DummyObserver)
        register_fake_quant("dummy_fake_quant")(DummyFakeQuant)
        register_backend("dummy_backend")(DummyBackend)

    def test_quantization_workflow(self):
        """Test the end-to-end quantization workflow."""
        # Smaller model for faster testing
        model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3))

        quant_config = ModelQuantizationConfig(
            backend="dummy_backend",
            layers=[
                LayerQuantizationConfig(
                    module_name="Linear",
                    weights=[
                        WeightQuantizationConfig(
                            dtype="int8",
                            observer_or_fake_quant="dummy_observer",
                        )
                    ],
                    activations=[
                        ActivationQuantizationConfig(
                            dtype="int8",
                            observer_or_fake_quant="dummy_fake_quant",
                        )
                    ],
                )
            ],
        )

        backend_class = get_backend_class("dummy_backend")
        quantizer = backend_class()

        prepared_model = quantizer.prepare(model, quant_config)

        # Essential assertions only
        assert isinstance(
            prepared_model[0], QuantizedLinear
        ), "First Linear should be wrapped with QuantizedLinear"
        assert isinstance(
            prepared_model[2], QuantizedLinear
        ), "Second Linear should be wrapped with QuantizedLinear"

        # Minimal data for faster testing
        dummy_data = torch.randn(2, 5)  # Even smaller: 2 samples, 5 features
        dummy_labels = torch.randint(0, 3, (2,))
        data_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=2)

        assert isinstance(
            quantizer, Calibratable
        ), "Dummy backend should implement Calibratable"
        quantizer.calibrate(prepared_model, data_loader)

        quantized_model = quantizer.quantize(prepared_model, quant_config)

        assert (
            quantized_model == prepared_model
        ), "Dummy backend quantize() is a no-op and should return the same model"

        # Final validation with minimal computation
        quantized_model.eval()
        with torch.no_grad():
            output = quantized_model(dummy_data)
            assert output.shape == (
                2,
                3,
            ), "Output tensor shape should match the final layer output size"

    def teardown_method(self):
        """Clean up the test environment by clearing the registry."""
        get_registry_manager().clear_all()
