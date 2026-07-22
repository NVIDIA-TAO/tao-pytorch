# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantization utility functions."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from nvidia_tao_pytorch.core.quantization.utils import match_layer, create_quantized_model_from_config


# A couple of common layer types to use in our tests
conv_layer = nn.Conv2d(3, 64, 3)
linear_layer = nn.Linear(10, 20)


@pytest.fixture(autouse=False)
def mock_all_backend_dependencies():
    """Mock all backend dependencies for tests that need quantization imports."""
    with patch.dict('sys.modules', {
        'torchao': Mock(),
        'torchao.quantization': Mock(),
        'modelopt': Mock(),
        'modelopt.torch': Mock(),
        'modelopt.torch.quantization': Mock(),
        'modelopt.torch.opt': Mock(),
        'nvidia_tao_pytorch.config.common.quantization.default_config': Mock(),
    }, clear=False), patch('nvidia_tao_pytorch.core.tlt_logging.logging') as mock_logging:
        mock_logging.info.return_value = None
        mock_logging.debug.return_value = None
        mock_logging.warning.return_value = None
        yield
        # Explicit cleanup is handled by context manager exit


@pytest.fixture
def mock_model_quantizer():
    """Mock ModelQuantizer to avoid backend dependencies."""
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    return mock_quantizer


@pytest.fixture
def mock_torch_load():
    """Mock torch.load to avoid file system dependencies."""
    with patch('torch.load') as mock_load:
        mock_load.return_value = {"weight": torch.tensor(42)}
        yield mock_load


@pytest.mark.parametrize(
    "module, module_name, pattern, expected, description",
    [
        (
            conv_layer,
            "features.conv1",
            "features.conv1",
            True,
            "Exact match on graph name",
        ),
        (
            linear_layer,
            "classifier.fc",
            "classifier.*",
            True,
            "Wildcard match on graph name",
        ),
        (
            conv_layer,
            "features.conv1",
            "*conv1",
            True,
            "Leading wildcard match on graph name",
        ),
        (linear_layer, "classifier.fc", "Conv2d", False, "Type mismatch on graph name"),
        (conv_layer, "irrelevant.path", "Conv2d", True, "Exact match on type name"),
        (linear_layer, "irrelevant.path", "Linear", True, "Exact match on type name"),
        (conv_layer, "irrelevant.path", "Conv*", True, "Wildcard match on type name"),
        (linear_layer, "irrelevant.path", "Lin*", True, "Wildcard match on type name"),
        (conv_layer, "features.conv1", "Linear", False, "No match for name or type"),
        (linear_layer, "classifier.fc", "Conv2d", False, "No match for name or type"),
        (
            conv_layer,
            "features.conv1",
            "*",
            True,
            "Global wildcard should always match",
        ),
    ],
)
def test_match_layer(module, module_name, pattern, expected, description):
    """Tests various scenarios for the match_layer function."""
    assert (
        match_layer(module, module_name, pattern) == expected
    ), f"{description}: expected {expected} for pattern '{pattern}' against module '{module_name}'"


def test_match_layer_precedence():
    """
    Tests that a match on the module's graph name is found, even if the type also matches.
    The implementation short-circuits, so this verifies the order of checks.
    """
    # This pattern matches the graph name but wouldn't match the type name.
    assert match_layer(
        conv_layer, "backbone.features.conv1", "backbone.features.*"
    ), "Graph-name match should take precedence and succeed"
    # This pattern doesn't match the graph name but does match the type name.
    assert match_layer(
        conv_layer, "backbone.features.conv1", "Conv2d"
    ), "Type-name match should also succeed if graph-name check didn't match earlier"


def test_match_layer_input_validation():
    """Tests the input validation for the match_layer function to ensure it handles bad inputs."""
    with pytest.raises(TypeError, match="module cannot be None"):
        match_layer(None, "some_name", "some_pattern")

    with pytest.raises(TypeError, match="module_name_in_graph must be a string"):
        match_layer(conv_layer, None, "some_pattern")

    with pytest.raises(TypeError, match="module_name_in_graph must be a string"):
        match_layer(conv_layer, 123, "some_pattern")

    with pytest.raises(TypeError, match="pattern must be a string"):
        match_layer(conv_layer, "some_name", None)

    with pytest.raises(TypeError, match="pattern must be a string"):
        match_layer(conv_layer, "some_name", ["a_list"])

    with pytest.raises(ValueError, match="pattern cannot be empty"):
        match_layer(conv_layer, "some_name", "")


def test_match_layer_with_mocked_dependencies(mock_all_backend_dependencies):
    """Test that match_layer works correctly with mocked backend dependencies."""
    # This test ensures that the match_layer function works independently
    # of any backend dependencies that might be imported elsewhere
    assert match_layer(conv_layer, "features.conv1", "features.conv1")
    assert match_layer(linear_layer, "classifier.fc", "Linear")
    assert not match_layer(conv_layer, "features.conv1", "Linear")


# Test classes for create_quantized_model_from_config
class DummyLightning(nn.Module):
    def __init__(self, experiment_config, **kwargs):
        super().__init__()
        self.experiment_config = experiment_config
        self.kwargs = kwargs
        self.loaded_state_dict = None
        # Add a simple layer to make it a valid PyTorch module
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


class MockConfig:
    """Mock config that supports both dot notation and dictionary access using proper mocking."""
    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            self._data[key] = value

    def __getattr__(self, name):
        """Handle attribute access using the internal data dictionary."""
        if name in self._data:
            return self._data[name]
        # Return a default MockConfig for missing keys to avoid KeyError
        return MockConfig()

    def __getitem__(self, key):
        """Handle dictionary-style access."""
        if key in self._data:
            return self._data[key]
        # Return a default MockConfig for missing keys to avoid KeyError
        return MockConfig()

    def __setitem__(self, key, value):
        """Handle dictionary-style assignment."""
        self._data[key] = value

    def __setattr__(self, name, value):
        """Handle attribute assignment."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value

    def get(self, key, default=None):
        """Dictionary-style get method."""
        return self._data.get(key, default)

    def __contains__(self, key):
        """Support 'in' operator."""
        return key in self._data

    def __iter__(self):
        """Support iteration over keys."""
        return iter(self._data.keys())

    def keys(self):
        """Dictionary-style keys method."""
        return list(self._data.keys())

    def values(self):
        """Dictionary-style values method."""
        return list(self._data.values())

    def items(self):
        """Dictionary-style items method."""
        return list(self._data.items())

    def __eq__(self, other):
        """Support equality comparison."""
        if not isinstance(other, MockConfig):
            return False
        return self._data == other._data


def make_experiment_config(backend="torchao"):
    """Create a mock experiment config for testing."""
    if backend == "torchao":
        quantize = MockConfig(
            backend=backend,
            mode="weight_only_ptq",
            layers=[]
        )
    else:
        quantize = MockConfig(
            backend=backend,
            mode="static_ptq",
            algorithm="max",
            layers=[]
        )
    return MockConfig(quantize=quantize)


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_basic(mock_quantizer_class, tmp_path, mock_torch_load, mock_all_backend_dependencies):
    """Test basic functionality of create_quantized_model_from_config."""
    # Setup mock quantizer
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    mock_quantizer_class.return_value = mock_quantizer

    exp_cfg = make_experiment_config()
    ckpt_path = tmp_path / "model.pth"
    torch.save({"weight": torch.tensor(42)}, ckpt_path)

    model = create_quantized_model_from_config(
        str(ckpt_path),
        DummyLightning,
        experiment_config=exp_cfg
    )

    assert isinstance(model, DummyLightning), f"Expected DummyLightning, got {type(model).__name__}"
    assert model.experiment_config == exp_cfg, "Experiment config not passed correctly"
    assert model.loaded_state_dict is not None, "State dict should be loaded"
    assert "model.weight" in model.loaded_state_dict, "State dict keys should be prefixed with 'model.'"

    # Verify ModelQuantizer was called correctly
    mock_quantizer_class.assert_called_once_with(exp_cfg.quantize)
    mock_quantizer.quantize_model.assert_called_once()


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_with_kwargs(mock_quantizer_class, tmp_path, mock_torch_load, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config with additional kwargs."""
    # Setup mock quantizer
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    mock_quantizer_class.return_value = mock_quantizer

    exp_cfg = make_experiment_config()
    ckpt_path = tmp_path / "model.pth"
    torch.save({"weight": torch.tensor(42)}, ckpt_path)

    model = create_quantized_model_from_config(
        str(ckpt_path),
        DummyLightning,
        experiment_config=exp_cfg,
        export=True,
        some_other_param="test"
    )

    assert isinstance(model, DummyLightning), f"Expected DummyLightning, got {type(model).__name__}"
    assert model.kwargs.get("export") is True, "Export flag should be passed"
    assert model.kwargs.get("some_other_param") == "test", "Additional kwargs should be passed"

    # Verify ModelQuantizer was called correctly
    mock_quantizer_class.assert_called_once_with(exp_cfg.quantize)
    mock_quantizer.quantize_model.assert_called_once()


def test_create_quantized_model_from_config_missing_experiment_config(tmp_path, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config with missing experiment_config."""
    ckpt_path = tmp_path / "model.pth"
    torch.save({"weight": torch.tensor(42)}, ckpt_path)

    with pytest.raises(KeyError, match="experiment_config"):
        create_quantized_model_from_config(str(ckpt_path), DummyLightning)


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_modelopt_backend(mock_quantizer_class, tmp_path, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config with modelopt backend and model_state_dict."""
    # Setup mock quantizer
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    mock_quantizer_class.return_value = mock_quantizer

    # Create config with modelopt backend
    exp_cfg = make_experiment_config(backend="modelopt.pytorch")
    ckpt_path = tmp_path / "model.pth"

    # Mock torch.load to return modelopt-style state dict
    with patch('torch.load') as mock_load:
        mock_load.return_value = {"model_state_dict": {"weight": torch.tensor(42)}}

        model = create_quantized_model_from_config(
            str(ckpt_path),
            DummyLightning,
            experiment_config=exp_cfg
        )

        assert isinstance(model, DummyLightning), f"Expected DummyLightning, got {type(model).__name__}"
        assert model.experiment_config == exp_cfg, "Experiment config not passed correctly"
        assert model.loaded_state_dict is not None, "State dict should be loaded"
        assert "model.weight" in model.loaded_state_dict, "State dict keys should be prefixed with 'model.'"

        # Verify ModelQuantizer was called correctly
        mock_quantizer_class.assert_called_once_with(exp_cfg.quantize)
        mock_quantizer.quantize_model.assert_called_once()


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_torchao_backend(mock_quantizer_class, tmp_path, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config with torchao backend."""
    # Setup mock quantizer
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    mock_quantizer_class.return_value = mock_quantizer

    # Create config with torchao backend
    exp_cfg = make_experiment_config(backend="torchao")
    ckpt_path = tmp_path / "model.pth"

    # Mock torch.load to return regular state dict
    with patch('torch.load') as mock_load:
        mock_load.return_value = {"weight": torch.tensor(42)}

        model = create_quantized_model_from_config(
            str(ckpt_path),
            DummyLightning,
            experiment_config=exp_cfg
        )

        assert isinstance(model, DummyLightning), f"Expected DummyLightning, got {type(model).__name__}"
        assert model.experiment_config == exp_cfg, "Experiment config not passed correctly"
        assert model.loaded_state_dict is not None, "State dict should be loaded"
        assert "model.weight" in model.loaded_state_dict, "State dict keys should be prefixed with 'model.'"

        # Verify ModelQuantizer was called correctly
        mock_quantizer_class.assert_called_once_with(exp_cfg.quantize)
        mock_quantizer.quantize_model.assert_called_once()


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_error_handling(mock_quantizer_class, tmp_path, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config error handling."""
    # Setup mock quantizer to raise an exception
    mock_quantizer = Mock()
    mock_quantizer.quantize_model.side_effect = RuntimeError("Quantization failed")
    mock_quantizer_class.return_value = mock_quantizer

    exp_cfg = make_experiment_config()
    ckpt_path = tmp_path / "model.pth"
    torch.save({"weight": torch.tensor(42)}, ckpt_path)

    with pytest.raises(RuntimeError, match="Quantization failed"):
        create_quantized_model_from_config(
            str(ckpt_path),
            DummyLightning,
            experiment_config=exp_cfg
        )


@patch('nvidia_tao_pytorch.core.quantization.quantizer.ModelQuantizer')
def test_create_quantized_model_from_config_file_not_found(mock_quantizer_class, tmp_path, mock_all_backend_dependencies):
    """Test create_quantized_model_from_config with non-existent file."""
    # Setup mock quantizer
    mock_quantizer = Mock()
    # The quantizer should return the model unchanged (it's already quantized)
    mock_quantizer.quantize_model.side_effect = lambda model: model
    mock_quantizer_class.return_value = mock_quantizer

    exp_cfg = make_experiment_config()
    ckpt_path = tmp_path / "nonexistent.pth"

    with pytest.raises(FileNotFoundError):
        create_quantized_model_from_config(
            str(ckpt_path),
            DummyLightning,
            experiment_config=exp_cfg
        )
