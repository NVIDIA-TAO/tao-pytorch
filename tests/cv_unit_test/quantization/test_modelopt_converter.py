# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ModelOpt converter utility."""

import torch.nn as nn

from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.utils import (
    convert_tao_to_modelopt_config,
)
from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
)


def _make_basic_layer(dtype: str = "int8"):
    """Helper to create a simple Linear-layer quantization spec."""
    return LayerQuantizationConfig(
        module_name="Linear",
        weights=WeightQuantizationConfig(
            dtype=dtype,
            observer_or_fake_quant="dummy_observer",
        ),
        activations=ActivationQuantizationConfig(
            dtype=dtype,
            observer_or_fake_quant="dummy_observer",
        ),
    )


def _make_toy_model():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.final_layer = nn.Linear(20, 5)

    return ToyModel()


def test_basic_int8_conversion():
    """Verify that a straightforward INT8 PTQ spec is converted correctly."""
    cfg = ModelQuantizationConfig(
        backend="modelopt.pytorch",
        mode="static_ptq",
        algorithm="max",
        layers=[_make_basic_layer(dtype="int8")],
        skip_names=["final_layer"],
    )

    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)

    # Top-level keys
    assert set(result.keys()) == {
        "quant_cfg",
        "algorithm",
    }, "Converter should only emit 'quant_cfg' and 'algorithm' keys"
    assert (
        result["algorithm"] == "max"
    ), "Default algorithm should be 'max' for PTQ flows"

    quant_cfg = result["quant_cfg"]

    # Layer rule – pattern 'Linear' should expand to concrete name 'linear1' quantizers
    assert (
        "linear1.weight_quantizer" in quant_cfg
    ), "Weight quantizer rule missing for linear1"
    assert (
        "linear1.input_quantizer" in quant_cfg
    ), "Input quantizer rule missing for linear1"
    assert (
        quant_cfg["linear1.weight_quantizer"]["num_bits"] == 8
    ), "INT8 weights should map to 8 bits"
    assert (
        quant_cfg["linear1.input_quantizer"]["num_bits"] == 8
    ), "INT8 activations should map to 8 bits"

    # Skip rule must disable both quantizers
    assert (
        quant_cfg["final_layer.weight_quantizer"]["enable"] is False
    ), "Skip rule should disable weight quantizer"
    assert (
        quant_cfg["final_layer.input_quantizer"]["enable"] is False
    ), "Skip rule should disable input quantizer"


def test_default_disable_when_no_layers():
    """Without any layer/skip specification, conversion should disable quantization by default."""
    cfg = ModelQuantizationConfig()
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)

    qc = result["quant_cfg"]
    assert qc == {
        "default": {"enable": False}
    }, "With no layers, default quant_cfg should disable quantization"
    # Default algorithm derives from mode when not set explicitly (static_ptq -> "max")
    assert (
        result["algorithm"] == "minmax"
    ), "Algorithm should default to 'minmax' for PTQ when not provided"


def test_algorithm_override_is_propagated():
    cfg = ModelQuantizationConfig(
        backend="modelopt.pytorch",
        mode="static_ptq",
        algorithm="smoothquant",
        layers=[_make_basic_layer(dtype="int8")],
    )
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)
    assert (
        result["algorithm"] == "smoothquant"
    ), "Explicit algorithm should be propagated to ModelOpt config"


def test_entropy_algorithm_is_passed_through():
    cfg = ModelQuantizationConfig(
        backend="modelopt.pytorch",
        mode="static_ptq",
        algorithm="entropy",
        layers=[_make_basic_layer(dtype="int8")],
    )
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)
    assert result["algorithm"] == "entropy", "'entropy' should be passed through unchanged"


def test_weights_native_disables_weight_quantizer():
    layer = LayerQuantizationConfig(
        module_name="Linear",
        weights=WeightQuantizationConfig(dtype="native", observer_or_fake_quant="dummy"),
        activations=ActivationQuantizationConfig(dtype="int8", observer_or_fake_quant="dummy"),
    )
    cfg = ModelQuantizationConfig(
        backend="modelopt.pytorch",
        mode="static_ptq",
        layers=[layer],
    )
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)
    qc = result["quant_cfg"]
    assert qc["linear1.weight_quantizer"]["enable"] is False, "weights 'native' should disable weight quantizer"
    assert qc["linear1.input_quantizer"]["num_bits"] == 8, "activations should still be configured"


def test_activations_native_disables_input_quantizer():
    layer = LayerQuantizationConfig(
        module_name="Linear",
        weights=WeightQuantizationConfig(dtype="int8", observer_or_fake_quant="dummy"),
        activations=ActivationQuantizationConfig(dtype="native", observer_or_fake_quant="dummy"),
    )
    cfg = ModelQuantizationConfig(
        backend="modelopt.pytorch",
        mode="static_ptq",
        layers=[layer],
    )
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)
    qc = result["quant_cfg"]
    assert qc["linear1.input_quantizer"]["enable"] is False, "activations 'native' should disable input quantizer"
    assert qc["linear1.weight_quantizer"]["num_bits"] == 8, "weights should still be configured"


def test_class_key_rules_emitted_without_model():
    # When no model is provided, class key rules (e.g., 'nn.Linear') should be emitted
    layer = _make_basic_layer(dtype="int8")
    cfg = ModelQuantizationConfig(backend="modelopt", mode="static_ptq", layers=[layer])
    result = convert_tao_to_modelopt_config(cfg, model=None)
    qc = result["quant_cfg"]
    assert "nn.Linear" in qc, "Expected class key 'nn.Linear' when model is not provided"
    assert "*weight_quantizer" in qc["nn.Linear"], "Weight class rule should be present"
    assert "*input_quantizer" in qc["nn.Linear"], "Activation class rule should be present"
