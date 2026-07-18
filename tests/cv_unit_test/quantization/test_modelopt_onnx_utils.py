# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ModelOpt ONNX utils module."""

from __future__ import annotations

import logging
import pytest
import tempfile
import torch.nn as nn
import numpy as np
from typing import cast

from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
)
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import (
    convert_tao_to_modelopt_onnx_params,
    _dtype_to_quantize_mode,
    _extract_op_types_to_quantize,
    _extract_nodes_to_exclude,
    _determine_calibration_method,
)


class TestDtypeToQuantizeMode:
    """Test suite for _dtype_to_quantize_mode function."""

    def test_int8_dtype(self):
        """Test int8 dtype conversion."""
        result = _dtype_to_quantize_mode("int8")
        assert result == "int8"

    def test_int8_dtype_case_insensitive(self):
        """Test int8 dtype conversion with different cases."""
        result = _dtype_to_quantize_mode("INT8")
        assert result == "int8"

        result = _dtype_to_quantize_mode("Int8")
        assert result == "int8"

    def test_fp8_e4m3fn_dtype(self):
        """Test FP8 E4M3FN dtype conversion."""
        result = _dtype_to_quantize_mode("fp8_e4m3fn")
        assert result == "fp8"

    def test_fp8_e5m2_dtype(self):
        """Test FP8 E5M2 dtype conversion."""
        result = _dtype_to_quantize_mode("fp8_e5m2")
        assert result == "fp8"

    def test_fp8_dtype_case_insensitive(self):
        """Test FP8 dtype conversion with different cases."""
        result = _dtype_to_quantize_mode("FP8_E4M3FN")
        assert result == "fp8"

        result = _dtype_to_quantize_mode("Fp8_E5M2")
        assert result == "fp8"

    def test_unsupported_dtype(self):
        """Test unsupported dtype raises error."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _dtype_to_quantize_mode("fp16")

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _dtype_to_quantize_mode("invalid")


class TestExtractOpTypesToQuantize:
    """Test suite for _extract_op_types_to_quantize function."""

    def test_linear_modules(self):
        """Test extraction of linear module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Linear"),
                LayerQuantizationConfig(module_name="linear_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Gemm" in op_types

    def test_conv_modules(self):
        """Test extraction of conv module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Conv2d"),
                LayerQuantizationConfig(module_name="conv_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Conv" in op_types

    def test_matmul_modules(self):
        """Test extraction of matmul module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="MatMul"),
                LayerQuantizationConfig(module_name="matmul_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "MatMul" in op_types

    def test_add_modules(self):
        """Test extraction of add module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Add"),
                LayerQuantizationConfig(module_name="add_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Add" in op_types

    def test_mul_modules(self):
        """Test extraction of mul module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Mul"),
                LayerQuantizationConfig(module_name="mul_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Mul" in op_types

    def test_pool_modules(self):
        """Test extraction of pool module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="AvgPool2d"),
                LayerQuantizationConfig(module_name="MaxPool2d"),
                LayerQuantizationConfig(module_name="pool_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "AveragePool" in op_types
        assert "MaxPool" in op_types

    def test_batchnorm_modules(self):
        """Test extraction of batchnorm module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="BatchNorm2d"),
                LayerQuantizationConfig(module_name="bn_layer"),
                LayerQuantizationConfig(module_name="batchnorm_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "BatchNormalization" in op_types

    def test_clip_modules(self):
        """Test extraction of clip module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Clip"),
                LayerQuantizationConfig(module_name="clip_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Clip" in op_types

    def test_global_modules(self):
        """Test extraction of global module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="GlobalAvgPool"),
                LayerQuantizationConfig(module_name="global_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "GlobalAveragePool" in op_types

    def test_transpose_modules(self):
        """Test extraction of transpose module op types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="ConvTranspose2d"),
                LayerQuantizationConfig(module_name="transpose_layer"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "ConvTranspose" in op_types

    def test_multiple_module_types(self):
        """Test extraction of multiple module types."""
        config = ModelQuantizationConfig(
            layers=[
                LayerQuantizationConfig(module_name="Linear"),
                LayerQuantizationConfig(module_name="Conv2d"),
                LayerQuantizationConfig(module_name="BatchNorm2d"),
            ]
        )

        op_types = _extract_op_types_to_quantize(config)
        assert op_types is not None
        assert isinstance(op_types, list)
        assert "Gemm" in op_types
        assert "Conv" in op_types
        assert "BatchNormalization" in op_types

    def test_empty_layers(self):
        """Test with empty layers list."""
        config = ModelQuantizationConfig(layers=[])
        op_types = _extract_op_types_to_quantize(config)
        assert not op_types

    def test_none_layers(self):
        """Test with None layers."""
        config = ModelQuantizationConfig(layers=None)
        op_types = _extract_op_types_to_quantize(config)
        assert not op_types

    def test_invalid_layer_type(self):
        """Test with invalid layer type."""
        config = ModelQuantizationConfig(layers=[
            LayerQuantizationConfig(module_name="unknown_module_type")
        ])
        op_types = _extract_op_types_to_quantize(config)
        assert not op_types


class TestExtractNodesToExclude:
    """Test suite for _extract_nodes_to_exclude function."""

    def test_no_skip_names(self):
        """Test with no skip names."""
        config = ModelQuantizationConfig(skip_names=None)

        nodes = _extract_nodes_to_exclude(config)
        assert not nodes

    def test_empty_skip_names(self):
        """Test with empty skip names."""
        config = ModelQuantizationConfig(skip_names=[])

        nodes = _extract_nodes_to_exclude(config)
        assert not nodes

    def test_simple_model_with_skip_names(self):
        """Test with simple model and skip names."""
        config = ModelQuantizationConfig(skip_names=["linear1"])

        nodes = _extract_nodes_to_exclude(config)
        assert "linear1" in nodes

    def test_complex_model_with_skip_names(self):
        """Test with complex model and skip names."""
        config = ModelQuantizationConfig(skip_names=["conv1", "bn1"])

        nodes = _extract_nodes_to_exclude(config)
        assert "conv1" in nodes
        assert "bn1" in nodes

    def test_with_onnx_node_names(self):
        """Test with ONNX node names in skip_names."""
        config = ModelQuantizationConfig(skip_names=["linear", "/conv1/Conv"])

        nodes = _extract_nodes_to_exclude(config)
        assert "linear" in nodes
        assert "/conv1/Conv" in nodes


class TestDetermineCalibrationMethod:
    """Test suite for _determine_calibration_method function."""

    def test_explicit_algorithm_entropy(self):
        """Test with explicit entropy algorithm."""
        config = ModelQuantizationConfig(algorithm="entropy")
        method = _determine_calibration_method(config)
        assert method == "entropy"

    def test_explicit_algorithm_max(self):
        """Test with explicit max algorithm."""
        config = ModelQuantizationConfig(algorithm="max")
        method = _determine_calibration_method(config)
        assert method == "max"

    def test_explicit_algorithm_awq_clip(self):
        """Test with explicit awq_clip algorithm."""
        config = ModelQuantizationConfig(algorithm="awq_clip")
        method = _determine_calibration_method(config)
        assert method == "awq_clip"

    def test_explicit_algorithm_awq_lite(self):
        """Test with explicit awq_lite algorithm."""
        config = ModelQuantizationConfig(algorithm="awq_lite")
        method = _determine_calibration_method(config)
        assert method == "awq_lite"

    def test_explicit_algorithm_awq_full(self):
        """Test with explicit awq_full algorithm."""
        config = ModelQuantizationConfig(algorithm="awq_full")
        method = _determine_calibration_method(config)
        assert method == "awq_full"

    def test_explicit_algorithm_rtn_dq(self):
        """Test with explicit rtn_dq algorithm."""
        config = ModelQuantizationConfig(algorithm="rtn_dq")
        method = _determine_calibration_method(config)
        assert method == "rtn_dq"

    def test_algorithm_case_insensitive(self):
        """Test algorithm case insensitive."""
        config = ModelQuantizationConfig(algorithm="MAX")
        method = _determine_calibration_method(config)
        assert method == "max"

    def test_algorithm_from_mode_static_ptq(self):
        """Test algorithm from static PTQ mode."""
        config = ModelQuantizationConfig(mode=QuantizationMode.STATIC_PTQ)
        method = _determine_calibration_method(config)
        assert method == "max"

    def test_algorithm_from_mode_string(self):
        """Test algorithm from mode string."""
        config = ModelQuantizationConfig(mode="static_ptq")
        method = _determine_calibration_method(config)
        assert method == "max"

    def test_algorithm_fallback(self):
        """Test algorithm fallback."""
        config = ModelQuantizationConfig(mode="unknown_mode")
        method = _determine_calibration_method(config)
        assert method == "max"

    def test_unsupported_algorithm(self):
        """Test unsupported algorithm falls back to mode."""
        config = ModelQuantizationConfig(algorithm="unsupported")
        method = _determine_calibration_method(config)
        assert method == "max"


class TestConvertTaoToModeloptOnnxParams:
    """Test suite for convert_tao_to_modelopt_onnx_params function."""

    def test_basic_conversion(self):
        """Test basic parameter conversion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="int8"),
                        activations=ActivationQuantizationConfig(dtype="int8"),
                    )
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path
            )

            assert params["onnx_path"] == onnx_path
            assert params["quantize_mode"] == "int8"
            assert params["calibration_method"] == "max"
            assert params["op_types_to_quantize"] is not None
            assert isinstance(params["op_types_to_quantize"], list)
            op_types = cast(list, params["op_types_to_quantize"])
            assert "Gemm" in op_types  # pylint: disable=unsupported-membership-test
            assert not params["nodes_to_exclude"]
            assert params["output_path"] == f"{temp_dir}/quantized_model.onnx"

    def test_fp8_conversion(self):
        """Test FP8 parameter conversion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                    )
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path
            )

            assert params["quantize_mode"] == "fp8"

    def test_with_calibration_data(self):
        """Test with calibration data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(module_name="Linear")
                ],
                results_dir=temp_dir
            )

            # Use smaller test data
            calibration_data = np.random.randn(10, 3, 8, 8)

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path,
                calibration_data=calibration_data
            )

            assert params["calibration_data"] is calibration_data

    def test_with_custom_output_path(self):
        """Test with custom output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[LayerQuantizationConfig(module_name="Linear")],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            custom_output_path = f"{temp_dir}/custom_quantized.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path,
                output_path=custom_output_path
            )

            assert params["output_path"] == custom_output_path

    def test_with_backend_kwargs(self):
        """Test with backend kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[LayerQuantizationConfig(module_name="Linear")],
                results_dir=temp_dir
            )

            backend_kwargs = {
                "custom_param": "custom_value",
                "calibration_method": "entropy"
            }

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path,
                backend_kwargs=backend_kwargs
            )

            assert params["custom_param"] == "custom_value"
            assert params["calibration_method"] == "entropy"  # Overridden

    def test_with_model_and_skip_names(self):
        """Test with model and skip names."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(5, 1)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TestModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[LayerQuantizationConfig(module_name="Linear")],
                skip_names=["linear1"],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=model,
                onnx_path=onnx_path
            )

            assert "linear1" in params["nodes_to_exclude"]

    def test_multiple_layer_types(self):
        """Test with multiple layer types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(module_name="Linear"),
                    LayerQuantizationConfig(module_name="Conv2d"),
                    LayerQuantizationConfig(module_name="BatchNorm2d"),
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path
            )

            op_types_list = cast(list, params["op_types_to_quantize"])
            assert op_types_list is not None
            assert isinstance(op_types_list, list)
            assert "Gemm" in op_types_list  # pylint: disable=unsupported-membership-test
            assert "Conv" in op_types_list  # pylint: disable=unsupported-membership-test
            assert "BatchNormalization" in op_types_list  # pylint: disable=unsupported-membership-test

    def test_none_config_raises_error(self):
        """Test that None config raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = f"{temp_dir}/model.onnx"
            with pytest.raises(TypeError, match="config cannot be None"):
                convert_tao_to_modelopt_onnx_params(
                    config=None,
                    model=None,
                    onnx_path=onnx_path
                )

    def test_algorithm_override(self):
        """Test algorithm override in backend kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[LayerQuantizationConfig(module_name="Linear")],
                algorithm="max",
                results_dir=temp_dir
            )

            backend_kwargs = {"calibration_method": "entropy"}

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path,
                backend_kwargs=backend_kwargs
            )

            # Backend kwargs should override config algorithm
            assert params["calibration_method"] == "entropy"

    def test_empty_layers_config(self):
        """Test with empty layers configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path
            )

            assert not params["op_types_to_quantize"]
            assert params["quantize_mode"] == "int8"  # Default

    def test_no_layers_config(self):
        """Test with no layers configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=None,
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            params = convert_tao_to_modelopt_onnx_params(
                config=config,
                model=None,
                onnx_path=onnx_path
            )

            assert not params["op_types_to_quantize"]
            assert params["quantize_mode"] == "int8"  # Default

    def test_warning_different_dtype_within_layer(self, caplog):
        """Test warning when weights and activations have different dtypes within the same layer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="int8"),
                        activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn"),
                    )
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            with caplog.at_level(logging.WARNING):
                params = convert_tao_to_modelopt_onnx_params(
                    config=config,
                    model=None,
                    onnx_path=onnx_path
                )

            # Check that warning was issued
            assert any(
                "different dtypes for weights and activations" in record.message
                for record in caplog.records
            )
            # Check that unsupported message is present
            assert any(
                "currently unsupported" in record.message
                for record in caplog.records
            )
            # Check that the first dtype wins (weights dtype)
            assert params["quantize_mode"] == "int8"

    def test_warning_different_dtype_across_layers(self, caplog):
        """Test warning when different layers have different dtypes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="int8"),
                    ),
                    LayerQuantizationConfig(
                        module_name="Conv2d",
                        weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                    ),
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            with caplog.at_level(logging.WARNING):
                params = convert_tao_to_modelopt_onnx_params(
                    config=config,
                    model=None,
                    onnx_path=onnx_path
                )

            # Check that warning was issued
            assert any("multiple different dtypes across layers" in record.message
                       for record in caplog.records)
            # Check that the first dtype wins
            assert params["quantize_mode"] == "int8"

    def test_warning_both_mismatch_types(self, caplog):
        """Test warning when both within-layer and across-layer dtype mismatches occur."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="int8"),
                        activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn"),
                    ),
                    LayerQuantizationConfig(
                        module_name="Conv2d",
                        weights=WeightQuantizationConfig(dtype="fp8_e5m2"),
                    ),
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            with caplog.at_level(logging.WARNING):
                convert_tao_to_modelopt_onnx_params(
                    config=config,
                    model=None,
                    onnx_path=onnx_path
                )

            # Check that both warnings were issued
            assert any("different dtypes for weights and activations" in record.message
                       for record in caplog.records)
            assert any("multiple different dtypes across layers" in record.message
                       for record in caplog.records)

    def test_no_warning_same_dtype(self, caplog):
        """Test no warning when all dtypes are the same."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelQuantizationConfig(
                layers=[
                    LayerQuantizationConfig(
                        module_name="Linear",
                        weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                        activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn"),
                    ),
                    LayerQuantizationConfig(
                        module_name="Conv2d",
                        weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                        activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn"),
                    ),
                ],
                results_dir=temp_dir
            )

            onnx_path = f"{temp_dir}/model.onnx"
            with caplog.at_level(logging.WARNING):
                params = convert_tao_to_modelopt_onnx_params(
                    config=config,
                    model=None,
                    onnx_path=onnx_path
                )

            # Check that no warnings about dtype mismatches were issued
            assert not any("different dtypes" in record.message
                           for record in caplog.records)
            assert params["quantize_mode"] == "fp8"
