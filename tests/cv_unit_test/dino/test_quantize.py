# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for DINO quantization functionality."""

import pytest

from nvidia_tao_pytorch.config.dino.dataset import DINODatasetConfig
from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_data_sources():
    """Test that DINODatasetConfig has quant_calibration_data_sources field."""
    config = DINODatasetConfig()
    assert hasattr(config, 'quant_calibration_data_sources'), (
        "DINODatasetConfig should have quant_calibration_data_sources field"
    )


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    config = ExperimentConfig()
    assert hasattr(config, 'quantize'), (
        "ExperimentConfig should have quantize field"
    )


@pytest.mark.cv_unit
def test_quantize_config_structure():
    """Test that quantize config has expected structure."""
    config = ExperimentConfig()
    quantize_config = config.quantize

    # Check required fields exist
    assert hasattr(quantize_config, 'backend'), "quantize should have backend field"
    assert hasattr(quantize_config, 'mode'), "quantize should have mode field"
    assert hasattr(quantize_config, 'model_path'), "quantize should have model_path field"
