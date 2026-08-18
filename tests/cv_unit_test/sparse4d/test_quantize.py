# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Sparse4D quantization functionality."""

import pytest

from nvidia_tao_pytorch.config.sparse4d.dataset import (
    Omniverse3DDetTrackDatasetConfig,
    QuantCalibrationDataset,
)
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_dataset():
    """Test that Omniverse3DDetTrackDatasetConfig has quant_calibration_dataset field."""
    data_config = Omniverse3DDetTrackDatasetConfig()
    assert hasattr(data_config, "quant_calibration_dataset")
    assert isinstance(data_config.quant_calibration_dataset, QuantCalibrationDataset)


@pytest.mark.cv_unit
def test_quant_calibration_dataset_has_images_dir():
    """Test that QuantCalibrationDataset has images_dir field."""
    calib_config = QuantCalibrationDataset()
    assert hasattr(calib_config, "images_dir")
    assert calib_config.images_dir == ""


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    exp_config = ExperimentConfig()
    assert hasattr(exp_config, "quantize")


@pytest.mark.cv_unit
def test_quantize_config_has_required_fields():
    """Test that quantize config has all required fields."""
    exp_config = ExperimentConfig()
    quantize_config = exp_config.quantize
    assert hasattr(quantize_config, "backend")
    assert hasattr(quantize_config, "mode")
    assert hasattr(quantize_config, "algorithm")
    assert hasattr(quantize_config, "model_path")
    assert hasattr(quantize_config, "results_dir")
