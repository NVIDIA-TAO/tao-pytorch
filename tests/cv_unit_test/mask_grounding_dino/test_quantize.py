# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Mask Grounding DINO quantization functionality."""

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import (
    ExperimentConfig,
    MaskGDINODatasetConfig
)
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.od_data_module import ODVGDataModule


TEST_BATCH_SIZE = 2
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset for testing."""
    tmp_top_obj = tempfile.TemporaryDirectory()
    tmp_top_dir = tmp_top_obj.name
    json_file = os.path.join(tmp_top_dir, "sample_json.json")

    json_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "person", "id": 1, "name": "person"},
            {"supercategory": "face", "id": 2, "name": "face"},
            {"supercategory": "bag", "id": 3, "name": "bag"}
        ]
    }
    check_and_create(tmp_top_dir)

    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH - 20, high=TEST_WIDTH + 20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT - 20, high=TEST_HEIGHT + 20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")
        img.save(img_file)

        images_info = {
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_w,
            "width": sample_w
        }
        json_output["images"].append(images_info)

        for cat_id in range(0, 5):
            annotation_id = image_id * 10 + cat_id
            category_id = int(np.random.randint(low=1, high=3, size=1)[0])
            x1 = int(np.random.randint(low=0, high=sample_w - TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=sample_h - TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            bbox = [x1, y1, w, h]
            area = bbox[2] * bbox[3]
            annotation_info = {
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            }
            json_output["annotations"].append(annotation_info)

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    yield {"tmp_dir": tmp_top_dir, "json_file": json_file, "tmp_obj": tmp_top_obj}

    tmp_top_obj.cleanup()


@pytest.fixture
def dataset_config_with_calib(temp_dataset):
    """Create dataset config with calibration data sources."""
    data_config = OmegaConf.structured(MaskGDINODatasetConfig())
    data_config.val_data_sources = {
        "image_dir": temp_dataset["tmp_dir"],
        "json_file": temp_dataset["json_file"],
        "data_type": "OD"
    }
    data_config.test_data_sources = {
        "image_dir": temp_dataset["tmp_dir"],
        "json_file": temp_dataset["json_file"],
        "data_type": "OD"
    }
    data_config.quant_calibration_data_sources = {
        "image_dir": temp_dataset["tmp_dir"],
        "json_file": temp_dataset["json_file"]
    }
    data_config.batch_size = TEST_BATCH_SIZE
    data_config.workers = 0
    data_config.max_labels = 50
    data_config.has_mask = False  # Disable mask for simpler testing
    return data_config


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_data_sources():
    """Test that MaskGDINODatasetConfig has quant_calibration_data_sources field."""
    config = MaskGDINODatasetConfig()
    assert hasattr(config, 'quant_calibration_data_sources'), (
        "MaskGDINODatasetConfig should have quant_calibration_data_sources field"
    )


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    config = ExperimentConfig()
    assert hasattr(config, 'quantize'), (
        "ExperimentConfig should have quantize field"
    )


@pytest.mark.cv_unit
def test_calib_dataloader_setup(dataset_config_with_calib):
    """Test that calibration dataloader can be set up."""
    dm = ODVGDataModule(dataset_config_with_calib)
    dm.setup(stage="calibration")
    assert dm.calib_dataset is not None, "Calibration dataset should be initialized"


@pytest.mark.cv_unit
def test_calib_dataloader_returns_data(dataset_config_with_calib):
    """Test that calibration dataloader returns data correctly."""
    dm = ODVGDataModule(dataset_config_with_calib)
    dm.setup(stage="calibration")
    calib_loader = dm.calib_dataloader()

    # Get one batch
    batch = next(iter(calib_loader))
    data, targets, _ = batch

    assert data is not None, "Data should not be None"
    assert len(targets) > 0, "Targets should not be empty"
    assert data.shape[0] == TEST_BATCH_SIZE, f"Batch size should be {TEST_BATCH_SIZE}"


@pytest.mark.cv_unit
def test_calib_dataloader_raises_without_setup(dataset_config_with_calib):
    """Test that calib_dataloader raises error if setup not called."""
    dm = ODVGDataModule(dataset_config_with_calib)
    # Don't call setup

    with pytest.raises(ValueError, match="Calibration dataset not initialized"):
        dm.calib_dataloader()
