# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Unit test for NVPanoptix3D dataloader. """

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader import (
    preprocessor as preproc_mod
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.datasets import (
    Front3DDataset,
    Matterport3DDataset,
    NVPanoptix3DPredictDataset,
)


# Declare vars
IMAGE_SIZE = (240, 320)  # H, W
TARGET_SIZE = (320, 240)  # due to PIL save (W, H) format
CROP_SIZE = (224, 224)
ENABLE_CROP = False
NUM_CLASS = 10
DEPTH_SIZE = (240, 320)
PROCESS_DEPTH_SIZE = (120, 160)
MASK_SIZE = (240, 320)
VOLUME_SIZE = (256, 256, 256)
DEPTH_MIN = 0.4
DEPTH_MAX = 6.0


def mock_pyexr_for_dataset():
    """Create a mock pyexr module for testing."""
    mock_pyexr = MagicMock()
    depth_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    mock_pyexr.read = MagicMock(
        return_value=np.random.rand(*depth_shape).astype(np.float32)
    )
    return mock_pyexr


# Helper functions for creating synthetic test data
def create_test_image(
    path: str,
    size: tuple = TARGET_SIZE,
    mode: str = "RGB"
):
    """Create a test image file.

    Args:
        path: Path to save image
        size: (width, height) tuple in PIL format
        mode: PIL image mode
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # size is already in PIL's (width, height) format
    if mode == "L":
        img = Image.new(mode, size, color=128)
    else:
        img = Image.new(mode, size, color=(128, 128, 128))
    img.save(path)
    return path


def create_test_depth(
    path: str,
    size: tuple = DEPTH_SIZE,
    ext: str = ".npy"
):
    """Create a test depth file.

    Args:
        path: Path to save depth
        size: (width, height) tuple - numpy (height, width)
        ext: File extension
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth = np.random.rand(size[0], size[1]).astype(np.float32) * DEPTH_MAX

    if ext == ".npy":
        np.save(path, depth)
    elif ext == ".exr":
        # Save as npy and rename (pyexr not always available)
        np.save(path.replace(".exr", ".npy"), depth)
        if os.path.exists(path.replace(".exr", ".npy")):
            os.rename(path.replace(".exr", ".npy"), path)
    return path


def create_test_mask(
    path: str,
    size: tuple = MASK_SIZE,
    channels: int = 1
):
    """Create a test segmentation mask.

    Args:
        path: Path to save mask
        size: (width, height) tuple in PIL format
        channels: Number of channels
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".npz"):
        # Create instance and semantic masks
        if "3d" in path or "segmentation" in path:
            # 3D mask shape [h, w, d, 2], last dim is [semantic, instance]
            mask_3d = np.zeros(VOLUME_SIZE + (2,), dtype=np.uint8)
            h_max = min(100, VOLUME_SIZE[0])
            w_max = min(100, VOLUME_SIZE[1])
            d_max = min(100, VOLUME_SIZE[2])
            mask_3d[:h_max, :w_max, :d_max, 0] = 1  # semantic
            mask_3d[:h_max, :w_max, :d_max, 1] = 1  # instance
            np.savez_compressed(path, data=mask_3d)
        else:
            # 2D mask (height, width, channels)
            np_shape = size + (2,)
            mask = np.zeros(np_shape, dtype=np.uint8)
            h_max = min(100, IMAGE_SIZE[0])
            w_max = min(100, IMAGE_SIZE[1])
            mask[:h_max, :w_max, 0] = 1  # semantic
            mask[:h_max, :w_max, 1] = 1  # instance
            np.savez_compressed(path, data=mask)
    else:
        # PNG mask - single channel with class IDs
        # Convert (H, W) to (W, H)
        np_shape = size[::-1]
        mask = np.zeros(np_shape, dtype=np.uint8)
        if channels == 1:
            h_max = min(100, IMAGE_SIZE[0])
            w_max = min(100, IMAGE_SIZE[1])
            mask[:h_max, :w_max] = NUM_CLASS
        img = Image.fromarray(mask)
        img.save(path)
    return path


def create_test_volume(
    path: str,
    vol_type: str = "tsdf",
    size: tuple = VOLUME_SIZE
):
    """Create test 3D volume data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if vol_type == "tsdf" or "geometry" in path:
        # TSDF volume
        vol_shape = (size[0], size[1], size[2])
        tsdf = np.random.randn(*vol_shape).astype(np.float32) * 0.1
        mask = np.ones(size, dtype=bool)
        np.savez_compressed(path, data=tsdf, mask=mask)
    elif vol_type == "occupancy":
        # Occupancy volume
        occ = np.random.rand(size[0], size[1], size[2]).astype(np.float32)
        np.savez_compressed(path, data=occ)
    elif vol_type == "weights":
        # Weight volume
        weights = np.ones(size, dtype=np.float32)
        np.savez_compressed(path, data=weights)
    return path


def create_test_config(
    dataset_type: str = "synthetic",
    enable_3d: bool = True
):
    """Create a test configuration."""
    is_matterport = dataset_type == "matterport"

    config = {
        "downsample_factor": 2 if is_matterport else 1,
        "iso_value": 2.0 if is_matterport else 1.0,
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "ignore_label": 255,
        "min_instance_pixels": 100,
        "img_format": "RGB",
        "target_size": TARGET_SIZE,
        "depth_bound": True,
        "depth_size": PROCESS_DEPTH_SIZE,
        "depth_min": DEPTH_MIN,
        "depth_max": DEPTH_MAX,
        "enable_3d": enable_3d,
        "occ_truncation_lvl": [0.5, 0.5],
        "truncation_range": [0.0, 3.0],
        "enable_mp_occ": False,
        "augmentation": {
            "gen_aug_weight": 0.0,
            "size_divisibility": -1,
            "train_min_size": [IMAGE_SIZE[0]],
            "train_max_size": 960,
            "test_min_size": IMAGE_SIZE[0],
            "test_max_size": 960,
            "enable_crop": ENABLE_CROP,
            "crop_size": CROP_SIZE,
            "single_category_max_area": 1.0,
            "color_aug_ssd": True,
            "random_flip": "none",
            "random_flip_prob": 0.0,
        }
    }

    return OmegaConf.create(config)


def create_synthetic_dataset_files(temp_dir: str):
    """Create synthetic dataset directory structure and files."""
    # Create JSON metadata
    cat1 = {
        "color": [220, 20, 60],
        "isthing": 1,
        "id": 1,
        "trainId": 1,
        "name": "cabinet"
    }
    cat2 = {
        "color": [102, 102, 156],
        "isthing": 0,
        "id": 10,
        "trainId": 10,
        "name": "wall"
    }

    metadata = {
        "categories": [cat1, cat2],
        "intrinsic": [[277.0, 0.0, 160.0, 0.0],
                      [0.0, 277.0, 120.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]],
        "frustum_mask_file_name": os.path.join(
            temp_dir,
            "frustum_mask.npz"
        ),
        "images": {
            "test_0001": {
                "config_id": "config1",
                "scene_id": "scene_001",
                "image_id": "img_001",
                "height": IMAGE_SIZE[0],
                "width": IMAGE_SIZE[1],
            }
        }
    }

    # Create frustum mask
    frustum_mask = np.ones(VOLUME_SIZE, dtype=bool)
    np.savez_compressed(
        metadata["frustum_mask_file_name"],
        data=frustum_mask
    )

    # Save JSON
    json_path = os.path.join(temp_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)

    # Create data files
    base_dir = "data/config1/scene_001/img_001"
    base_data_dir = os.path.join(temp_dir, base_dir)

    create_test_image(
        os.path.join(base_data_dir, "rgb/rgb_001.png")
    )
    create_test_depth(
        os.path.join(
            base_data_dir,
            "distance_to_camera/distance_to_camera_001.npy"
        )
    )
    create_test_mask(
        os.path.join(
            base_data_dir,
            "semantic_segmentation/semantic_segmentation_001.png"
        )
    )
    create_test_mask(
        os.path.join(
            base_data_dir,
            "instance_segmentation/instance_segmentation_001.png"
        )
    )
    create_test_volume(
        os.path.join(
            base_data_dir,
            "volumes/camera_params_001_tsdf.npz"
        ),
        "tsdf"
    )
    create_test_volume(
        os.path.join(
            base_data_dir,
            "volumes/camera_params_001_occupancy.npz"
        ),
        "occupancy"
    )
    create_test_mask(
        os.path.join(
            base_data_dir,
            "volumes/camera_params_001_segmentation.npz"
        )
    )
    create_test_volume(
        os.path.join(
            base_data_dir,
            "volumes/camera_params_001_weights.npz"
        ),
        "weights"
    )

    return json_path, temp_dir


def create_front3d_dataset_files(temp_dir: str):
    """Create Front3D dataset directory structure and files."""
    # Create JSON metadata
    metadata = [
        {
            "scene_id": "scene_001",
            "image_id": "0001",
            "height": IMAGE_SIZE[0],
            "width": IMAGE_SIZE[1],
        }
    ]

    # Save JSON
    json_path = os.path.join(temp_dir, "front3d.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)

    # Create frustum mask
    frustum_path = os.path.join(temp_dir, "frustum_mask.npz")
    frustum_mask = np.ones(VOLUME_SIZE, dtype=bool)
    np.savez_compressed(frustum_path, mask=frustum_mask)

    # Create data files
    base_data_dir = os.path.join(temp_dir, "data/scene_001")

    create_test_image(os.path.join(base_data_dir, "rgb_0001.png"))
    create_test_depth(
        os.path.join(base_data_dir, "depth_0001.exr"),
        ext=".exr"
    )

    # Front3D uses npz format for 2D masks
    seg_data = np.zeros(IMAGE_SIZE + (2,), dtype=np.uint8)
    h_max = min(100, IMAGE_SIZE[0])
    w_max = min(100, IMAGE_SIZE[1])
    seg_data[:h_max, :w_max, 0] = 10
    seg_data[:h_max, :w_max, 1] = 1
    seg_path = os.path.join(base_data_dir, "segmap_0001.mapped.npz")
    np.savez_compressed(seg_path, data=seg_data)

    create_test_volume(
        os.path.join(base_data_dir, "geometry_0001.npz"),
        "tsdf"
    )
    create_test_mask(
        os.path.join(base_data_dir, "segmentation_0001.mapped.npz")
    )
    create_test_volume(
        os.path.join(base_data_dir, "weighting_0001.npz"),
        "weights"
    )

    return json_path, frustum_path, temp_dir


def create_matterport_dataset_files(temp_dir: str):
    """Create Matterport dataset directory structure and files."""
    # Create JSON metadata
    metadata = [
        {
            "scene_id": "scene_001",
            "image_id": "room_1_2",
            "height": IMAGE_SIZE[0],
            "width": IMAGE_SIZE[1],
        }
    ]

    # Save JSON
    json_path = os.path.join(temp_dir, "matterport.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)

    # Create data files
    base_data_dir = os.path.join(temp_dir, "data/scene_001")
    depth_dir = os.path.join(temp_dir, "depth_gen/scene_001")
    room_mask_dir = os.path.join(temp_dir, "room_mask/scene_001")

    create_test_image(
        os.path.join(base_data_dir, "room_i1_2.jpg"),
        size=TARGET_SIZE
    )

    # Matterport depth as PNG (will be loaded and converted)
    os.makedirs(depth_dir, exist_ok=True)
    depth_img = Image.new("I", IMAGE_SIZE, 4000)
    depth_img.save(os.path.join(depth_dir, "room_d1_2.png"))

    # Intrinsic
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = 277.0
    intrinsic[0, 2] = 320.0
    intrinsic[1, 2] = 256.0
    np.save(
        os.path.join(base_data_dir, "room_intrinsics_1.npy"),
        intrinsic
    )

    # Room mask
    create_test_image(
        os.path.join(room_mask_dir, "room_rm1_2.png"),
        size=IMAGE_SIZE,
        mode="L"
    )

    # Segmentation
    seg_data = np.zeros(IMAGE_SIZE + (2,), dtype=np.uint8)
    h_max = min(100, IMAGE_SIZE[0])
    w_max = min(100, IMAGE_SIZE[1])
    seg_data[:h_max, :w_max, 0] = NUM_CLASS
    seg_data[:h_max, :w_max, 1] = 1
    seg_path = os.path.join(base_data_dir, "room_segmap1_2.mapped.npz")
    np.savez_compressed(seg_path, data=seg_data)

    # 3D data
    create_test_volume(
        os.path.join(base_data_dir, "room_geometry1_2.npz"),
        "tsdf"
    )
    create_test_mask(
        os.path.join(base_data_dir, "room_segmentation1_2.mapped.npz")
    )
    create_test_volume(
        os.path.join(base_data_dir, "room_weighting1_2.npz"),
        "weights"
    )

    return json_path, temp_dir


# Helper functions for common test assertions
def assert_2d_outputs(data, check_depth=True, check_instances=True):
    """Assert 2D outputs are correct."""
    # Check image
    assert "image" in data, "Missing 'image' in data"
    img_shape = tuple(data["image"].shape[1:])
    assert img_shape == IMAGE_SIZE, f"Image shape mismatch: {img_shape}"
    assert data["image"].shape[0] == 3, "Expected 3 channels (CHW format)"

    # Check semantic segmentation
    assert "sem_seg" in data, "Missing 'sem_seg' in data"
    sem_shape = tuple(data["sem_seg"].shape)
    assert sem_shape == IMAGE_SIZE, f"Sem_seg shape mismatch: {sem_shape}"
    assert data["sem_seg"].dtype == torch.long, "Expected long dtype"

    # Check depth
    if check_depth:
        assert "depth" in data, "Missing 'depth' in data"
        depth_shape = tuple(data["depth"].shape)
        assert depth_shape == IMAGE_SIZE, \
            f"Depth shape mismatch: {depth_shape}"

    # Check instances
    if check_instances:
        assert "instances" in data, "Missing 'instances' in data"
        assert "gt_masks" in data["instances"], "Missing 'gt_masks'"
        if len(data["instances"]["gt_masks"]) > 0:
            mask_shape = tuple(data["instances"]["gt_masks"].shape[1:])
            assert mask_shape == IMAGE_SIZE, \
                f"Instance masks shape mismatch: {mask_shape}"
        assert "gt_classes" in data["instances"], "Missing 'gt_classes'"
        if (check_depth and "gt_depths" in data["instances"] and
                len(data["instances"]["gt_depths"]) > 0):
            depth_shape = tuple(data["instances"]["gt_depths"].shape[1:])
            assert depth_shape == IMAGE_SIZE, \
                f"Instance depths shape mismatch: {depth_shape}"


def assert_3d_outputs(data):
    """Assert 3D outputs are correct."""
    # Check geometry
    assert "geometry" in data, "Missing 'geometry' in data"
    geom_shape = tuple(data["geometry"].shape[-3:])
    assert geom_shape == VOLUME_SIZE, \
        f"Geometry shape mismatch: {geom_shape}"

    # Check multi-scale occupancy
    assert "occupancy_256" in data, "Missing 'occupancy_256'"
    occ256_shape = tuple(data["occupancy_256"].shape[-3:])
    assert occ256_shape == VOLUME_SIZE, \
        f"Occupancy_256 shape mismatch: {occ256_shape}"

    assert "occupancy_128" in data, "Missing 'occupancy_128'"
    expected_128 = tuple(i // 2 for i in VOLUME_SIZE)
    occ128_shape = tuple(data["occupancy_128"].shape[-3:])
    assert occ128_shape == expected_128, \
        f"Occupancy_128 shape mismatch: {occ128_shape}"

    assert "occupancy_64" in data, "Missing 'occupancy_64'"
    expected_64 = tuple(i // 4 for i in VOLUME_SIZE)
    occ64_shape = tuple(data["occupancy_64"].shape[-3:])
    assert occ64_shape == expected_64, \
        f"Occupancy_64 shape mismatch: {occ64_shape}"

    # Check 3D instance masks if present
    if "instances" in data and "gt_masks_3d_256" in data["instances"]:
        n_instances = len(data["instances"]["gt_classes"])
        if n_instances > 0:
            mask3d = data["instances"]["gt_masks_3d_256"]
            assert mask3d.shape[0] == n_instances, \
                f"3D masks count mismatch: {mask3d.shape[0]}"
            mask3d_shape = tuple(mask3d.shape[1:])
            assert mask3d_shape == VOLUME_SIZE, \
                f"3D masks shape mismatch: {mask3d_shape}"


class TestFront3DDataset:
    """Test Front3DDataset functionality."""

    @pytest.fixture
    def setup_dataset(self):
        """Create dataset with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, frustum_path, _ = create_front3d_dataset_files(
                temp_dir
            )
            config = create_test_config("front3d")
            dataset = Front3DDataset(
                json_path=json_path,
                base_dir=temp_dir,
                frustum_mask_path=frustum_path,
                is_training=True,
                cfg=config
            )
            yield dataset, temp_dir, config

    def test_2d_outputs(self, setup_dataset):
        """Test 2D data outputs."""
        dataset, _, config = setup_dataset
        dataset.enable_3d = False

        # Mock pyexr if not available
        with patch.object(preproc_mod, "pyexr", mock_pyexr_for_dataset()):
            data = dataset[0]

        # Use helper function to validate all 2D outputs
        assert_2d_outputs(data, check_depth=True, check_instances=True)

    def test_3d_outputs(self, setup_dataset):
        """Test 3D data outputs."""
        dataset, _, config = setup_dataset
        dataset.enable_3d = True

        # Mock pyexr if not available
        with patch.object(preproc_mod, "pyexr", mock_pyexr_for_dataset()):
            data = dataset[0]

        # Use helper functions to validate outputs
        assert_2d_outputs(data, check_depth=True, check_instances=True)
        assert_3d_outputs(data)

    def test_mp_occ_loading(self, setup_dataset):
        """Test multiplane occupancy loading."""
        dataset, _, config = setup_dataset
        dataset.enable_mp_occ = True
        dataset.enable_3d = False

        # Mock pyexr if not available
        with patch.object(preproc_mod, "pyexr", mock_pyexr_for_dataset()):
            data = dataset[0]

        assert "mp_occ_256" in data, "Missing 'mp_occ_256' in data"
        mp_occ_shape = tuple(data["mp_occ_256"].shape[-3:])
        assert mp_occ_shape == VOLUME_SIZE, \
            f"MP occupancy shape mismatch: {mp_occ_shape}"


class TestMatterport3DDataset:
    """Test Matterport3DDataset functionality."""

    @pytest.fixture
    def setup_dataset(self):
        """Create dataset with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, _ = create_matterport_dataset_files(temp_dir)
            config = create_test_config("matterport")
            dataset = Matterport3DDataset(
                json_path=json_path,
                base_dir=temp_dir,
                is_training=True,
                cfg=config
            )
            yield dataset, temp_dir, config

    def test_2d_outputs(self, setup_dataset):
        """Test 2D data outputs (with PNG depth loading)."""
        dataset, _, config = setup_dataset
        dataset.enable_3d = False

        # Load data (Matterport loads depth from PNG directly)
        data = dataset[0]

        # Use helper function to validate all 2D outputs
        assert_2d_outputs(data, check_depth=True, check_instances=True)

    def test_3d_outputs(self, setup_dataset):
        """Test 3D data outputs."""
        dataset, _, config = setup_dataset
        dataset.enable_3d = True

        # Load data (Matterport loads depth from PNG directly)
        data = dataset[0]

        # Use helper functions to validate outputs
        assert_2d_outputs(data, check_depth=True, check_instances=True)
        assert_3d_outputs(data)


class TestNVPanoptix3DPredictDataset:
    """Test prediction dataset functionality (inference without GT)."""

    @pytest.fixture
    def setup_dataset(self):
        """Create predict dataset with test images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            create_test_image(os.path.join(temp_dir, "img1.jpg"))
            create_test_image(os.path.join(temp_dir, "img2.png"))
            create_test_image(os.path.join(temp_dir, "img3.jpg"))

            config = create_test_config()
            dataset = NVPanoptix3DPredictDataset(
                image_dir=temp_dir,
                cfg=config
            )
            yield dataset, temp_dir, config

    def test_predict_outputs(self, setup_dataset):
        """Test prediction dataset outputs."""
        dataset, _, config = setup_dataset

        # Check dataset properties
        assert len(dataset) == 3, f"Expected 3, got {len(dataset)}"
        assert any("img1" in p for p in dataset.img_list), "img1 not found"
        assert any("img2" in p for p in dataset.img_list), "img2 not found"
        assert any("img3" in p for p in dataset.img_list), "img3 not found"

        data = dataset[0]
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert "image" in data, "Missing 'image' in data"
        assert data["image"].shape[0] == 3, "Expected 3 channels (CHW format)"
        img_shape = tuple(data["image"].shape[1:])
        assert img_shape == IMAGE_SIZE, f"Shape mismatch: {img_shape}"
        assert "intrinsic" in data, "Missing 'intrinsic' in data"
        assert "frustum_mask" in data, "Missing 'frustum_mask' in data"
        assert "image_id" in data, "Missing 'image_id' in data"


# Helper function to run a specific test class
def run_single_test_class(test_class_name: str):
    """Run a single test class by name."""
    pytest.main(["-v", "-k", test_class_name, __file__])


# Helper function to run all tests
def run_all_tests():
    """Run all tests in this module."""
    pytest.main(["-v", __file__])


if __name__ == "__main__":
    run_all_tests()
