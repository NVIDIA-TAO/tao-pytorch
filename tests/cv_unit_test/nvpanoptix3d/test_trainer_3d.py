# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test NVPanoptix3D trainer for 3D stage."""

import gc
import json
import os
import tempfile
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import \
    ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.pl_data_module import \
    NVPanoptix3DDataModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader import preprocessor as preproc_mod
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_3d import \
    NVPanoptix3DPlModule
from nvidia_tao_pytorch.cv.mask2former.utils.d2.catalog import MetadataCatalog
from nvidia_tao_pytorch.core.utilities import check_and_create


NUM_SAMPLES = 5
FAST_DEV_RUN = 0  # Run dry run 2 times
TEST_BATCH_SIZE = 1
TEST_WIDTH = 320
TEST_HEIGHT = 240
GRID_DIM = 256
INTRINSIC = np.array([
    [277.12, 0., 159.5, 0.],
    [0., 277.12, 119.5, 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]
    ]).reshape((4, 4))

classlist = [
    "cabinet", "bed", "chair",
    "sofa", "table", "desk",
    "dresser", "lamp", "other",
    "wall", "floor", "ceiling"
    ]
colors = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
          (0, 0, 230), (119, 11, 32), (190, 50, 60), (102, 102, 156), (128, 64, 128), (70, 70, 70)]
thing_classes = classlist[:9]  # first 9 are things
stuff_classes = classlist[9:]  # last 3 are stuff
labelmap = []
for i, (label, color) in enumerate(zip(classlist, colors)):
    isthing = 1 if i < 9 else 0
    labelmap.append({"color": color, "isthing": isthing, "id": i + 1, "trainId": i + 1, "name": label})


def mock_pyexr_for_dataset():
    """Create a mock pyexr module for testing."""
    mock_pyexr = MagicMock()
    depth_shape = (TEST_HEIGHT, TEST_WIDTH, 1)
    rng = np.random.RandomState(42)
    mock_pyexr.read = MagicMock(
        return_value=rng.rand(*depth_shape).astype(np.float32)
    )
    return mock_pyexr


@pytest.fixture(autouse=True)
def _mock_pyexr():
    """Mock pyexr to avoid optional dependency in tests."""
    with patch.object(preproc_mod, "pyexr", mock_pyexr_for_dataset()):
        yield


@pytest.fixture(scope="module")
def _tmp_top_dir() -> Iterator[str]:
    """Create a temporary workspace for this test module."""
    with tempfile.TemporaryDirectory() as tmp_top_dir:
        yield tmp_top_dir


@pytest.fixture(scope="module")
def _json_file(_tmp_top_dir: str) -> str:
    return os.path.join(_tmp_top_dir, "sample_3d_json.json")


@pytest.fixture(scope="module")
def _colormap_file(_tmp_top_dir: str) -> str:
    return os.path.join(_tmp_top_dir, "colormap.json")


@pytest.fixture(scope="module")
def _frustum_mask_file(_tmp_top_dir: str) -> str:
    return os.path.join(_tmp_top_dir, "frustum_mask.npz")


@pytest.fixture(autouse=True)
def _seed_and_cleanup():
    """Seed RNGs before each test and clean up GPU memory + global state after."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    yield

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if "custom" in MetadataCatalog:
        MetadataCatalog.remove("custom")



@pytest.fixture(scope="module")
def _test_sample_3d_json(
    _tmp_top_dir: str,
    _json_file: str,
    _colormap_file: str,
    _frustum_mask_file: str,
) -> None:
    """Create test data following Front3D dataset format for 3D stage."""
    np.random.seed(42)
    check_and_create(_tmp_top_dir)

    # Write colormap
    with open(_colormap_file, "w") as f:
        json.dump(labelmap, f)

    # Create frustum mask
    frustum_mask = np.ones((GRID_DIM, GRID_DIM, GRID_DIM), dtype=bool)
    np.savez_compressed(_frustum_mask_file, mask=frustum_mask)

    # Create base directories
    data_dir = os.path.join(_tmp_top_dir, "data")
    check_and_create(data_dir)

    # Prepare JSON data
    json_output = []

    for image_id in range(0, NUM_SAMPLES):
        sample_w = TEST_WIDTH
        sample_h = TEST_HEIGHT

        # Use a simple scene ID for testing
        scene_id = "test_scene"
        scene_dir = os.path.join(data_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # Generate RGB image
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))

        # Generate depth image (as EXR format is expected)
        depth_values = np.ascontiguousarray(
            np.random.uniform(0.4, 5.9, size=(sample_h, sample_w)).astype(np.float32)
        )

        segm_semantic = np.random.randint(low=1, high=11, size=(sample_h, sample_w), dtype=np.uint8)
        segm_instance = np.random.randint(low=0, high=10, size=(sample_h, sample_w), dtype=np.uint8)
        segm_2d = np.ascontiguousarray(np.stack([segm_semantic, segm_instance], axis=-1))

        segm_3d_semantic = np.random.randint(
            low=1, high=11,
            size=(GRID_DIM, GRID_DIM, GRID_DIM),
            dtype=np.uint8
            )
        segm_3d_instance = np.random.randint(
            low=0, high=10,
            size=(GRID_DIM, GRID_DIM, GRID_DIM),
            dtype=np.uint8
            )
        segm_3d = np.ascontiguousarray(np.stack([segm_3d_semantic, segm_3d_instance], axis=-1))

        # Generate geometry (SDF values) - ensure C-contiguous
        geometry = np.ascontiguousarray(
            np.random.uniform(-3.0, 3.0, size=(GRID_DIM, GRID_DIM, GRID_DIM)).astype(np.float32)
        )

        # Generate weighting mask (single array for 3D volume) - ensure C-contiguous
        weighting = np.ascontiguousarray(
            np.ones((GRID_DIM, GRID_DIM, GRID_DIM), dtype=np.float32)
        )

        # Save files using Front3D naming convention
        img_file = os.path.join(scene_dir, f"rgb_{str(image_id)}.png")
        depth_file = os.path.join(scene_dir, f"depth_{str(image_id)}.exr")
        segm_file = os.path.join(scene_dir, f"segmap_{str(image_id)}.mapped.npz")
        geometry_file = os.path.join(scene_dir, f"geometry_{str(image_id)}.npz")
        segm_3d_file = os.path.join(scene_dir, f"segmentation_{str(image_id)}.mapped.npz")
        weighting_file = os.path.join(scene_dir, f"weighting_{str(image_id)}.npz")

        # Save files
        img.save(img_file)

        # Save depth as NPY and rename to .exr so the loader path exists.
        tmp_depth = depth_file.replace(".exr", ".npy")
        np.save(tmp_depth, depth_values)
        os.replace(tmp_depth, depth_file)

        # Save 2D segmentation as npz with "data" key (shape: H, W, 2)
        np.savez(segm_file, data=segm_2d)

        # Save geometry as npz (expects "data" key)
        np.savez(geometry_file, data=geometry)

        # Save 3D segmentation masks as npz with "data" key (shape: D, H, W, 2)
        np.savez(segm_3d_file, data=segm_3d)

        # Save weighting masks as npz with "data" key
        np.savez(weighting_file, data=weighting)

        # Create sample dict in the format expected by Front3DDataset
        sample_dict = {
            "scene_id": scene_id,
            "image_id": str(image_id),
            "height": sample_h,
            "width": sample_w
        }

        json_output.append(sample_dict)

    # Save as a list of samples (not a nested dict)
    with open(_json_file, "w+") as outfile:
        json.dump(json_output, outfile)


@pytest.fixture
def _train_spec_3d(
    _tmp_top_dir: str,
    _json_file: str,
    _colormap_file: str,
    _frustum_mask_file: str,
):
    """Create training configuration for NVPanoptix3D."""
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(_tmp_top_dir, "results_train")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    # Training configuration
    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.train.optim.max_steps = 10
    experiment_config.train.optim.lr = 0.0001
    experiment_config.train.optim.weight_decay = 0.05
    experiment_config.train.optim.lr_scheduler = "WarmupPoly"
    experiment_config.train.optim.warmup_factor = 1.0
    experiment_config.train.optim.warmup_iters = 0
    experiment_config.train.optim.monitor_name = "train_loss"
    experiment_config.train.clip_grad_norm = 0.01

    # Dataset configuration
    experiment_config.dataset.name = "front3d"
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.label_map = _colormap_file
    experiment_config.dataset.downsample_factor = 1
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file
    experiment_config.dataset.iso_value = 1.0
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.enable_3d = True
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.num_thing_classes = 9
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.occ_truncation_lvl = [8.0, 6.0]
    experiment_config.dataset.truncation_range = [0.0, 3.0]

    # Augmentation config
    experiment_config.dataset.augmentation.train_min_size = [TEST_HEIGHT]
    experiment_config.dataset.augmentation.train_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.size_divisibility = -1
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    # Train dataset
    experiment_config.dataset.train.json_path = _json_file
    experiment_config.dataset.train.base_dir = _tmp_top_dir
    experiment_config.dataset.train.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.train.num_workers = 0  # Avoid multiprocessing issues in tests

    # Val dataset
    experiment_config.dataset.val.json_path = _json_file
    experiment_config.dataset.val.base_dir = _tmp_top_dir
    experiment_config.dataset.val.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.val.num_workers = 0

    # Model configuration
    experiment_config.model.mode = "panoptic"
    experiment_config.model.object_mask_threshold = 0.8
    experiment_config.model.overlap_threshold = 0.8
    experiment_config.model.test_topk_per_image = 100
    experiment_config.model.sem_seg_head.num_classes = 13

    # Backbone
    experiment_config.model.backbone.backbone_type = "vggt"
    # Use empty string to avoid torch.load(None) in backbone init.
    experiment_config.model.backbone.pretrained_model_path = ""

    # 3D Frustum configuration
    experiment_config.model.frustum3d.truncation = 3.0
    experiment_config.model.frustum3d.iso_recon_value = 1.0
    experiment_config.model.frustum3d.panoptic_weight = 25.0
    experiment_config.model.frustum3d.completion_weights = [50.0, 25.0, 10.0]
    experiment_config.model.frustum3d.surface_weight = 5.0
    experiment_config.model.frustum3d.unet_output_channels = 8
    experiment_config.model.frustum3d.unet_features = 8
    experiment_config.model.frustum3d.use_multi_scale = True
    experiment_config.model.frustum3d.grid_dimensions = GRID_DIM
    experiment_config.model.frustum3d.frustum_dims = GRID_DIM
    experiment_config.model.frustum3d.signed_channel = 3

    # Projection configuration
    experiment_config.model.projection.voxel_size = 0.03
    experiment_config.model.projection.sign_channel = True
    experiment_config.model.projection.depth_feature_dim = 256
    experiment_config.model.mask_former.num_object_queries = 10

    yield experiment_config


@pytest.fixture
def _eval_spec_3d(
    _tmp_top_dir: str,
    _json_file: str,
    _colormap_file: str,
    _frustum_mask_file: str,
):
    """Create evaluation configuration for NVPanoptix3D."""
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(_tmp_top_dir, "results_eval")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.results_dir = results_dir
    experiment_config.evaluate.num_gpus = 1

    # Dataset configuration (same as training)
    experiment_config.dataset.name = "front3d"
    experiment_config.dataset.label_map = _colormap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.enable_3d = True
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.iso_value = 1.0
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.occ_truncation_lvl = [1.0, 1.0, 1.0]
    experiment_config.dataset.truncation_range = [0.0, 3.0]
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file

    # Augmentation config
    experiment_config.dataset.augmentation.test_min_size = TEST_HEIGHT
    experiment_config.dataset.augmentation.test_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.size_divisibility = -1
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    experiment_config.dataset.val.json_path = _json_file
    experiment_config.dataset.val.base_dir = _tmp_top_dir
    experiment_config.dataset.val.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.val.num_workers = 0

    experiment_config.dataset.test.json_path = _json_file
    experiment_config.dataset.test.base_dir = _tmp_top_dir
    experiment_config.dataset.test.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test.num_workers = 0

    # Evaluation configuration
    experiment_config.model.mode = "panoptic"
    experiment_config.model.sem_seg_head.num_classes = 13
    experiment_config.model.backbone.backbone_type = "vggt"
    experiment_config.model.backbone.pretrained_model_path = ""
    experiment_config.model.mask_former.num_object_queries = 10
    experiment_config.model.mask_former.dec_layers = 10
    experiment_config.model.frustum3d.grid_dimensions = GRID_DIM
    experiment_config.model.frustum3d.frustum_dims = GRID_DIM
    experiment_config.model.frustum3d.unet_output_channels = 8
    experiment_config.model.frustum3d.unet_features = 8
    experiment_config.model.projection.depth_feature_dim = 256

    yield experiment_config


@pytest.fixture
def _infer_spec_3d(
    _tmp_top_dir: str,
    _json_file: str,
    _colormap_file: str,
    _frustum_mask_file: str,
):
    """Create inference configuration for NVPanoptix3D."""
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(_tmp_top_dir, "results_infer")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.results_dir = results_dir

    # Dataset configuration
    experiment_config.dataset.name = "front3d"
    experiment_config.dataset.label_map = _colormap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.enable_3d = True
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.iso_value = 1.0
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.occ_truncation_lvl = [1.0, 1.0, 1.0]
    experiment_config.dataset.truncation_range = [0.0, 3.0]
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file

    # Augmentation config
    experiment_config.dataset.augmentation.train_min_size = [TEST_HEIGHT]
    experiment_config.dataset.augmentation.train_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.test_min_size = TEST_HEIGHT
    experiment_config.dataset.augmentation.test_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.size_divisibility = 32
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    experiment_config.dataset.test.json_path = _json_file
    # For predict_dataloader, it needs base_dir to point to where the
    # images are
    experiment_config.dataset.test.base_dir = os.path.join(
        _tmp_top_dir, "data", "test_scene"
    )
    experiment_config.dataset.test.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test.num_workers = 0

    # Inference configuration
    experiment_config.model.mode = "panoptic"
    experiment_config.model.sem_seg_head.num_classes = 13
    experiment_config.model.backbone.backbone_type = "vggt"
    experiment_config.model.backbone.pretrained_model_path = ""
    experiment_config.model.mask_former.num_object_queries = 10
    experiment_config.model.mask_former.dec_layers = 10
    experiment_config.model.frustum3d.grid_dimensions = GRID_DIM
    experiment_config.model.frustum3d.frustum_dims = GRID_DIM
    experiment_config.model.frustum3d.unet_output_channels = 8
    experiment_config.model.frustum3d.unet_features = 8
    experiment_config.model.projection.depth_feature_dim = 256
    experiment_config.inference.images_dir = os.path.join(
        _tmp_top_dir, "data", "test_scene"
    )

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.train
def test_trainer_3d_fit(_test_sample_3d_json, _train_spec_3d):
    """Test NVPanoptix3D training for 3D stage."""
    dm = NVPanoptix3DDataModule(_train_spec_3d)
    dm.setup(stage="fit")
    pt_model = NVPanoptix3DPlModule(_train_spec_3d)

    trainer = Trainer(
        devices=_train_spec_3d.train.num_gpus,
        num_nodes=_train_spec_3d.train.num_nodes,
        max_epochs=_train_spec_3d.train.num_epochs,
        check_val_every_n_epoch=1,
        default_root_dir=_train_spec_3d.results_dir,
        accelerator="auto",
        gradient_clip_val=_train_spec_3d.train.clip_grad_norm,
        use_distributed_sampler=False,
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=FAST_DEV_RUN
    )
    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.evaluate
def test_trainer_3d_evaluate(_test_sample_3d_json, _eval_spec_3d):
    """Test NVPanoptix3D evaluation."""

    dm = NVPanoptix3DDataModule(_eval_spec_3d)
    dm.setup(stage="test")
    pt_model = NVPanoptix3DPlModule(_eval_spec_3d)

    trainer = Trainer(
        devices=_eval_spec_3d.evaluate.num_gpus,
        default_root_dir=_eval_spec_3d.results_dir,
        accelerator="auto",
        precision="16-mixed",
        fast_dev_run=FAST_DEV_RUN
    )

    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.inference
def test_trainer_3d_inference(_test_sample_3d_json, _infer_spec_3d):
    """Test NVPanoptix3D inference."""
    dm = NVPanoptix3DDataModule(_infer_spec_3d)
    dm.setup(stage="predict")
    pt_model = NVPanoptix3DPlModule(_infer_spec_3d)

    trainer = Trainer(
        devices=_infer_spec_3d.inference.num_gpus,
        default_root_dir=_infer_spec_3d.results_dir,
        accelerator="auto",
        precision="16-mixed",
        fast_dev_run=FAST_DEV_RUN
    )

    predictions = trainer.predict(pt_model, dm)
    assert predictions, "Expected non-empty predictions from 3D inference"
    if isinstance(predictions[0], list):
        total = sum(len(p) for p in predictions)
        assert total == NUM_SAMPLES, (
            f"Expected {NUM_SAMPLES} predictions, got {total}"
        )
    else:
        assert len(predictions) == NUM_SAMPLES, (
            f"Expected {NUM_SAMPLES} predictions, got {len(predictions)}"
        )
