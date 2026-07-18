# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test NVPanoptix3D trainer for 2D stage."""

from collections.abc import Iterator
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import \
    ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.pl_data_module import \
    NVPanoptix3DDataModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader import preprocessor as preproc_mod
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_2d import \
    Mask2formerPlModule
from nvidia_tao_pytorch.core.utilities import check_and_create


FAST_DEV_RUN = 2  # Run dry run 2 times
TEST_BATCH_SIZE = 2
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
    mock_pyexr.read = MagicMock(
        return_value=np.random.rand(*depth_shape).astype(np.float32)
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
    return os.path.join(_tmp_top_dir, "sample_2d_json.json")


@pytest.fixture(scope="module")
def _colormap_file(_tmp_top_dir: str) -> str:
    return os.path.join(_tmp_top_dir, "colormap.json")


@pytest.fixture(scope="module")
def _frustum_mask_file(_tmp_top_dir: str) -> str:
    return os.path.join(_tmp_top_dir, "frustum_mask.npz")


@pytest.fixture(scope="module")
def _test_sample_2d_json(
    _tmp_top_dir: str,
    _json_file: str,
    _colormap_file: str,
    _frustum_mask_file: str,
) -> None:
    """Create test data following Front3D dataset format for 2D stage."""
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

    for image_id in range(0, 5):
        sample_w = TEST_WIDTH
        sample_h = TEST_HEIGHT

        # Use a simple scene ID for testing
        scene_id = "test_scene"
        scene_dir = os.path.join(data_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # Generate RGB image
        img = Image.fromarray(
            np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8)
        )

        # Generate depth image
        depth_values = np.ascontiguousarray(
            np.random.uniform(0.4, 5.9, size=(sample_h, sample_w)).astype(np.float32)
        )

        # Create semantic segmentation
        segm_semantic = np.random.randint(low=1, high=11, size=(sample_h, sample_w), dtype=np.uint8)

        # Create instance segmentation
        segm_instance = np.random.randint(low=0, high=10, size=(sample_h, sample_w), dtype=np.uint8)

        # Stack semantic and instance segmentation
        segm_2d = np.ascontiguousarray(np.stack([segm_semantic, segm_instance], axis=-1))

        # Generate geometry (SDF values) - ensure C-contiguous, for multi-plane occupancy
        geometry = np.ascontiguousarray(
            np.random.uniform(-3.0, 3.0, size=(GRID_DIM, GRID_DIM, GRID_DIM)).astype(np.float32)
        )

        # Save files using Front3D naming convention
        img_file = os.path.join(scene_dir, f"rgb_{str(image_id)}.png")
        depth_file = os.path.join(scene_dir, f"depth_{str(image_id)}.exr")
        segm_file = os.path.join(scene_dir, f"segmap_{str(image_id)}.mapped.npz")
        geometry_file = os.path.join(scene_dir, f"geometry_{str(image_id)}.npz")

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
def _train_spec_2d(
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
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.enable_3d = False
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.num_thing_classes = 9
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.img_format = "RGB"

    # Augmentation config
    experiment_config.dataset.augmentation.train_min_size = [TEST_HEIGHT]
    experiment_config.dataset.augmentation.train_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.size_divisibility = 32
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    # Train dataset
    experiment_config.dataset.train.json_path = _json_file
    experiment_config.dataset.train.base_dir = _tmp_top_dir
    experiment_config.dataset.train.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.train.num_workers = 0

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

    # Backbone
    experiment_config.model.backbone.backbone_type = "vggt"
    # Use empty string to avoid torch.load(None) in backbone init.
    experiment_config.model.backbone.pretrained_model_path = ""

    # Semantic segmentation head
    experiment_config.model.sem_seg_head.num_classes = 13
    experiment_config.model.sem_seg_head.common_stride = 4
    experiment_config.model.sem_seg_head.transformer_enc_layers = 6
    experiment_config.model.sem_seg_head.convs_dim = 256
    experiment_config.model.sem_seg_head.mask_dim = 256
    experiment_config.model.sem_seg_head.depth_dim = 256
    experiment_config.model.sem_seg_head.ignore_value = 255

    # Mask2Former
    experiment_config.model.mask_former.dropout = 0.0
    experiment_config.model.mask_former.nheads = 8
    experiment_config.model.mask_former.num_object_queries = 100
    experiment_config.model.mask_former.hidden_dim = 256
    experiment_config.model.mask_former.dim_feedforward = 2048
    experiment_config.model.mask_former.dec_layers = 10
    experiment_config.model.mask_former.pre_norm = False
    experiment_config.model.mask_former.class_weight = 2.0
    experiment_config.model.mask_former.dice_weight = 5.0
    experiment_config.model.mask_former.mask_weight = 5.0
    experiment_config.model.mask_former.depth_weight = 5.0
    experiment_config.model.mask_former.mp_occ_weight = 5.0
    experiment_config.model.mask_former.train_num_points = 12544
    experiment_config.model.mask_former.oversample_ratio = 3.0
    experiment_config.model.mask_former.importance_sample_ratio = 0.75
    experiment_config.model.mask_former.deep_supervision = True
    experiment_config.model.mask_former.no_object_weight = 0.1
    experiment_config.model.mask_former.size_divisibility = 32

    yield experiment_config


@pytest.fixture
def _eval_spec_2d(
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
    experiment_config.dataset.enable_3d = False
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.min_instance_pixels = 100
    experiment_config.dataset.img_format = "RGB"
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file

    # Augmentation config
    experiment_config.dataset.augmentation.test_min_size = TEST_HEIGHT
    experiment_config.dataset.augmentation.test_max_size = TEST_WIDTH
    experiment_config.dataset.augmentation.size_divisibility = 32
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    experiment_config.dataset.val.json_path = _json_file
    experiment_config.dataset.val.base_dir = _tmp_top_dir
    experiment_config.dataset.val.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.val.num_workers = 0

    experiment_config.dataset.test.json_path = _json_file
    experiment_config.dataset.test.base_dir = _tmp_top_dir
    experiment_config.dataset.test.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test.num_workers = 0

    # Model configuration
    experiment_config.model.mode = "panoptic"
    experiment_config.model.backbone.pretrained_model_path = ""

    yield experiment_config


@pytest.fixture
def _infer_spec_2d(
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

    # Dataset config
    experiment_config.dataset.name = "front3d"
    experiment_config.dataset.label_map = _colormap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.enable_3d = False
    experiment_config.dataset.enable_mp_occ = True
    experiment_config.dataset.target_size = [TEST_WIDTH, TEST_HEIGHT]
    experiment_config.dataset.reduced_target_size = [TEST_WIDTH // 2, TEST_HEIGHT // 2]
    experiment_config.dataset.depth_size = [TEST_HEIGHT // 2, TEST_WIDTH // 2]
    experiment_config.dataset.ignore_label = 255
    experiment_config.dataset.min_instance_pixels = 100
    experiment_config.dataset.img_format = "RGB"
    experiment_config.dataset.depth_min = 0.4
    experiment_config.dataset.depth_max = 6.0
    experiment_config.dataset.frustum_mask_path = _frustum_mask_file
    experiment_config.dataset.augmentation.gen_aug_weight = 0.0

    # Inference config
    experiment_config.inference.num_gpus = 1
    experiment_config.inference.images_dir = os.path.join(
        _tmp_top_dir, "data", "test_scene"
    )

    experiment_config.dataset.test.json_path = _json_file
    experiment_config.dataset.test.base_dir = _tmp_top_dir
    experiment_config.dataset.test.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test.num_workers = 0

    # Model configuration
    experiment_config.model.mode = "panoptic"
    experiment_config.model.backbone.pretrained_model_path = ""

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.train
def test_trainer_2d_fit(_test_sample_2d_json, _train_spec_2d):
    """Test NVPanoptix3D training for 2D stage."""
    dm = NVPanoptix3DDataModule(_train_spec_2d)
    dm.setup(stage="fit")
    pt_model = Mask2formerPlModule(_train_spec_2d)

    trainer = Trainer(
        devices=_train_spec_2d.train.num_gpus,
        num_nodes=_train_spec_2d.train.num_nodes,
        max_epochs=_train_spec_2d.train.num_epochs,
        check_val_every_n_epoch=1,
        default_root_dir=_train_spec_2d.results_dir,
        accelerator="auto",
        gradient_clip_val=_train_spec_2d.train.clip_grad_norm,
        use_distributed_sampler=False,
        fast_dev_run=FAST_DEV_RUN
    )

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.evaluate
def test_trainer_2d_evaluate(_test_sample_2d_json, _eval_spec_2d):
    """Test NVPanoptix3D evaluation on 2D stage."""
    dm = NVPanoptix3DDataModule(_eval_spec_2d)
    dm.setup(stage="test")
    pt_model = Mask2formerPlModule(_eval_spec_2d)

    trainer = Trainer(
        devices=_eval_spec_2d.evaluate.num_gpus,
        default_root_dir=_eval_spec_2d.results_dir,
        accelerator="auto",
        fast_dev_run=FAST_DEV_RUN
    )

    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.nvpanoptix3d
@pytest.mark.inference
def test_trainer_2d_inference(_test_sample_2d_json, _infer_spec_2d):
    """Test inference."""
    dm = NVPanoptix3DDataModule(_infer_spec_2d)
    dm.setup(stage="predict")
    pt_model = Mask2formerPlModule(_infer_spec_2d)

    trainer = Trainer(
        devices=_infer_spec_2d.inference.num_gpus,
        default_root_dir=_infer_spec_2d.results_dir,
        accelerator="auto",
        fast_dev_run=FAST_DEV_RUN
    )

    # Test predict
    trainer.predict(pt_model, dm)
