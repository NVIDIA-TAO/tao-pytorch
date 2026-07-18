# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import tempfile
import numpy as np
from PIL import Image
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.centerpose.dataloader.pl_cp_data_module import CPDataModule
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel


TEST_BATCH_SIZE = 4
FAST_DEV_RUN = 2  # Run dry run 2 times
NUM_IMAGES = 10
TEST_WIDTH = int(np.random.randint(low=128, high=2048, size=1)[0])
TEST_HEIGHT = int(np.random.randint(low=128, high=2048, size=1)[0])
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name


@pytest.fixture
def _test_sample_json():
    tmp_top_train_dir = os.path.join(tmp_top_dir, "train")
    check_and_create(tmp_top_train_dir)
    tmp_top_test_dir = os.path.join(tmp_top_dir, "test")
    check_and_create(tmp_top_test_dir)
    tmp_top_infer_dir = os.path.join(tmp_top_dir, "infer")
    check_and_create(tmp_top_infer_dir)
    tmp_top_val_dir = os.path.join(tmp_top_dir, "val")
    check_and_create(tmp_top_val_dir)

    for image_id in range(NUM_IMAGES):
        camera_view_matrix = np.random.rand(4, 4)
        cam_projection_matrix = np.random.rand(4, 4)
        location_world = np.random.rand(3)
        quaternion_world_xyzw = np.random.rand(4)
        cam_intrinsics = np.random.rand(3, 3)
        plane_center = np.random.rand(3)
        plane_normal = np.random.rand(3)
        json_output = {
                    "camera_data": {
                        "width": TEST_WIDTH,
                        'height': TEST_HEIGHT,
                        'camera_view_matrix': camera_view_matrix.tolist(),
                        'camera_projection_matrix':cam_projection_matrix.tolist(),
                        'location_world': location_world.tolist(),
                        'quaternion_world_xyzw': quaternion_world_xyzw.tolist(),
                        'intrinsics': {
                            'fx':cam_intrinsics[1][1],
                            'fy':cam_intrinsics[0][0],
                            'cx':cam_intrinsics[1][2],
                            'cy':cam_intrinsics[0][2]
                        }
                    }, 
                    "objects": [],
                    "AR_data": {
                        'plane_center': plane_center.tolist(),
                        'plane_normal': plane_normal.tolist()
                    }
                }

        object_translations = np.random.rand(3)
        quaternion_xyzw = np.random.rand(4)
        projected_cuboid = np.random.rand(9, 2)
        object_scale = np.random.rand(3)
        keypoints_3d = np.random.rand(9, 3)
        vis = np.random.rand(1)

        dict_obj = {
                    'class': "unit_tests",
                    'name': "unit_tests",
                    'provenance': "unit_tests",
                    'location': object_translations.tolist(),
                    'quaternion_xyzw': quaternion_xyzw.tolist(),
                    'projected_cuboid': projected_cuboid.tolist(),
                    'scale': object_scale.tolist(),
                    'keypoints_3d': keypoints_3d.tolist(),
                    'visibility': vis.tolist()
                }

        for idx in range(5):
            json_output['objects'].append(dict_obj)

        # Training data
        json_file = os.path.join(tmp_top_train_dir, f"test_{str(image_id)}.json")
        img_file = os.path.join(tmp_top_train_dir, f"test_{str(image_id)}.jpg")

        with open(json_file, 'w+') as outfile:
            json.dump(json_output, outfile)

        sample_w = int(TEST_WIDTH)
        sample_h = int(TEST_HEIGHT)
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img.save(img_file)

        # Testing data
        json_file = os.path.join(tmp_top_test_dir, f"test_{str(image_id)}.json")
        img_file = os.path.join(tmp_top_test_dir, f"test_{str(image_id)}.jpg")

        with open(json_file, 'w+') as outfile:
            json.dump(json_output, outfile)

        sample_w = int(TEST_WIDTH)
        sample_h = int(TEST_HEIGHT)
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img.save(img_file)

        # Validation data
        json_file = os.path.join(tmp_top_val_dir, f"test_{str(image_id)}.json")
        img_file = os.path.join(tmp_top_val_dir, f"test_{str(image_id)}.jpg")

        with open(json_file, 'w+') as outfile:
            json.dump(json_output, outfile)

        sample_w = int(TEST_WIDTH)
        sample_h = int(TEST_HEIGHT)
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img.save(img_file)

        # Inference data, only contains images
        img_file = os.path.join(tmp_top_infer_dir, f"test_{str(image_id)}.jpg")
        sample_w = int(TEST_WIDTH)
        sample_h = int(TEST_HEIGHT)
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img.save(img_file)


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    experiment_config.dataset.train_data = os.path.join(tmp_top_dir, "train")
    experiment_config.dataset.val_data = os.path.join(tmp_top_dir, "val")
    experiment_config.dataset.batch_size = 2

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.results_dir = tmp_top_dir
    experiment_config.dataset.test_data = os.path.join(tmp_top_dir, "test")
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.dataset.inference_data = os.path.join(tmp_top_dir, "infer")
    experiment_config.dataset.batch_size = 2
    
    # Set valid camera intrinsics for PnP solver
    # Using reasonable defaults to avoid OpenCV calibration errors
    experiment_config.inference.focal_length_x = max(TEST_WIDTH, TEST_HEIGHT) * 0.8
    experiment_config.inference.focal_length_y = max(TEST_WIDTH, TEST_HEIGHT) * 0.8
    experiment_config.inference.principle_point_x = TEST_WIDTH / 2.0
    experiment_config.inference.principle_point_y = TEST_HEIGHT / 2.0
    experiment_config.inference.skew = 0.0

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.centerpose
@pytest.mark.train
def test_trainer_fit(_test_sample_json, _train_spec):

    dm = CPDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    pt_model = CenterPosePlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      check_val_every_n_epoch=1,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      gradient_clip_val=_train_spec.train.clip_grad_val,
                      use_distributed_sampler=False,
                      sync_batchnorm=False,
                      deterministic=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.centerpose
@pytest.mark.evaluate
def test_trainer_evaluate(_test_sample_json, _eval_spec):

    dm = CPDataModule(_eval_spec.dataset)
    dm.setup(stage="test")
    pt_model = CenterPosePlModel(_eval_spec)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.centerpose
@pytest.mark.inference
def test_trainer_inference(_test_sample_json, _infer_spec):

    dm = CPDataModule(_infer_spec.dataset)
    dm.setup(stage="predict")
    pt_model = CenterPosePlModel(_infer_spec)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
