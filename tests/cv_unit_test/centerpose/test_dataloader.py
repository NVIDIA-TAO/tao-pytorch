# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.centerpose.dataloader.pl_cp_data_module import CPDataModule


TEST_BATCH_SIZE = 4
TEST_WIDTH = int(np.random.randint(low=128, high=2048, size=1)[0])
TEST_HEIGHT = int(np.random.randint(low=128, high=2048, size=1)[0])
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name

@pytest.fixture
def _test_sample_json():
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
    dict_obj={
                'class': "unit_tests",
                'name': "unit_tests",
                'provenance': "unit_tests",
                'location': object_translations.tolist(),
                'quaternion_xyzw': quaternion_xyzw.tolist(),
                'projected_cuboid': projected_cuboid.tolist(),
                'scale': object_scale.tolist(),
                'keypoints_3d': keypoints_3d.tolist(),
            }
    
    for idx in range(5):
        json_output['objects'].append(dict_obj)

    # Generate train files
    tmp_top_train_dir = os.path.join(tmp_top_dir, "train")
    check_and_create(tmp_top_train_dir)

    json_file = os.path.join(tmp_top_train_dir, "sample.json")
    img_file = os.path.join(tmp_top_train_dir, "sample.jpg")

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    sample_w = int(TEST_WIDTH)
    sample_h = int(TEST_HEIGHT)
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
    img.save(img_file)

    # Generate test files
    tmp_top_test_dir = os.path.join(tmp_top_dir, "test")
    check_and_create(tmp_top_test_dir)

    json_file = os.path.join(tmp_top_test_dir, "sample.json")
    img_file = os.path.join(tmp_top_test_dir, "sample.jpg")

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    sample_w = int(TEST_WIDTH)
    sample_h = int(TEST_HEIGHT)
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
    img.save(img_file)

    # Generate val files
    tmp_top_val_dir = os.path.join(tmp_top_dir, "val")
    check_and_create(tmp_top_val_dir)

    json_file = os.path.join(tmp_top_val_dir, "sample.json")
    img_file = os.path.join(tmp_top_val_dir, "sample.jpg")

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    sample_w = int(TEST_WIDTH)
    sample_h = int(TEST_HEIGHT)
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
    img.save(img_file)

    # Generate inference files
    tmp_top_infer_dir = os.path.join(tmp_top_dir, "infer")
    check_and_create(tmp_top_infer_dir)

    json_file = os.path.join(tmp_top_infer_dir, "sample.json")
    img_file = os.path.join(tmp_top_infer_dir, "sample.jpg")

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    sample_w = int(TEST_WIDTH)
    sample_h = int(TEST_HEIGHT)
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
    img.save(img_file)


@pytest.fixture
def _dataset_spec():
    data_config = OmegaConf.structured(CenterPoseDatasetConfig())
    data_config.train_data = os.path.join(tmp_top_dir, "train")
    data_config.test_data = os.path.join(tmp_top_dir, "test")
    data_config.val_data = os.path.join(tmp_top_dir, "val")
    data_config.inference_data = os.path.join(tmp_top_dir, "infer")
    data_config.batch_size = TEST_BATCH_SIZE
    data_config.workers = 0

    yield data_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("num_symmetry", [1, 4, 12])
def test_train_dataloader(_test_sample_json, _dataset_spec, num_symmetry):
    _dataset_spec.num_symmetry = num_symmetry
    dm = CPDataModule(_dataset_spec)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    for batch in train_loader:
        data = batch['input']
        b, c, height, width = data.shape
        assert b == TEST_BATCH_SIZE, "Incorrect batch size"
        assert c == 4, "Incorrect channel size"

        assert height == _dataset_spec.input_res, "Incorrect image height"
        assert width == _dataset_spec.input_res, "Incorrect image width"

        h, w = batch["hm"]
        assert h == _dataset_spec.output_res, "Incorrect label height"
        assert w == _dataset_spec.output_res, "Incorrect label width"

@pytest.mark.cv_unit
def test_pred_dataloader(_test_sample_json, _dataset_spec):
    dm = CPDataModule(_dataset_spec)
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    total_iter = len(pred_loader)
    total_size = len(dm.pred_dataset)
    for i, batch in enumerate(pred_loader):
        data = batch['input']
        b, _, height, width = data.shape
        if i + 1 == total_iter:
            assert b == (total_size % TEST_BATCH_SIZE), "Incorrect handling at the last batch"
        else:
            assert b == TEST_BATCH_SIZE, "Incorrect batch size"

        assert height == _dataset_spec.input_res, "Incorrect image height"
        assert width == _dataset_spec.input_res, "Incorrect image width"

    tmp_top_obj.cleanup()
