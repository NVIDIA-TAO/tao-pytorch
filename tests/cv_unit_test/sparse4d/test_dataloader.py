# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from platform import machine
import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests take very long (~12 hours) on ARM architecture. TODO: Fix this.",
)

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule

DATA_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/full_data/"
TRAIN_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_train_split/SURF_Booth_031325+bev-sensor-buffer-zone-c4_infos_train.pkl"
VAL_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_infos_val.pkl"
TEST_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_test_split/SURF_Booth_031325+bev-sensor-buffer-zone-c4_infos_test.pkl"
ANCHOR_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/_ov_kmeans900_sample100_.npy"
CHECKPOINT_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/sparse4d_tracking_aic25v0.3_moving_classes_iter_60900_v1.1.pth"
BATCH_SIZE = 1
NUM_CAMS = 4


@pytest.fixture
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model.head.instance_bank.anchor = ANCHOR_PATH
    experiment_config.model.head.deformable_model.use_camera_embed = True
    experiment_config.dataset.classes = ['person', 'gr1_t2', 'agility_digit', 'nova_carter', 'transporter', 'forklift', 'pallet']
    experiment_config.dataset.use_h5_file_for_rgb = True
    experiment_config.dataset.use_h5_file_for_depth = True
    experiment_config.dataset.train_dataset.ann_file = TRAIN_ANNO_ROOT
    experiment_config.dataset.data_root = DATA_ROOT
    experiment_config.dataset.val_dataset.ann_file = VAL_ANNO_ROOT
    experiment_config.dataset.test_dataset.ann_file = TEST_ANNO_ROOT

    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.dataset.num_workers = 0
    yield experiment_config


@pytest.mark.parametrize("stage", ['fit', 'test', 'predict'], "")
@pytest.mark.cv_unit
def test_build_dataloader(_test_exp_spec, stage):

    dm = Sparse4DDataModule(_test_exp_spec)
    dm.setup(stage)
    if stage == 'fit':
        loader = dm.train_dataloader()
    elif stage == 'test':
        loader = dm.val_dataloader()
    elif stage == 'predict':
        loader = dm.predict_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")

    img_h = _test_exp_spec.model.input_shape[1]
    img_w = _test_exp_spec.model.input_shape[0]
    _test_exp_spec.visualize.show = False
    _test_exp_spec.inference.output_nvschema = False

    count = 0
    for batch in loader:
        if isinstance(batch, dict) and 'img' in batch:
            img = batch['img']
            assert img.shape[0] == BATCH_SIZE, f"Incorrect image batch size. Expected {BATCH_SIZE}, got {img.shape[0]}"
            assert img.shape[1] == NUM_CAMS, f"Incorrect no. of sensors. Expected {NUM_CAMS}, got {img.shape[1]}"
            assert img.shape[2] == 3, f"Incorrect image channels. Expected {3}, got {img.shape[2]}"
            assert img.shape[3] == img_h, f"Incorrect image height. Expected {img_h}, got {img.shape[3]}"
            assert img.shape[4] == img_w, f"Incorrect image width. Expected {img_w}, got {img.shape[4]}"

            if stage == 'fit':
                print(batch.keys())
                assert 'gt_bboxes_3d' in batch, "gt_boxes should be present in the batch for fit stage"
                assert 'timestamp' in batch, "timestamp should be present in the batch for fit stage"
                assert 'img' in batch, "img should be present in the batch for fit stage"
            else:
                assert 'gt_bboxes_3d' not in batch, "gt_boxes should not be present in the batch for test stage"
        else:
             print(f"Unexpected batch format: {type(batch)}")

        count += 1
        if count >= 2:
            break
