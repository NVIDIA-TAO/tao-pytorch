# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained integration tests for the Sparse4D data module."""

import pickle

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import (
    Sparse4DDataModule,
)


BATCH_SIZE = 1
NUM_CAMS = 4
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 48
NUM_FRAMES = 2


def _write_fixture_data(tmp_path):
    """Write two annotated frames and their RGB/depth arrays to HDF5."""
    h5_path = tmp_path / "sensor_data.h5"
    camera_names = [f"camera_{index}" for index in range(NUM_CAMS)]
    infos = []

    with h5py.File(h5_path, "w") as stream:
        for frame_idx in range(NUM_FRAMES):
            cameras = {}
            for camera_idx, camera_name in enumerate(camera_names):
                rgb_key = f"rgb/{frame_idx}/{camera_name}"
                depth_key = f"depth/{frame_idx}/{camera_name}"
                rgb = np.full(
                    (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                    20 * frame_idx + camera_idx,
                    dtype=np.uint8,
                )
                depth = np.full(
                    (IMAGE_HEIGHT, IMAGE_WIDTH),
                    1000 + 10 * camera_idx,
                    dtype=np.uint16,
                )
                stream.create_dataset(rgb_key, data=rgb)
                stream.create_dataset(depth_key, data=depth)
                cameras[camera_name] = {
                    "data_path": (str(h5_path), rgb_key),
                    "depth_map_path": (str(h5_path), depth_key),
                    "sensor2world_transform": np.eye(4, dtype=np.float32),
                    "cam_intrinsic": np.array(
                        [
                            [40.0, 0.0, IMAGE_WIDTH / 2],
                            [0.0, 40.0, IMAGE_HEIGHT / 2],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                }

            infos.append(
                {
                    "token": f"fixture_scene__{frame_idx}",
                    "timestamp": frame_idx * 100_000,
                    "scene_name": "fixture_scene",
                    "frame_idx": frame_idx,
                    "cams": cameras,
                    "gt_boxes": np.array(
                        [[1.0 + frame_idx, 2.0, 0.5, 0.8, 0.9, 1.7, 0.0]],
                        dtype=np.float32,
                    ),
                    "gt_names": np.array(["person"]),
                    "valid_flag": np.array([True]),
                    "num_lidar_pts": np.array([1], dtype=np.int64),
                    "gt_velocity": np.array([[0.1, 0.0]], dtype=np.float32),
                    "instance_inds": np.array([100 + frame_idx], dtype=np.int64),
                    "asset_inds": np.array([200 + frame_idx], dtype=np.int64),
                    "gt_visibility": [dict.fromkeys(camera_names, 1.0)],
                }
            )

    annotation_path = tmp_path / "annotations.pkl"
    with annotation_path.open("wb") as stream:
        pickle.dump(
            {"infos": infos, "metadata": {"version": "test-fixture"}},
            stream,
        )
    return annotation_path


@pytest.fixture
def test_exp_spec(tmp_path):
    """Return a small experiment config backed only by temporary files."""
    annotation_path = _write_fixture_data(tmp_path)
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model.input_shape = [IMAGE_WIDTH, IMAGE_HEIGHT]
    experiment_config.dataset.classes = ["person"]
    experiment_config.dataset.use_h5_file_for_rgb = True
    experiment_config.dataset.use_h5_file_for_depth = True
    experiment_config.dataset.data_root = str(tmp_path)
    experiment_config.dataset.train_dataset.ann_file = str(annotation_path)
    experiment_config.dataset.val_dataset.ann_file = str(annotation_path)
    experiment_config.dataset.test_dataset.ann_file = str(annotation_path)
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.dataset.num_workers = 0
    experiment_config.dataset.sequences.split_num = NUM_FRAMES
    experiment_config.dataset.augmentation.image_size = [IMAGE_HEIGHT, IMAGE_WIDTH]
    experiment_config.dataset.augmentation.final_dim = [IMAGE_HEIGHT, IMAGE_WIDTH]
    experiment_config.dataset.augmentation.resize_lim = [1.0, 1.0]
    experiment_config.dataset.augmentation.bot_pct_lim = [0.0, 0.0]
    experiment_config.dataset.augmentation.rot_lim = [0.0, 0.0]
    experiment_config.dataset.augmentation.rot3d_range = [0.0, 0.0]
    experiment_config.dataset.augmentation.rand_flip = False
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    return experiment_config


@pytest.mark.parametrize(
    ("setup_stage", "loader_name", "has_ground_truth"),
    [
        pytest.param("fit", "train_dataloader", True, id="train"),
        pytest.param("fit", "val_dataloader", False, id="validate"),
        pytest.param("test", "test_dataloader", False, id="test"),
        pytest.param("predict", "predict_dataloader", False, id="predict"),
    ],
)
@pytest.mark.cv_unit
def test_build_dataloader(
    test_exp_spec,
    setup_stage,
    loader_name,
    has_ground_truth,
):
    """Build and iterate each public dataloader using real transforms."""
    dm = Sparse4DDataModule(test_exp_spec)
    dm.setup(setup_stage)
    loader = getattr(dm, loader_name)()

    count = 0
    for batch in loader:
        img = batch["img"]
        assert tuple(img.shape) == (
            BATCH_SIZE,
            NUM_CAMS,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        assert tuple(batch["projection_mat"].shape) == (
            BATCH_SIZE,
            NUM_CAMS,
            4,
            4,
        )
        assert tuple(batch["image_wh"].shape) == (
            BATCH_SIZE,
            NUM_CAMS,
            2,
        )

        if has_ground_truth:
            assert len(batch["gt_bboxes_3d"]) == BATCH_SIZE
            assert tuple(batch["gt_bboxes_3d"][0].shape) == (1, 9)
            assert batch["has_3d_gt"].tolist() == [True]
            assert tuple(batch["gt_depth"][0].shape) == (
                BATCH_SIZE,
                NUM_CAMS,
                IMAGE_HEIGHT // 4,
                IMAGE_WIDTH // 4,
            )
        else:
            assert "gt_bboxes_3d" not in batch

        count += 1
        if count == NUM_FRAMES:
            break

    assert count == NUM_FRAMES
