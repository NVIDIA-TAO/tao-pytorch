# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Sparse4D dataset and distributed-training integration."""

from unittest.mock import patch

import numpy as np
import pytest

from nvidia_tao_pytorch.cv.sparse4d.dataloader import dataset as dataset_module
from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import (
    Omniverse3DDetTrackDataset,
)
from nvidia_tao_pytorch.cv.sparse4d.scripts.train import select_train_strategy


pytestmark = pytest.mark.cv_unit


def _camera(name):
    return {
        "data_path": f"{name}.jpg",
        "cam_intrinsic": np.eye(3),
        "sensor2world_transform": np.eye(4),
    }


def test_lazy_camera_sampling_reuses_info_for_camera_indexed_annotations():
    """Lazy image and annotation fields use one random camera subset."""
    full_info = {
        "token": "scene__000001",
        "timestamp": 1.0,
        "scene_name": "scene",
        "frame_idx": 1,
        "cams": {"cam0": _camera("cam0"), "cam1": _camera("cam1")},
        "gt_boxes": np.zeros((1, 7), dtype=np.float32),
        "gt_names": np.asarray(["person"]),
        "valid_flag": np.asarray([True]),
        "instance_inds": np.asarray([11]),
        "asset_inds": np.asarray([21]),
        "gt_visibility": [{"cam0": 0.25, "cam1": 0.75}],
    }
    dataset = object.__new__(Omniverse3DDetTrackDataset)
    dataset.lazy_load = True
    dataset._frame_index = [{"pkl_path": "scene.pkl", "local_idx": 0}]
    dataset._get_cached_pkl = lambda _: {"infos": [full_info]}
    dataset.data_infos = [{"group_idx": 3, "scene_idx": 4}]
    dataset.test_mode = False
    dataset.max_cameras = 1
    dataset.modality = {"use_camera": True}
    dataset.data_root = ""
    dataset.use_valid_flag = True
    dataset.with_velocity = False
    dataset.CLASSES = ["person"]

    with patch.object(
        dataset_module.random,
        "sample",
        side_effect=[["cam0"], ["cam1"]],
    ) as sample:
        output = dataset.get_data_info(0)

    assert sample.call_count == 1
    assert output["cam_names"] == ["cam0"]
    assert output["img_filename"] == ["cam0.jpg"]
    np.testing.assert_allclose(output["gt_visibility"], [[0.25]])
    assert output["group_idx"] == 3
    assert output["scene_idx"] == 4


@pytest.mark.parametrize(
    "devices,num_nodes,param_touch,expected",
    [
        ([0], 1, False, "auto"),
        ([0], 2, False, "ddp_find_unused_parameters_true"),
        ([0], 2, True, "ddp"),
        ([0, 1], 1, False, "ddp_find_unused_parameters_true"),
        (2, 1, True, "ddp"),
    ],
)
def test_strategy_uses_total_world_size(devices, num_nodes, param_touch, expected):
    """One GPU per node still selects DDP when multiple nodes participate."""
    assert select_train_strategy(devices, num_nodes, param_touch) == expected
