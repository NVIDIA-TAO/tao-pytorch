# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Sparse4D co-training dataset metadata and collation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.cv.sparse4d.dataloader.callbacks import (
    PklResampleCallback,
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import (
    Omniverse3DDetTrackDataset,
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import (
    collate_fn,
)


pytestmark = pytest.mark.cv_unit


def test_dataset_constructor_preserves_annotation_path_alias(monkeypatch):
    """Annotation loading and sampler sidecars use the historical ann_file name."""
    observed_paths = []

    def fake_load_annotations(self, path):
        observed_paths.append(path)
        return []

    monkeypatch.setattr(
        Omniverse3DDetTrackDataset,
        "load_annotations",
        fake_load_annotations,
    )
    dataset = Omniverse3DDetTrackDataset(
        data_root="",
        anno_file="/tmp/sparse4d_train.txt",
        classes=["person"],
    )

    assert observed_paths == ["/tmp/sparse4d_train.txt"]
    assert dataset.ann_file == dataset.anno_file


def _dataset_without_initialization(info):
    dataset = object.__new__(Omniverse3DDetTrackDataset)
    dataset.lazy_load = False
    dataset.data_infos = [info]
    dataset.modality = {"use_camera": True}
    dataset.data_root = "/data"
    dataset.test_mode = False
    dataset.use_valid_flag = True
    dataset.with_velocity = True
    dataset.CLASSES = ["person"]
    return dataset


def test_no_3d_gt_sample_preserves_camera_and_route_metadata():
    """Real frames produce empty targets and an explicit 2D-only route marker."""
    camera_transform = np.arange(16, dtype=np.float32).reshape(4, 4)
    camera_transform[3] = [0, 0, 0, 1]
    info = {
        "token": "RealScene__000000007",
        "timestamp": 7,
        "scene_name": "RealScene",
        "frame_idx": 7,
        "group_idx": 2,
        "scene_idx": 3,
        "cams": {
            "cam0": {
                "data_path": "/data/frame_000007.jpg",
                "cam_intrinsic": np.eye(3, dtype=np.float32),
                "sensor2world_transform": camera_transform,
            }
        },
        "gt_boxes": None,
    }
    sample = _dataset_without_initialization(info).get_data_info(0)

    assert sample["has_3d_gt"] is False
    assert sample["gt_bboxes_3d"].shape == (0, 9)
    assert sample["instance_inds"].shape == (0,)
    assert sample["asset_inds"].shape == (0,)
    assert sample["frame_idx"] == 7
    assert sample["group_idx"] == 2
    assert sample["scene_idx"] == 3
    np.testing.assert_array_equal(sample["cam2world_transform"][0], camera_transform)


def test_collate_preserves_auxiliary_targets_and_stacks_extrinsics():
    """Variable GT/detection arrays remain per sample while matrices stack."""
    samples = []
    for has_3d_gt in (True, False):
        sample = {
            "img": torch.zeros((1, 3, 4, 4)),
            "projection_mat": np.eye(4, dtype=np.float32)[None],
            "cam2world_transform": np.eye(4, dtype=np.float32)[None],
            "image_wh": np.array([[4, 4]], dtype=np.float32),
            "gt_boxes_2d_visible": torch.zeros((0, 1, 4)),
            "gt_occ_weight": torch.zeros((0, 1)),
            "det_boxes_2d": [torch.zeros((0, 4))],
            "det_classes_2d": [torch.zeros((0,), dtype=torch.long)],
            "det_scores_2d": [torch.zeros((0,))],
            "has_3d_gt": has_3d_gt,
            "scene_name": "Synthetic" if has_3d_gt else "Real",
        }
        if not has_3d_gt:
            sample["has_2d_pseudo"] = False
        samples.append(sample)

    batch = collate_fn(samples)

    assert batch["img"].shape == (2, 1, 3, 4, 4)
    assert batch["cam2world_transform"].shape == (2, 1, 4, 4)
    assert len(batch["gt_boxes_2d_visible"]) == 2
    assert len(batch["det_boxes_2d"]) == 2
    torch.testing.assert_close(batch["has_3d_gt"], torch.tensor([True, False]))
    torch.testing.assert_close(batch["has_2d_pseudo"], torch.tensor([True, False]))


def test_pkl_resample_callback_updates_before_the_next_epoch():
    """Epoch-end resampling refreshes dataset and sampler state."""
    calls = []

    class Dataset:
        def resample_pkls(self, epoch):
            calls.append(("dataset", epoch))

    class Sampler:
        def update_from_dataset(self):
            calls.append(("sampler", None))

    trainer = SimpleNamespace(
        current_epoch=2,
        global_step=30,
        train_dataloader=SimpleNamespace(
            dataset=Dataset(),
            batch_sampler=Sampler(),
        ),
    )

    PklResampleCallback(num_iters_per_epoch=10).on_train_epoch_end(
        trainer,
        pl_module=None,
    )

    assert calls == [("dataset", 3), ("sampler", None)]
