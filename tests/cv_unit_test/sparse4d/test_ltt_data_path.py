# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Sparse4D LTT, RT-DETR, and SV2D data transforms."""

import json
import warnings

import h5py
import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.cv.sparse4d.dataloader.augment import (
    BBoxRotation,
    ResizeCropFlipImage,
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.transforms import (
    AICitySparse4DAdaptor,
    InstanceNameFilter,
    LoadDepthMap,
    LoadLooseToTight2DGT,
    LoadMultiViewImageFromFiles,
    LoadRTDETR2D,
    ResizeToCanonical2D,
)


pytestmark = pytest.mark.cv_unit


def _encoded_metadata(**metadata):
    return np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)


def test_multiview_h5_loader_normalizes_grayscale_and_rgba(tmp_path):
    """SV2D images always enter downstream OpenCV transforms as HWC3."""
    h5_path = tmp_path / "images.h5"
    with h5py.File(h5_path, "w") as stream:
        stream.create_dataset("gray", data=np.zeros((4, 6), dtype=np.uint8))
        stream.create_dataset("rgba", data=np.zeros((4, 6, 4), dtype=np.uint8))

    output = LoadMultiViewImageFromFiles(h5_file=True)(
        {"img_filename": [(str(h5_path), "gray"), (str(h5_path), "rgba")]}
    )

    assert [image.shape for image in output["img"]] == [
        (4, 6, 3),
        (4, 6, 3),
    ]


def test_depth_loader_preserves_missing_sentinel_and_valid_depth(tmp_path):
    """Missing depth stays invalid while valid millimetres convert and clip."""
    h5_path = tmp_path / "depth.h5"
    with h5py.File(h5_path, "w") as stream:
        stream.create_dataset(
            "depth",
            data=np.array([[0, 50, 100], [1000, 50000, 60000]], dtype=np.uint16),
        )

    output = LoadDepthMap(max_depth=50, default_shape=(2, 3), h5_file=True,)(
        {
            "depth_map_filename": [None, (str(h5_path), "depth")],
            "lidar2img": [np.eye(4), np.eye(4)],
        }
    )

    np.testing.assert_array_equal(output["gt_depth"][0], -np.ones((2, 3)))
    np.testing.assert_allclose(
        output["gt_depth"][1],
        [[-1.0, 0.1, 0.1], [1.0, 50.0, 50.0]],
    )


@pytest.mark.parametrize(
    "scene_name", ["CT1_distill__SceneA", "CT1_distill__SceneA+training_group_0"]
)
def test_load_loose_to_tight_sidecar_aligns_instances_and_cameras(
    tmp_path, scene_name
):
    """LTT sidecars join prefixed/grouped scenes to their raw scene artifact."""
    np.savez(
        tmp_path / "SceneA__ltt2dgt.npz",
        _meta=_encoded_metadata(cam_names=["cam0", "cam1"]),
        frame_id=np.array([42, 42]),
        instance_id=np.array([7, 8]),
        cam=np.array([0, 1]),
        box3=np.array([[1, 2, 11, 12], [3, 4, 13, 14]], np.float32),
        occ=np.array([0.25, 0.75], np.float32),
    )
    transform = LoadLooseToTight2DGT(tmp_path)
    results = transform(
        {
            "scene_name": scene_name,
            "cam_names": ["cam0", "cam1"],
            "img_filename": ["frame_000042.jpg", "frame_000042.jpg"],
            "instance_inds": np.array([8, 7]),
        }
    )

    assert results["gt_boxes_2d_visible"].shape == (2, 2, 4)
    np.testing.assert_array_equal(results["gt_boxes_2d_visible"][0, 1], [3, 4, 13, 14])
    np.testing.assert_array_equal(results["gt_boxes_2d_visible"][1, 0], [1, 2, 11, 12])
    np.testing.assert_allclose(results["gt_occ_weight"], [[0, 0.75], [0.25, 0]])


def test_load_rtdetr_filters_scores_and_marks_cached_scene_real(tmp_path):
    """RT-DETR caches preserve camera order and apply class thresholds."""
    np.savez(
        tmp_path / "RealScene__rtdetr2d.npz",
        _meta=_encoded_metadata(
            cam_names=["cam0", "cam1"],
            class_names=["person", "forklift"],
        ),
        frame_id=np.array([42, 42, 42]),
        cam=np.array([0, 1, 0]),
        class_id=np.array([0, 1, 1]),
        box=np.array(
            [
                [1, 2, 10, 20],
                [3, 4, 30, 40],
                [5, 6, 50, 60],
            ],
            np.float32,
        ),
        score=np.array([0.9, 0.4, 0.2], np.float32),
    )
    transform = LoadRTDETR2D(
        cache_dir=tmp_path,
        score_thr=0.5,
        per_class_score_thr={"forklift": 0.3},
    )
    results = transform(
        {
            "scene_name": "RealScene+bev-c4",
            "sample_idx": "RealScene__000000042",
            "cam_names": ["cam1", "cam0"],
            "has_3d_gt": True,
        }
    )

    assert results["has_3d_gt"] is False
    assert results["has_2d_pseudo"] is True
    np.testing.assert_array_equal(results["det_classes_2d"][0], [1])
    np.testing.assert_array_equal(results["det_classes_2d"][1], [0])
    np.testing.assert_allclose(results["det_scores_2d"][0], [0.4])


def test_load_rtdetr_rejects_cache_taxonomy_mismatch(tmp_path):
    """Pseudo caches cannot silently relabel detector classes by position."""
    np.savez(
        tmp_path / "RealScene__rtdetr2d.npz",
        _meta=_encoded_metadata(
            cam_names=["cam0"],
            class_names=["forklift", "person"],
        ),
        frame_id=np.array([42]),
        cam=np.array([0]),
        class_id=np.array([0]),
        box=np.array([[1, 2, 10, 20]], np.float32),
        score=np.array([0.9], np.float32),
    )
    transform = LoadRTDETR2D(
        cache_dir=tmp_path,
        class_names=["person", "forklift"],
    )

    with pytest.raises(ValueError, match="class_names/order"):
        transform(
            {
                "scene_name": "RealScene",
                "frame_idx": 42,
                "cam_names": ["cam0"],
            }
        )


def test_load_rtdetr_rejects_out_of_range_class_ids(tmp_path):
    """Malformed pseudo classes fail at cache load instead of being dropped."""
    np.savez(
        tmp_path / "RealScene__rtdetr2d.npz",
        _meta=_encoded_metadata(cam_names=["cam0"], class_names=["person"]),
        frame_id=np.array([42]),
        cam=np.array([0]),
        class_id=np.array([1]),
        box=np.array([[1, 2, 10, 20]], np.float32),
        score=np.array([0.9], np.float32),
    )

    with pytest.raises(ValueError, match="outside"):
        LoadRTDETR2D(cache_dir=tmp_path)(
            {
                "scene_name": "RealScene",
                "frame_idx": 42,
                "cam_names": ["cam0"],
            }
        )


def test_load_rtdetr_marks_explicit_empty_frame_as_valid(tmp_path):
    """Processed empty frames retain background-only supervision semantics."""
    np.savez(
        tmp_path / "EmptyScene__rtdetr2d.npz",
        _meta=_encoded_metadata(cam_names=["cam0", "cam1"], class_names=["person"]),
        frame_id=np.zeros(0, dtype=np.int32),
        cam=np.zeros(0, dtype=np.int16),
        class_id=np.zeros(0, dtype=np.int16),
        box=np.zeros((0, 4), dtype=np.float32),
        score=np.zeros(0, dtype=np.float32),
        valid_frame_id=np.array([42, 42], dtype=np.int32),
        valid_cam=np.array([0, 1], dtype=np.int16),
    )

    results = LoadRTDETR2D(cache_dir=tmp_path)(
        {
            "scene_name": "EmptyScene",
            "frame_idx": 42,
            "cam_names": ["cam0", "cam1"],
        }
    )

    assert results["has_2d_pseudo"] is True
    assert all(boxes.shape == (0, 4) for boxes in results["det_boxes_2d"])


def test_load_rtdetr_requires_cache_canonical_image_shape(tmp_path):
    """SV2D cache coordinates cannot silently diverge from resized images."""
    np.savez(
        tmp_path / "SV2D__rtdetr2d.npz",
        _meta=_encoded_metadata(
            cam_names=["cam0"],
            class_names=["person"],
            virtual_camera={"width": 12, "height": 8},
        ),
        frame_id=np.array([0], dtype=np.int32),
        cam=np.array([0], dtype=np.int16),
        class_id=np.array([0], dtype=np.int16),
        box=np.array([[1, 2, 10, 7]], dtype=np.float32),
        score=np.array([1.0], dtype=np.float32),
        valid_frame_id=np.array([0], dtype=np.int32),
        valid_cam=np.array([0], dtype=np.int16),
    )
    transform = LoadRTDETR2D(
        cache_dir=tmp_path,
        class_names=["person"],
    )
    sample = {
        "scene_name": "SV2D",
        "frame_idx": 0,
        "cam_names": ["cam0"],
        "has_3d_gt": False,
        "img": [np.zeros((4, 6, 3), dtype=np.float32)],
    }

    with pytest.raises(ValueError, match="resize_to_canonical_2d"):
        transform(dict(sample))

    resized = ResizeToCanonical2D(height=8, width=12)(sample)
    output = transform(resized)
    assert output["has_2d_pseudo"] is True
    np.testing.assert_array_equal(output["det_boxes_2d"][0], [[1, 2, 10, 7]])


def test_load_rtdetr_missing_cache_warns_once_and_marks_invalid(tmp_path):
    """A missing sidecar cannot masquerade as a background-only frame."""
    transform = LoadRTDETR2D(cache_dir=tmp_path)
    sample = {
        "scene_name": "MissingScene",
        "frame_idx": 42,
        "cam_names": ["cam0"],
    }

    with pytest.warns(RuntimeWarning, match="found no cache"):
        first = transform(dict(sample))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = transform(dict(sample))

    assert first["has_2d_pseudo"] is False
    assert second["has_2d_pseudo"] is False
    assert not caught


def test_load_rtdetr_missing_frame_join_warns_and_marks_invalid(tmp_path):
    """An old cache only validates frame/camera pairs represented by raw rows."""
    np.savez(
        tmp_path / "LegacyScene__rtdetr2d.npz",
        _meta=_encoded_metadata(cam_names=["cam0"], class_names=["person"]),
        frame_id=np.array([41], dtype=np.int32),
        cam=np.array([0], dtype=np.int16),
        class_id=np.array([0], dtype=np.int16),
        box=np.array([[1, 2, 10, 20]], dtype=np.float32),
        score=np.array([0.9], dtype=np.float32),
    )

    with pytest.warns(RuntimeWarning, match="no valid cache join"):
        results = LoadRTDETR2D(cache_dir=tmp_path)(
            {
                "scene_name": "LegacyScene",
                "frame_idx": 42,
                "cam_names": ["cam0"],
            }
        )

    assert results["has_2d_pseudo"] is False
    assert results["det_boxes_2d"][0].shape == (0, 4)


def test_image_augmentation_transforms_ltt_and_detection_boxes():
    """The image homography is applied to both fixed and variable 2D boxes."""
    results = {
        "img": [np.zeros((100, 200, 3), dtype=np.float32)],
        "lidar2img": [np.eye(4)],
        "cam_intrinsic": [np.eye(3)],
        "gt_boxes_2d_visible": np.array([[[20, 10, 60, 30]]], np.float32),
        "det_boxes_2d": [np.array([[20, 10, 60, 30]], np.float32)],
        "aug_config": {
            "resize": 0.5,
            "resize_dims": (100, 50),
            "crop": (0, 0, 100, 50),
            "flip": True,
            "rotate": 0,
        },
    }

    output = ResizeCropFlipImage()(results)

    expected = np.array([70, 5, 90, 15], np.float32)
    np.testing.assert_allclose(output["gt_boxes_2d_visible"][0, 0], expected)
    np.testing.assert_allclose(output["det_boxes_2d"][0][0], expected)
    assert output["img"][0].shape == (50, 100, 3)


def test_bbox_rotation_keeps_camera_extrinsics_consistent():
    """Image projection and camera extrinsics use the same rotated ego frame."""
    angle = np.pi / 2
    results = {
        "aug_config": {"rotate_3d": angle},
        "lidar2img": [np.eye(4)],
        "cam2world_transform": [np.eye(4)],
        "gt_bboxes_3d": np.zeros((0, 9), dtype=np.float32),
    }
    output = BBoxRotation()(results)
    rotation = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    expected = np.linalg.inv(rotation)
    np.testing.assert_allclose(output["lidar2img"][0], expected, atol=1e-7)
    np.testing.assert_allclose(output["cam2world_transform"][0], expected, atol=1e-7)


def test_canonical_resize_and_adaptor_preserve_auxiliary_targets():
    """SV2D image shapes and auxiliary data survive final adaptation."""
    resized = ResizeToCanonical2D(height=8, width=12)(
        {
            "img": [np.zeros((4, 6, 3), dtype=np.float32)],
            "has_3d_gt": False,
        }
    )
    assert resized["img"][0].shape == (8, 12, 3)
    assert resized["img_shape"] == [(8, 12, 3)]

    resized.update(
        {
            "lidar2img": [np.eye(4)],
            "cam2world_transform": [np.eye(4)],
            "cam_intrinsic": [np.eye(3)],
            "instance_inds": np.array([5]),
            "asset_inds": np.array([6]),
            "gt_bboxes_3d": np.zeros((1, 9), dtype=np.float32),
            "gt_labels_3d": np.array([0]),
            "gt_boxes_2d_visible": np.zeros((1, 1, 4), dtype=np.float32),
            "gt_occ_weight": np.ones((1, 1), dtype=np.float32),
            "det_boxes_2d": [np.zeros((0, 4), dtype=np.float32)],
            "det_classes_2d": [np.zeros((0,), dtype=np.int64)],
            "det_scores_2d": [np.zeros((0,), dtype=np.float32)],
        }
    )
    output = AICitySparse4DAdaptor()(resized)

    assert output["img"].shape == (1, 3, 8, 12)
    assert output["cam2world_transform"].shape == (1, 4, 4)
    assert output["gt_boxes_2d_visible"].dtype == torch.float32
    assert output["det_classes_2d"][0].dtype == torch.long


def test_canonical_resize_only_changes_2d_member_of_mixed_routes():
    """Noncanonical calibrated images and their projection geometry stay paired."""
    transform = ResizeToCanonical2D(height=8, width=12)
    projection = np.arange(16, dtype=np.float32).reshape(4, 4)
    intrinsic = np.arange(9, dtype=np.float32).reshape(3, 3)
    calibrated = {
        "img": [np.zeros((5, 7, 3), dtype=np.float32)],
        "has_3d_gt": True,
        "lidar2img": [projection.copy()],
        "cam_intrinsic": [intrinsic.copy()],
    }
    calibration_free = {
        "img": [np.zeros((4, 6, 3), dtype=np.float32)],
        "has_3d_gt": False,
    }

    transformed_3d = transform(calibrated)
    transformed_2d = transform(calibration_free)

    assert transformed_3d["img"][0].shape == (5, 7, 3)
    np.testing.assert_array_equal(transformed_3d["lidar2img"][0], projection)
    np.testing.assert_array_equal(transformed_3d["cam_intrinsic"][0], intrinsic)
    assert transformed_2d["img"][0].shape == (8, 12, 3)


def test_instance_filter_keeps_ltt_rows_aligned():
    """Class filtering applies the same mask to every instance-level target."""
    output = InstanceNameFilter(["person"])(
        {
            "gt_labels_3d": np.array([0, -1]),
            "gt_bboxes_3d": np.zeros((2, 9), dtype=np.float32),
            "instance_inds": np.array([10, 20]),
            "asset_inds": np.array([30, 40]),
            "gt_boxes_2d_visible": np.arange(8).reshape(2, 1, 4),
            "gt_occ_weight": np.array([[0.5], [0.75]]),
        }
    )

    np.testing.assert_array_equal(output["instance_inds"], [10])
    np.testing.assert_array_equal(output["asset_inds"], [30])
    np.testing.assert_array_equal(
        output["gt_boxes_2d_visible"], np.arange(4).reshape(1, 1, 4)
    )
    np.testing.assert_allclose(output["gt_occ_weight"], [[0.5]])
