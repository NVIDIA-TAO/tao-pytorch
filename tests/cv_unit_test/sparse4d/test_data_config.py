# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused schema tests for Sparse4D data-path and co-training controls."""

from omegaconf import OmegaConf
import pytest

from nvidia_tao_pytorch.config.sparse4d.dataset import (
    Omniverse3DDetTrackDatasetConfig,
)


pytestmark = [pytest.mark.cv_unit, pytest.mark.config]


def test_new_data_path_features_are_disabled_by_default():
    """Released behavior stays unchanged until sidecars/routes are configured."""
    config = Omniverse3DDetTrackDatasetConfig()

    assert config.lazy_load is False
    assert config.pkl_sample_size == 0
    assert config.ltt_2dgt_sidecar_dir == ""
    assert config.rtdetr_2d_cache_dir == ""
    assert config.rtdetr_2d_cache_path == ""
    assert config.resize_to_canonical_2d is False
    assert config.sync_route is False
    assert config.real_block_prob == -1.0
    assert config.scene_switch_iters == 0


def test_data_path_options_merge_into_structured_schema():
    """Hydra accepts every lazy, LTT, RT-DETR, SV2D, and route setting."""
    schema = OmegaConf.structured(Omniverse3DDetTrackDatasetConfig)
    overrides = OmegaConf.create(
        {
            "lazy_load": True,
            "lazy_load_cache_size": 12,
            "pkl_sample_size": 20,
            "pkl_cam_counts_path": "/data/camera_counts.pkl",
            "fps_drop_prob": 0.25,
            "target_fps_choices": [30, 15],
            "max_cameras": 8,
            "eval_dist_fcn": "both",
            "eval_hota": False,
            "ltt_2dgt_sidecar_dir": "/data/ltt",
            "ltt_2dgt_frame_regex": r"frame_(\d+)",
            "ltt_2dgt_cache_size": 3,
            "rtdetr_2d_cache_dir": "/data/rtdetr",
            "rtdetr_2d_score_thr": 0.4,
            "rtdetr_2d_per_class_score_thr": {"forklift": 0.25},
            "resize_to_canonical_2d": True,
            "canonical_2d_height": 720,
            "canonical_2d_width": 1280,
            "sync_route": True,
            "real_scene_keywords": ["Zanker", "BuildingK"],
            "real_block_prob": 0.4,
            "scene_switch_iters": 16,
        }
    )

    merged = OmegaConf.merge(schema, overrides)

    assert merged.lazy_load_cache_size == 12
    assert merged.ltt_2dgt_frame_regex == r"frame_(\d+)"
    assert merged.rtdetr_2d_per_class_score_thr.forklift == 0.25
    assert merged.canonical_2d_width == 1280
    assert merged.real_scene_keywords == ["Zanker", "BuildingK"]
    assert merged.scene_switch_iters == 16
