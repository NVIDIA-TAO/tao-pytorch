# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the VideoCLIP experiment config schema and shipped spec.

These are fast (CPU, no model/weights) and validate the config contract that
the train / evaluate / inference / export entrypoints depend on.
"""

import os

import pytest
from omegaconf import OmegaConf

import nvidia_tao_pytorch.multimodal.video_clip as _vc_pkg
from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPEvaluateConfig,
    VideoCLIPExperimentConfig,
    VideoCLIPInferenceConfig,
    VideoCLIPMetricsConfig,
    VideoCLIPRunConfig,
    VideoCLIPSearchConfig,
)

SPEC_PATH = os.path.join(
    os.path.dirname(_vc_pkg.__file__), "experiment_specs", "experiment_spec.yaml"
)


@pytest.mark.multimodal_unit
class TestVideoCLIPConfigSchema:
    """Schema-level invariants for the refactored config."""

    def test_model_name_defaults_to_video_clip(self):
        """model_name self-populates to 'video_clip' (not 'clip')."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert cfg.model_name == "video_clip"

    def test_resolution_defaults_match_internvideo2_l14(self):
        """image_size / export dims default to 224 (IV2CLIP), not 256."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert cfg.model.image_size == 224
        assert cfg.export.input_height == 224
        assert cfg.export.input_width == 224

    def test_dynamic_export_and_trt_profile_defaults(self):
        """ONNX is dynamic by default; TRT owns its runtime batch range."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert cfg.export.batch_size == -1
        assert cfg.gen_trt_engine.batch_size == -1
        assert cfg.gen_trt_engine.tensorrt.min_batch_size == 1
        assert cfg.gen_trt_engine.tensorrt.opt_batch_size == 8
        assert cfg.gen_trt_engine.tensorrt.max_batch_size == 16

    def test_evaluate_and_inference_share_run_config(self):
        """Both task configs extend the shared VideoCLIPRunConfig base."""
        assert issubclass(VideoCLIPEvaluateConfig, VideoCLIPRunConfig)
        assert issubclass(VideoCLIPInferenceConfig, VideoCLIPRunConfig)

    @pytest.mark.parametrize("task", ["evaluate", "inference", "export"])
    def test_checkpoint_is_required(self, task):
        """Every checkpoint-backed task exposes a mandatory schema field."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert OmegaConf.is_missing(cfg[task], "checkpoint")

    def test_dataset_has_metrics_not_evaluation(self):
        """dataset.evaluation was renamed to dataset.metrics."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert "metrics" in cfg.dataset
        assert "evaluation" not in cfg.dataset
        assert isinstance(
            OmegaConf.to_object(cfg.dataset.metrics), dict
        ) or cfg.dataset.metrics.mode == "retrieval"

    def test_search_is_nested_and_uses_enabled(self):
        """Search is a nested sub-config (enabled flag) on both tasks; the
        flat top_k/search_metric/normalize knobs are gone from inference."""
        cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
        assert cfg.inference.search.enabled is False
        assert cfg.evaluate.search.enabled is False
        for flat in ("top_k", "search_metric", "normalize"):
            assert flat not in cfg.inference

    def test_search_config_field_is_enabled_not_enable(self):
        """Boolean is standardized to 'enabled' (matches peft/regularization)."""
        s = OmegaConf.structured(VideoCLIPSearchConfig())
        assert "enabled" in s
        assert "enable" not in s

    def test_metrics_config_defaults(self):
        m = OmegaConf.structured(VideoCLIPMetricsConfig())
        assert m.mode == "retrieval"
        assert list(m.exclude_categories) == ["Normal", "Abnormal"]


@pytest.mark.multimodal_unit
class TestShippedSpecMergesAgainstSchema:
    """The shipped experiment_spec.yaml must merge cleanly onto the schema and
    exercise the train/evaluate/inference/export subtrees."""

    def _merged(self):
        schema = OmegaConf.structured(VideoCLIPExperimentConfig())
        spec = OmegaConf.load(SPEC_PATH)
        return OmegaConf.merge(schema, spec)

    def test_spec_merges_and_has_all_task_subtrees(self):
        merged = self._merged()
        for key in ("model", "dataset", "train", "evaluate", "inference", "export"):
            assert key in merged, f"missing task/config subtree: {key}"

    def test_num_frames_single_source_interpolation(self):
        """dataset/query num_frames interpolate from model.num_frames."""
        merged = self._merged()
        nf = merged.model.num_frames
        assert merged.dataset.train.video_text.num_frames == nf
        assert merged.dataset.val.video_text.num_frames == nf
        assert merged.dataset.inference.video_text.num_frames == nf
        assert merged.inference.query.num_frames == nf

    def test_yaml_anchor_shares_block_and_overrides_split(self):
        merged = self._merged()
        assert merged.dataset.train.video_text.split == "train"
        assert merged.dataset.val.video_text.split == "test"
        assert merged.dataset.inference.video_text.split == "test"
        # shared fields flow through the anchor
        assert merged.dataset.val.video_text.format == "auto"
        assert merged.dataset.inference.video_text.format == "auto"

    def test_inference_search_block_resolved(self):
        merged = self._merged()
        assert merged.inference.mode == "embeddings"
        assert merged.inference.search.top_k == 5
        assert merged.inference.search.search_metric == "cosine"
        assert merged.inference.search.normalize is True

    def test_metrics_block_resolved(self):
        merged = self._merged()
        assert merged.dataset.metrics.mode == "retrieval"

    def test_spec_drops_dead_val_datasets(self):
        """The empty dataset.val.datasets list is no longer shipped in the spec."""
        spec = OmegaConf.load(SPEC_PATH)
        assert "datasets" not in spec.dataset.val
