# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-compose tests for the shared evaluation-suite schema."""

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.ssl_evaluation.default_config import (
    EvalSuiteConfig,
    KNNEvalConfig,
)


@pytest.mark.config
def test_knn_protocol_defaults():
    """KNN config defaults preserve the protocol (k=20, enabled, ImageNet-normalized)."""
    cfg = OmegaConf.structured(KNNEvalConfig)
    assert cfg.k == 20
    assert cfg.enabled is True
    assert cfg.imagenet_normalize is True
    assert cfg.dataset_type == "image_folder"


@pytest.mark.config
def test_eval_suite_blocks_present_and_default_off():
    """EvalSuiteConfig exposes knn/segmentation/retrieval; seg+retrieval default disabled."""
    cfg = OmegaConf.structured(EvalSuiteConfig)
    assert cfg.knn.enabled is True
    assert cfg.segmentation.enabled is False
    assert cfg.retrieval.enabled is False
    assert cfg.cache_dir is None


@pytest.mark.config
@pytest.mark.schema_validation
def test_nvdinov2_experiment_config_has_evaluate_suite():
    """nvdinov2 ExperimentConfig composes the evaluate suite (knn k=20)."""
    from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig

    cfg = OmegaConf.structured(ExperimentConfig)
    assert cfg.evaluate.knn.k == 20
    assert cfg.evaluate.segmentation.enabled is False
    # override composes cleanly
    merged = OmegaConf.merge(cfg, OmegaConf.create({"evaluate": {"knn": {"k": 5}}}))
    assert merged.evaluate.knn.k == 5


@pytest.mark.config
@pytest.mark.schema_validation
def test_mae_experiment_config_has_evaluate_suite():
    """MAE ExperimentConfig mixes in the evaluate suite alongside its EvaluateConfig fields."""
    from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig

    cfg = OmegaConf.structured(ExperimentConfig)
    assert cfg.evaluate.knn.k == 20
    assert cfg.evaluate.retrieval.enabled is False
    assert cfg.evaluate.num_gpus == 1              # from common EvaluateConfig (mixin works)
