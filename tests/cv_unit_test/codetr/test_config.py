# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple tests for CoDETR config loading."""

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.codetr.model import CoDETRModelConfig


@pytest.mark.cv_unit
def test_codetr_experiment_config_defaults():
    """ExperimentConfig instantiates with sane CoDETR defaults."""
    cfg = ExperimentConfig()
    assert cfg.model_name == "codetr"
    assert isinstance(cfg.model, CoDETRModelConfig)
    assert cfg.model.num_co_heads >= 1


@pytest.mark.cv_unit
def test_codetr_model_config_defaults():
    """CoDETRModelConfig has the expected default values."""
    cfg = CoDETRModelConfig()
    assert cfg.hidden_dim == 256
    assert cfg.num_queries == 900
    assert cfg.num_feature_levels == 4
    assert len(cfg.return_interm_indices) == cfg.num_feature_levels


@pytest.mark.cv_unit
def test_codetr_config_yaml_merge():
    """Structured CoDETR config merges with a user-supplied override."""
    schema = OmegaConf.structured(ExperimentConfig())
    override = OmegaConf.create({
        "model": {
            "num_queries": 1500,
            "num_co_heads": 2,
            "num_feature_levels": 5,
            "return_interm_indices": [0, 1, 2, 3, 4],
        },
        "dataset": {"num_classes": 80},
    })
    merged = OmegaConf.merge(schema, override)
    assert merged.model.num_queries == 1500
    assert merged.model.num_co_heads == 2
    assert list(merged.model.return_interm_indices) == [0, 1, 2, 3, 4]
    assert merged.dataset.num_classes == 80
