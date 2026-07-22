# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple tests for CoDETR model building."""

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.codetr.model.build_nn_model import CoDETRModel, build_model


@pytest.fixture
def _experiment_spec():
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.dataset.num_classes = 4
    cfg.model.backbone = "resnet_50"
    cfg.model.num_feature_levels = 2
    cfg.model.return_interm_indices = [1, 2]
    cfg.model.num_queries = 100
    cfg.model.num_co_heads = 1
    cfg.model.co_head_num_convs = 1
    yield cfg


@pytest.mark.cv_unit
def test_build_codetr_train_mode(_experiment_spec):
    """Train-mode build attaches collaborative heads and downsample module."""
    model = build_model(_experiment_spec, export=False)
    assert isinstance(model, CoDETRModel)
    assert model.export is False
    assert model.collab_heads is not None
    assert len(model.collab_heads) == _experiment_spec.model.num_co_heads
    assert model.downsample is not None


@pytest.mark.cv_unit
def test_build_codetr_export_mode(_experiment_spec):
    """Export-mode build skips the auxiliary collab heads."""
    _experiment_spec.model.aux_loss = False
    model = build_model(_experiment_spec, export=True)
    assert model.export is True
    assert model.collab_heads is None


@pytest.mark.cv_unit
@pytest.mark.parametrize("num_co_heads", [1, 2])
def test_build_codetr_num_co_heads(_experiment_spec, num_co_heads):
    """num_co_heads controls how many ATSS heads are instantiated."""
    _experiment_spec.model.num_co_heads = num_co_heads
    model = build_model(_experiment_spec, export=False)
    assert len(model.collab_heads) == num_co_heads
