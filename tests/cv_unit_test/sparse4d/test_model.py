# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from platform import machine
import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig, Omniverse3DDetTrackDatasetConfig, Sparse4DModelConfig
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel

# Skip Sparse4D tests on ARM due to extremely long runtime
pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests take very long (~12 hours) on ARM architecture. TODO: Fix this.",
)

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(Omniverse3DDetTrackDatasetConfig())
    model_config = OmegaConf.structured(Sparse4DModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.model
def test_sparse4d_model_build(_test_experiment_spec):
    """Tests if the Sparse4DPlModel can be instantiated."""
    try:
        model = Sparse4DPlModel(_test_experiment_spec)
        assert model is not None, "Model instantiation failed."
    except Exception as e:
        pytest.fail(f"Sparse4DPlModel instantiation raised an exception: {e}")