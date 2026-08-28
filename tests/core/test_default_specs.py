# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for default experiment specification generation."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utils import default_specs


@dataclass
class _ExperimentConfig:
    """Minimal experiment configuration for generation tests."""

    value: int = 1


@pytest.fixture
def _mock_config_module(monkeypatch):
    """Replace module discovery and import with a small local schema."""
    module = SimpleNamespace(
        __name__="nvidia_tao_pytorch.config.test.default_config",
        ExperimentConfig=_ExperimentConfig,
    )
    monkeypatch.setattr(default_specs, "get_supported_modules", lambda: ["test"])
    monkeypatch.setattr(default_specs, "import_module_from_path", lambda _: module)


def test_default_specs_prints_yaml_without_results_dir(_mock_config_module, capsys):
    """Omitting results_dir prints the generated specification to stdout."""
    default_specs.main(SimpleNamespace(module_name="test", results_dir=None))

    output = capsys.readouterr().out
    assert OmegaConf.create(output) == {"value": 1}


def test_default_specs_writes_yaml_with_results_dir(_mock_config_module, tmp_path):
    """An explicit results_dir preserves experiment.yaml file output."""
    output_dir = tmp_path / "specs"
    default_specs.main(SimpleNamespace(
        module_name="test",
        results_dir=str(output_dir),
    ))

    output_path = output_dir / "experiment.yaml"
    assert OmegaConf.load(output_path) == {"value": 1}
