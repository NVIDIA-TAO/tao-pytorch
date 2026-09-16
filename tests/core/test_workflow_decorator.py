# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the status decorator's experiment-spec publication policy."""

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

import nvidia_tao_pytorch.core.loggers.api_logging as api_logging  # noqa: E402
from nvidia_tao_pytorch.core.decorators import workflow  # noqa: E402
from nvidia_tao_pytorch.core.tlt_logging import logger  # noqa: E402


@pytest.fixture(autouse=True)
def restore_global_logging_state():
    """Undo the process-global logger mutations made by ``monitor_status``."""
    prior_handlers = logger.handlers.copy()
    prior_status_logger = api_logging.get_status_logger()
    try:
        yield
    finally:
        logger.handlers = prior_handlers
        api_logging._STATUS_LOGGER = prior_status_logger  # pylint: disable=W0212


@pytest.mark.parametrize("write_experiment_spec", [True, False])
def test_experiment_spec_write_is_explicit_and_defaults_on(
    tmp_path, monkeypatch, write_experiment_spec
):
    """Existing callers write as before; an opted-out caller owns publication."""
    monkeypatch.setattr(workflow, "update_results_dir", lambda cfg, **_: cfg)
    config = OmegaConf.create({"results_dir": str(tmp_path)})

    @workflow.monitor_status(
        name="status-test",
        mode="train",
        write_experiment_spec=write_experiment_spec,
    )
    def run(_cfg):
        return None

    run(config)
    assert (tmp_path / "experiment.yaml").exists() is write_experiment_spec


def test_existing_caller_uses_default_writer(tmp_path, monkeypatch):
    """Omitting the new keyword retains the historical writer."""
    monkeypatch.setattr(workflow, "update_results_dir", lambda cfg, **_: cfg)
    config = OmegaConf.create({"results_dir": str(tmp_path), "value": "legacy"})

    @workflow.monitor_status(name="legacy", mode="train")
    def run(_cfg):
        return None

    run(config)
    assert OmegaConf.load(tmp_path / "experiment.yaml").value == "legacy"
