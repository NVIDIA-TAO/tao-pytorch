# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 train entrypoint unit tests."""

from unittest import mock

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.scripts import train


@pytest.mark.ssl_unit
def test_run_experiment_forwards_logging_interval(monkeypatch):
    """The DINOv3 train config controls Lightning's logging cadence."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.train.log_every_n_steps = 7

    captured = {}
    trainer = mock.Mock()
    model = mock.Mock()

    class _Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, *args, **kwargs):
            trainer.fit(*args, **kwargs)

    monkeypatch.setattr(
        train,
        "initialize_train_experiment",
        lambda *_: (None, {"devices": [0], "max_epochs": cfg.train.num_epochs}),
    )
    monkeypatch.setattr(train, "DinoV2DataModule", lambda *_: object())
    monkeypatch.setattr(train, "DinoV3PlModel", lambda *_: model)
    monkeypatch.setattr(train, "Trainer", _Trainer)

    train.run_experiment(cfg, key="")

    assert captured["log_every_n_steps"] == 7
    trainer.fit.assert_called_once()
