# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 train entrypoint unit tests."""

from unittest import mock

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.core import initialize_experiments
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

    initialize_options = {}

    def initialize(*_args, **kwargs):
        initialize_options.update(kwargs)
        return None, {"devices": [0], "max_epochs": cfg.train.num_epochs}

    monkeypatch.setattr(train, "initialize_train_experiment", initialize)
    monkeypatch.setattr(train, "DinoV3DataModule", lambda *_: object())
    monkeypatch.setattr(train, "DinoV3PlModel", lambda *_: model)
    monkeypatch.setattr(train, "Trainer", _Trainer)

    train.run_experiment(cfg, key="")

    assert captured["log_every_n_steps"] == 7
    assert initialize_options == {
        "allow_off_cadence_final_checkpoint": True,
        "auto_resume": cfg.train.auto_resume,
    }
    trainer.fit.assert_called_once()


@pytest.mark.ssl_unit
@pytest.mark.parametrize("resume_path", [None, ""])
def test_run_experiment_starts_fresh_when_auto_resume_is_disabled(
    tmp_path, monkeypatch, resume_path,
):
    """An unset resume path with auto-resume disabled reaches Lightning as None."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.results_dir = str(tmp_path)
    cfg.train.auto_resume = False
    cfg.train.resume_training_checkpoint_path = resume_path
    # Keep the real initializer away from global RNG, cuDNN and WandB state.
    cfg.train.seed = -1
    cfg.train.cudnn.benchmark = False
    cfg.wandb.enable = False
    # Discovery would pick this up; disabling auto-resume must ignore it.
    (tmp_path / "dinov3_latest.pth").touch()
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    # Keep the CI host's WandB credentials and TensorBoard install out of the result.
    monkeypatch.setattr(initialize_experiments, "check_wandb_logged_in", lambda: False)
    monkeypatch.setattr(initialize_experiments, "TensorBoardLogger", lambda **_: mock.Mock())

    trainer = mock.Mock()
    monkeypatch.setattr(train, "DinoV3DataModule", lambda *_: object())
    monkeypatch.setattr(train, "DinoV3PlModel", lambda *_: mock.Mock())
    monkeypatch.setattr(train, "Trainer", lambda **_: trainer)

    train.run_experiment(cfg, key="")

    assert trainer.fit.call_args.kwargs["ckpt_path"] is None
