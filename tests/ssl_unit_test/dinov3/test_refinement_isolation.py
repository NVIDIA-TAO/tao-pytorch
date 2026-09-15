# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-only worker and checkpoint regressions."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.dataloader.pl_dinov3_data_module import DinoV3DataModule
from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint import DinoV3ModelCheckpoint

BATCH_SIZE = 2


@pytest.fixture
def _test_exp_spec(tmp_path):
    config = OmegaConf.structured(ExperimentConfig())
    for index in range(4):
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(tmp_path / f"{index}.png")
    config.dataset.train_dataset.images_dir = str(tmp_path)
    config.dataset.batch_size = BATCH_SIZE
    config.results_dir = str(tmp_path)
    return config


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_dataloader_worker_modes(_test_exp_spec, workers, stage):
    """Training and prediction support synchronous and persistent workers."""
    _test_exp_spec.dataset.workers = workers
    _test_exp_spec.dataset.test_dataset.images_dir = (
        _test_exp_spec.dataset.train_dataset.images_dir
    )
    data_module = DinoV3DataModule(_test_exp_spec)
    data_module.setup(stage)
    loader = (
        data_module.train_dataloader() if stage == 'fit'
        else data_module.predict_dataloader()
    )
    assert loader.persistent_workers is (workers > 0)
    batch = next(iter(loader))
    if stage == 'fit':
        assert batch['global_crops'].shape[0] == BATCH_SIZE * 2
    else:
        assert batch['images'].shape[0] == 1
        assert len(batch['input_path']) == 1



def test_converted_checkpoints_are_published_only_by_global_rank_zero(
    tmp_path, monkeypatch
):
    """Nonzero ranks join state-dict collection but never race on export files."""
    calls = {"state_dict": 0, "save": 0, "atomic": 0}

    class Module:
        model_config = SimpleNamespace(distill=SimpleNamespace(enable=False))

        @staticmethod
        def state_dict():
            calls["state_dict"] += 1
            return {"student.backbone.weight": torch.tensor([1.0])}

    trainer = SimpleNamespace(
        save_checkpoint=lambda *_: calls.__setitem__("save", calls["save"] + 1),
        lightning_module=Module(),
        is_global_zero=False,
        default_root_dir=str(tmp_path),
        current_epoch=0,
        global_step=10,
        loggers=[],
    )
    monkeypatch.setattr(
        "nvidia_tao_pytorch.ssl.dinov3.model.checkpoint._atomic_torch_save",
        lambda *_: calls.__setitem__("atomic", calls["atomic"] + 1),
    )

    callback = DinoV3ModelCheckpoint(dirpath=tmp_path)
    callback._save_checkpoint(trainer, str(tmp_path / "last.ckpt"))

    assert calls == {"state_dict": 1, "save": 1, "atomic": 0}


@pytest.mark.parametrize("unit", ["epoch", "step"])
@pytest.mark.parametrize("manifest", [None, "train.parquet"])
def test_dinov3_callbacks_do_not_mutate_shared_classes(tmp_path, unit, manifest):
    """DINOv3 construction leaves NVDINOv2 and shared exception names intact."""
    from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
    from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import CustomModelCheckpoint
    from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

    shared = (CustomModelCheckpoint, TAOExceptionCheckpoint)
    names = ("FILE_EXTENSION", "CHECKPOINT_EQUALS_CHAR", "CHECKPOINT_NAME_LAST")
    before = [[getattr(cls, name, None) for name in names] for cls in shared]
    model = SimpleNamespace(
        experiment_spec={"results_dir": str(tmp_path),
                         "dataset": {"train_manifest": manifest}, "train": {
            "checkpoint_interval_unit": unit, "checkpoint_interval": 2}},
        _configure_best_checkpoint=lambda callbacks, _root: callbacks,
    )
    callbacks = DinoV3PlModel.configure_callbacks(model)
    periodic = next(cb for cb in callbacks if isinstance(cb, DinoV3ModelCheckpoint))
    assert periodic.CHECKPOINT_NAME_LAST == "dinov3_model_latest"
    assert periodic._every_n_train_steps == (2 if unit == "step" else 0)
    assert periodic.save_final_epoch is bool(manifest)
    assert before == [[getattr(cls, name, None) for name in names] for cls in shared]
