# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-only worker and checkpoint regressions."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.dataloader.pl_dinov3_data_module import DinoV3DataModule
from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint import DinoV3ModelCheckpoint
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import CustomModelCheckpoint

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

    class Strategy:
        @staticmethod
        def lightning_module_state_dict():
            calls["state_dict"] += 1
            return {"student.backbone.weight": torch.tensor([1.0])}

    trainer = SimpleNamespace(
        save_checkpoint=lambda *_: calls.__setitem__("save", calls["save"] + 1),
        lightning_module=Module(),
        strategy=Strategy(),
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
    assert periodic.save_final_epoch is True
    assert before == [[getattr(cls, name, None) for name in names] for cls in shared]


@pytest.mark.parametrize(
    ("max_epochs", "current_epoch", "global_step", "callback_options"),
    [
        (10, 9, 31, {"every_n_epochs": 3}),
        (2, 1, 7, {"every_n_train_steps": 5}),
    ],
)
def test_terminal_checkpoint_is_published_off_periodic_cadence(
    tmp_path,
    monkeypatch,
    max_epochs,
    current_epoch,
    global_step,
    callback_options,
):
    """The terminal manifest extent is saved even when no cadence boundary lands."""
    calls = []
    monkeypatch.setattr(
        CustomModelCheckpoint,
        "on_train_epoch_end",
        lambda _self, _trainer, _module: calls.append("periodic"),
    )
    callback = DinoV3ModelCheckpoint(
        dirpath=tmp_path,
        save_final_epoch=True,
        **callback_options,
    )
    callback._last_global_step_saved = global_step - 1
    monkeypatch.setattr(callback, "_should_skip_saving_checkpoint", lambda _trainer: False)
    monkeypatch.setattr(callback, "_monitor_candidates", lambda _trainer: {})
    monkeypatch.setattr(
        callback,
        "_save_topk_checkpoint",
        lambda _trainer, _candidates: calls.append("topk"),
    )
    monkeypatch.setattr(
        callback,
        "_save_last_checkpoint",
        lambda _trainer, _candidates: calls.append("last"),
    )
    trainer = SimpleNamespace(
        max_epochs=max_epochs,
        current_epoch=current_epoch,
        global_step=global_step,
    )
    callback.on_train_epoch_end(trainer, SimpleNamespace())
    assert calls == ["periodic", "topk", "last"]


@pytest.mark.parametrize(
    ("every_n_epochs", "expected_steps"),
    [(10, [3]), (1, [1, 2, 3])],
)
def test_real_lightning_callback_saves_terminal_extent_once(
    tmp_path, every_n_epochs, expected_steps
):
    """Lightning scheduling and the final-extent hook never double-save a step."""

    class RecordingCheckpoint(DinoV3ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved_steps = []

        def _save_checkpoint(self, trainer, filepath):
            self.saved_steps.append(trainer.global_step)
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

    class TinyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            del batch, batch_idx
            return self.weight * 0

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    callback = RecordingCheckpoint(
        dirpath=tmp_path,
        every_n_epochs=every_n_epochs,
        save_top_k=-1,
        save_final_epoch=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        TinyModule(),
        train_dataloaders=DataLoader(TensorDataset(torch.ones(1, 1))),
    )
    assert callback.saved_steps == expected_steps
    assert len(callback.saved_steps) == len(set(callback.saved_steps))


@pytest.mark.parametrize(
    ("every_n_train_steps", "expected_steps"),
    [(10, [3]), (2, [2, 3]), (3, [3])],
)
def test_real_lightning_step_cadence_saves_terminal_extent_once(
    tmp_path, every_n_train_steps, expected_steps
):
    """Step cadence and terminal publication never save one extent twice."""

    class RecordingCheckpoint(DinoV3ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved_steps = []

        def _save_checkpoint(self, trainer, filepath):
            self.saved_steps.append(trainer.global_step)
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath

    class TinyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            del batch, batch_idx
            return self.weight * 0

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    callback = RecordingCheckpoint(
        dirpath=tmp_path,
        every_n_train_steps=every_n_train_steps,
        save_top_k=-1,
        save_final_epoch=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        TinyModule(),
        train_dataloaders=DataLoader(TensorDataset(torch.ones(1, 1))),
    )
    assert callback.saved_steps == expected_steps
    assert len(callback.saved_steps) == len(set(callback.saved_steps))
