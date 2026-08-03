# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for network-aware best-checkpoint saving."""

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (
    TAOLightningModule, validate_monitor_metric,
)


class _DummyModule(TAOLightningModule):
    """Minimal concrete module to exercise configure_callbacks()."""

    def __init__(self, spec, monitor="val_acc", mode="max"):
        super().__init__(spec)
        self.checkpoint_filename = "dummy_model"
        self.monitor_metric = monitor
        self.monitor_mode = mode


def _spec(results_dir, checkpointer=None):
    train = {"checkpoint_interval": 1, "checkpoint_interval_unit": "epoch"}
    if checkpointer is not None:
        train["checkpointer"] = checkpointer
    return {"results_dir": str(results_dir), "dataset": {}, "model": {}, "train": train}


def _best_callbacks(callbacks):
    """ModelCheckpoints that actually monitor a metric (the 'best' ones)."""
    return [c for c in callbacks if isinstance(c, ModelCheckpoint) and c.monitor is not None]


def _periodic_callbacks(callbacks):
    """Unbounded, unmonitored periodic ModelCheckpoints."""
    return [
        c for c in callbacks
        if isinstance(c, ModelCheckpoint) and c.monitor is None and c.save_top_k == -1
    ]


def test_base_defaults_present():
    m = TAOLightningModule.__new__(TAOLightningModule)  # no model build
    TAOLightningModule.__init__(m, _spec("/tmp"))
    assert m.monitor_metric == "val_loss"
    assert m.monitor_mode == "min"


def test_no_checkpointer_block_is_backward_compatible(tmp_path):
    # Old spec: no 'checkpointer' key at all -> no monitored checkpoint appended.
    m = _DummyModule(_spec(tmp_path))
    cbs = m.configure_callbacks()
    assert _best_callbacks(cbs) == []
    assert len(_periodic_callbacks(cbs)) == 1


def test_disabled_topk_appends_nothing(tmp_path):
    m = _DummyModule(_spec(tmp_path, checkpointer={"enable_topk": False}))
    callbacks = m.configure_callbacks()
    assert _best_callbacks(callbacks) == []
    assert len(_periodic_callbacks(callbacks)) == 1


def test_replace_periodic_requires_topk_to_be_enabled(tmp_path):
    cfg = {"enable_topk": False, "replace_periodic": True}
    callbacks = _DummyModule(_spec(tmp_path, checkpointer=cfg)).configure_callbacks()
    assert _best_callbacks(callbacks) == []
    assert len(_periodic_callbacks(callbacks)) == 1


def test_enabled_uses_network_default_metric(tmp_path):
    m = _DummyModule(_spec(tmp_path, checkpointer={"enable_topk": True}),
                     monitor="val_acc", mode="max")
    best = _best_callbacks(m.configure_callbacks())
    assert len(best) == 1
    cb = best[0]
    assert cb.monitor == "val_acc"
    assert cb.mode == "max"
    # best callback must not fight the periodic callback over *_latest, and must
    # rank at validation end:
    assert cb.save_last is False
    assert cb._save_on_train_epoch_end is False
    assert str(cb.dirpath) == str(tmp_path)   # defaults to results_dir
    assert len(_periodic_callbacks(m.configure_callbacks())) == 1


def test_replace_periodic_keeps_one_best_and_owns_latest_symlink(tmp_path):
    cfg = {
        "enable_topk": True,
        "replace_periodic": True,
        "save_top_k": 3,
    }
    callbacks = _DummyModule(
        _spec(tmp_path, checkpointer=cfg), monitor="val_acc", mode="max"
    ).configure_callbacks()

    best = _best_callbacks(callbacks)
    assert len(best) == 1
    assert _periodic_callbacks(callbacks) == []
    assert best[0].monitor == "val_acc"
    assert best[0].mode == "max"
    assert best[0].save_top_k == 1
    assert best[0].save_last == "link"
    assert best[0].CHECKPOINT_NAME_LAST == "dummy_model_latest"
    assert best[0]._save_on_train_epoch_end is False


def test_config_overrides_network_default(tmp_path):
    cfg = {"enable_topk": True, "monitor": "val_loss", "mode": "min", "save_top_k": 3}
    m = _DummyModule(_spec(tmp_path, checkpointer=cfg), monitor="val_acc", mode="max")
    cb = _best_callbacks(m.configure_callbacks())[0]
    assert cb.monitor == "val_loss"   # override wins over network default
    assert cb.mode == "min"
    assert cb.save_top_k == 3


def test_custom_dirpath_respected(tmp_path):
    sub = tmp_path / "best"
    cfg = {"enable_topk": True, "dirpath": str(sub)}
    m = _DummyModule(_spec(tmp_path, checkpointer=cfg))
    cb = _best_callbacks(m.configure_callbacks())[0]
    assert str(cb.dirpath) == str(sub)


def test_helper_returns_list_when_disabled(tmp_path):
    # Models that fully override configure_callbacks call
    #   callbacks = self._configure_best_checkpoint(callbacks, results_dir)
    # so the helper must return the list (not None) even when the feature is off.
    m = _DummyModule(_spec(tmp_path))
    cbs = []
    out = m._configure_best_checkpoint(cbs, str(tmp_path))
    assert out is cbs
    assert _best_callbacks(out) == []


def test_helper_returns_list_when_enabled(tmp_path):
    m = _DummyModule(_spec(tmp_path, checkpointer={"enable_topk": True}),
                     monitor="val_mAP", mode="max")
    cbs = []
    out = m._configure_best_checkpoint(cbs, str(tmp_path))
    assert out is cbs                       # same list object returned
    best = _best_callbacks(out)
    assert len(best) == 1 and best[0].monitor == "val_mAP" and best[0].mode == "max"


def test_helper_removes_override_periodic_callback_in_replace_mode(tmp_path):
    cfg = {"enable_topk": True, "replace_periodic": True}
    m = _DummyModule(_spec(tmp_path, checkpointer=cfg))
    periodic = ModelCheckpoint(
        dirpath=tmp_path,
        monitor=None,
        save_top_k=-1,
        every_n_epochs=1,
    )
    callbacks = [periodic]

    out = m._configure_best_checkpoint(callbacks, str(tmp_path))

    assert out is callbacks
    assert periodic not in out
    assert _periodic_callbacks(out) == []
    assert len(_best_callbacks(out)) == 1


def test_runtime_guard_raises_on_unlogged_metric():
    with pytest.raises(ValueError, match="is not logged"):
        validate_monitor_metric("val_miou", {"val_loss", "val_acc"})


def test_runtime_guard_passes_when_present():
    validate_monitor_metric("val_acc", {"val_loss", "val_acc"})  # no raise


def test_config_schema_default_disabled():
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.config.common.common_config import TrainConfig
    schema = OmegaConf.structured(TrainConfig)
    assert schema.checkpointer.enable_topk is False
    assert schema.checkpointer.replace_periodic is False
    assert schema.checkpointer.monitor is None
    assert schema.checkpointer.mode is None
