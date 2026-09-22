# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Lightning Trainer smoke tests for Sparse4D."""

import pickle

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch import nn
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset import (
    Omniverse3DDetTrackDataset,
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import (
    Sparse4DDataModule,
)
from nvidia_tao_pytorch.cv.sparse4d.model import sparse4d_pl_model as pl_model_module
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.cv.sparse4d.scripts import train as train_script


FAST_DEV_RUN = 1


class _TinyInstanceBank(nn.Module):
    """Parameter-free stand-in for the detector's instance bank."""


class _TinyHead(nn.Module):
    """Expose the instance-bank attribute required by Sparse4DPlModel."""

    def __init__(self):
        super().__init__()
        self.instance_bank = _TinyInstanceBank()


class _TinyDetector(nn.Module):
    """Cheap differentiable detector used to exercise Lightning orchestration."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.head = _TinyHead()

    def forward(self, img, metas):
        """Return one graph-connected result per batch sample."""
        del metas
        signal = 1.0 + img.float().mean() / 1000.0
        score = self.weight * signal
        return [{"score": score} for _ in range(img.shape[0])]

    def simple_test(self, img, metas):
        """Use the same tiny prediction path for Trainer.test."""
        return self(img, metas)


class _TinyCriterion(nn.Module):
    """Cheap trainable criterion matching SetCriterion's public contract."""

    def __init__(
        self,
        model_config,
        instance_bank,
        class_names=None,
        sync_positive_counts=True,
        scrub_nan_gradients=False,
    ):
        super().__init__()
        del model_config, instance_bank, class_names
        self.sync_positive_counts = sync_positive_counts
        self.scrub_nan_gradients = scrub_nan_gradients
        self.loss_keys = ("loss_tiny",)
        self.offset = nn.Parameter(torch.tensor(0.25))

    def forward(self, outputs, data):
        """Return a scalar loss connected to detector and criterion parameters."""
        del data
        score = torch.stack([result["score"] for result in outputs]).mean()
        return {"loss_tiny": (score + self.offset).square()}


@pytest.fixture(autouse=True)
def _lightweight_runtime(monkeypatch):
    """Replace heavyweight vision/metric work while retaining real Trainer hooks."""
    monkeypatch.setattr(
        pl_model_module,
        "build_model",
        lambda experiment_config, export=False: _TinyDetector(),
    )
    monkeypatch.setattr(pl_model_module, "SetCriterion", _TinyCriterion)

    def evaluate(dataset, results, *args, **kwargs):
        del args, kwargs
        assert results
        dataset._test_evaluate_calls = getattr(dataset, "_test_evaluate_calls", 0) + 1
        return {"tiny_metric": 1.0}

    def format_results(dataset, results, *args, **kwargs):
        del args, kwargs
        assert results
        dataset._test_format_calls = getattr(dataset, "_test_format_calls", 0) + 1
        return {"tiny_results": "in-memory"}, None

    monkeypatch.setattr(Omniverse3DDetTrackDataset, "evaluate", evaluate)
    monkeypatch.setattr(Omniverse3DDetTrackDataset, "format_results", format_results)


@pytest.fixture
def _base_spec(tmp_path):
    """Build a complete spec backed only by temporary local artifacts."""
    image_path = tmp_path / "camera_000000.jpg"
    Image.fromarray(np.full((16, 16, 3), 96, dtype=np.uint8)).save(image_path)
    depth_path = tmp_path / "depth_000000.png"
    Image.fromarray(np.full((16, 16), 1000, dtype=np.uint16)).save(depth_path)

    camera_name = "camera_front"
    annotation_path = tmp_path / "scene_infos.pkl"
    info = {
        "token": "TinyScene__000000",
        "timestamp": 0.0,
        "scene_name": "TinyScene",
        "frame_idx": 0,
        "cams": {
            camera_name: {
                "data_path": str(image_path),
                "depth_map_path": str(depth_path),
                "cam_intrinsic": np.eye(3, dtype=np.float32),
                "sensor2world_transform": np.eye(4, dtype=np.float32),
            }
        },
        "gt_boxes": np.asarray([[0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        "gt_names": np.asarray(["person"]),
        "valid_flag": np.asarray([True]),
        "num_lidar_pts": np.asarray([1]),
        "gt_velocity": np.zeros((1, 2), dtype=np.float32),
        "instance_inds": np.asarray([1], dtype=np.int64),
        "asset_inds": np.asarray([1], dtype=np.int64),
        "gt_visibility": [{camera_name: 1.0}],
    }
    with annotation_path.open("wb") as stream:
        pickle.dump({"infos": [info], "metadata": {"version": "tiny"}}, stream)

    anchor_path = tmp_path / "anchors.npy"
    np.save(anchor_path, np.zeros((1, 11), dtype=np.float32))
    checkpoint_path = tmp_path / "tiny_checkpoint.pth"
    torch.save({"state_dict": {}}, checkpoint_path)

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.results_dir = str(results_dir)
    cfg.dataset.data_root = str(tmp_path)
    cfg.dataset.train_dataset.ann_file = str(annotation_path)
    cfg.dataset.val_dataset.ann_file = str(annotation_path)
    cfg.dataset.test_dataset.ann_file = str(annotation_path)
    cfg.dataset.classes = ["person"]
    cfg.dataset.batch_size = 1
    cfg.dataset.num_workers = 0
    cfg.dataset.num_frames = 1
    cfg.dataset.use_h5_file_for_rgb = False
    cfg.dataset.use_h5_file_for_depth = False
    cfg.dataset.sequences.split_num = 1
    cfg.dataset.train_dataset.sequences_split_num = 1
    cfg.dataset.augmentation.image_size = [16, 16]
    cfg.dataset.augmentation.final_dim = [16, 16]
    cfg.dataset.augmentation.resize_lim = [1.0, 1.0]
    cfg.dataset.augmentation.bot_pct_lim = [0.0, 0.0]
    cfg.dataset.augmentation.rot_lim = [0.0, 0.0]
    cfg.dataset.augmentation.rot3d_range = [0.0, 0.0]
    cfg.dataset.augmentation.rand_flip = False
    cfg.model.input_shape = [16, 16]
    cfg.model.head.instance_bank.anchor = str(anchor_path)
    cfg.train.pretrained_model_path = str(checkpoint_path)
    cfg.evaluate.checkpoint = str(checkpoint_path)
    cfg.inference.checkpoint = str(checkpoint_path)
    cfg.inference.output_nvschema = False
    cfg.inference.jsonfile_prefix = str(tmp_path / "predictions")
    cfg.visualize.vis_dir = str(tmp_path / "visualizations")
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture
def _train_spec(_base_spec):
    """Return the self-contained training spec."""
    cfg = _base_spec.copy()
    cfg.train.num_epochs = 1
    cfg.train.optim.lr = 1e-2
    cfg.train.checkpoint_interval = 1
    cfg.train.validation_interval = 1
    return cfg


@pytest.fixture
def _eval_spec(_base_spec):
    """Return the self-contained evaluation spec."""
    return _base_spec.copy()


@pytest.fixture
def _infer_spec(_base_spec):
    """Return the self-contained inference spec."""
    return _base_spec.copy()


def _load_temporary_checkpoint(model, checkpoint_path):
    """Exercise Sparse4D's checkpoint loader against the temporary checkpoint."""
    state_dict = load_pretrained_weights(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert missing
    assert not unexpected


def _trainer(spec, devices):
    """Build a deterministic one-batch CPU Trainer."""
    return Trainer(
        devices=devices,
        num_nodes=1,
        accelerator="cpu",
        default_root_dir=spec.results_dir,
        num_sanity_val_steps=0,
        fast_dev_run=FAST_DEV_RUN,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_positive_count_sync_policy_is_wired_conservatively(_train_spec):
    """Only route-specific early losses disable baseline count reduction."""
    baseline = Sparse4DPlModel(_train_spec, build_training_losses=True)
    assert baseline.criterion.sync_positive_counts is True

    _train_spec.model.cotrain_param_touch = True
    parameter_touch_only = Sparse4DPlModel(_train_spec, build_training_losses=True)
    assert parameter_touch_only.criterion.sync_positive_counts is True

    _train_spec.model.head.loose_to_tight.enable = True
    _train_spec.model.head.loose_to_tight.pseudo_enable = True
    ltt_pseudo = Sparse4DPlModel(_train_spec, build_training_losses=True)
    assert ltt_pseudo.criterion.sync_positive_counts is False

    _train_spec.model.head.loose_to_tight.enable = False
    _train_spec.model.head.loose_to_tight.pseudo_enable = False
    _train_spec.model.sv_aux_head.enable = True
    _train_spec.dataset.sync_route = True
    sv_with_declared_route_sync = Sparse4DPlModel(_train_spec, build_training_losses=True)
    assert sv_with_declared_route_sync.criterion.sync_positive_counts is False


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.train
def test_trainer_fit(_train_spec):
    """Run Sparse4DPlModel through a real Trainer fit and validation cycle."""
    dm = Sparse4DDataModule(_train_spec)
    model = Sparse4DPlModel(_train_spec, build_training_losses=True)
    _load_temporary_checkpoint(model, _train_spec.train.pretrained_model_path)
    initial_weight = model.model.weight.detach().clone()

    trainer = _trainer(_train_spec, _train_spec.train.num_gpus)
    trainer.fit(model, dm)

    assert trainer.global_step == 1
    assert not torch.equal(model.model.weight.detach(), initial_weight)
    assert dm.val_dataset._test_evaluate_calls == 1


@pytest.mark.cv_unit
@pytest.mark.sparse4d
def test_run_experiment_uses_step_budget_as_only_stop_condition(monkeypatch):
    """A resumed Sparse4D run must consume its increased step budget."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.train.num_epochs = 2
    cfg.train.num_nodes = 1
    cfg.train.num_gpus = 1
    cfg.train.pretrained_model_path = None
    cfg.dataset.batch_size = 1
    cfg.dataset.num_frames = 3
    cfg.dataset.num_bev_groups = 1

    captured = {}

    class _Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, model, dm, ckpt_path=None):
            captured["ckpt_path"] = ckpt_path

    monkeypatch.setattr(
        train_script,
        "initialize_train_experiment",
        lambda experiment_config, key: (
            "/tmp/epoch_1.pth",
            {"devices": [0], "max_epochs": cfg.train.num_epochs},
        ),
    )
    monkeypatch.setattr(train_script, "Sparse4DDataModule", lambda experiment_config: object())
    monkeypatch.setattr(train_script, "Sparse4DPlModel", lambda experiment_config, **kwargs: object())
    monkeypatch.setattr(train_script, "LearningRateMonitor", lambda **kwargs: object())
    monkeypatch.setattr(train_script, "Trainer", _Trainer)

    train_script.run_experiment(cfg, key="")

    assert captured["max_epochs"] == -1
    assert captured["max_steps"] == 6
    assert captured["ckpt_path"] == "/tmp/epoch_1.pth"


@pytest.mark.parametrize("export", [False, True])
def test_nontraining_model_does_not_build_loss_dependencies(_train_spec, monkeypatch, export):
    """A saved LTT-enabled spec needs no training-only MLP for inference/export."""
    _train_spec.model.head.loose_to_tight.enable = True
    _train_spec.model.head.loose_to_tight.mlp_ckpt = "/missing/training-only.pth"
    def unexpected_criterion(*_args, **_kwargs):
        raise AssertionError("non-training construction must not build criterion")
    monkeypatch.setattr(pl_model_module, "SetCriterion", unexpected_criterion)
    model = Sparse4DPlModel(_train_spec, export=export)
    assert model.criterion is None


def test_training_losses_are_registered_before_checkpoint_restore(_train_spec):
    """Training-only parameters exist before Lightning loads/resumes a checkpoint."""
    _train_spec.train.scrub_nan_gradients = True
    model = Sparse4DPlModel(_train_spec, build_training_losses=True)
    assert model.criterion.scrub_nan_gradients
    state = model.state_dict()
    assert "criterion.offset" in state
    state["criterion.offset"] = torch.tensor(4.0)
    restored = Sparse4DPlModel(_train_spec, build_training_losses=True)
    restored.load_state_dict(state)
    assert restored.criterion.offset.item() == 4.0


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.evaluate
def test_trainer_evaluate(_eval_spec):
    """Run Sparse4DPlModel through a real Trainer test cycle."""
    dm = Sparse4DDataModule(_eval_spec)
    model = Sparse4DPlModel(_eval_spec)
    _load_temporary_checkpoint(model, _eval_spec.evaluate.checkpoint)

    trainer = _trainer(_eval_spec, _eval_spec.evaluate.num_gpus)
    trainer.test(model, dm)

    assert dm.test_dataset._test_evaluate_calls == 1
    assert dm.test_dataset.results == []


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.inference
def test_trainer_inference(_infer_spec):
    """Run Sparse4DPlModel through a real Trainer predict cycle."""
    dm = Sparse4DDataModule(_infer_spec)
    model = Sparse4DPlModel(_infer_spec)
    _load_temporary_checkpoint(model, _infer_spec.inference.checkpoint)

    trainer = _trainer(_infer_spec, _infer_spec.inference.num_gpus)
    predictions = trainer.predict(model, dm)

    assert len(predictions) == 1
    assert dm.test_dataset._test_format_calls == 1
