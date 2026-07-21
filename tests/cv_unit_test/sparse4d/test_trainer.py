# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from platform import machine
import pytest
import os
import tempfile
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.cv.sparse4d.scripts import train as train_script

pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests take very long (~12 hours) on ARM architecture. TODO: Fix this.",
)

DATA_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/full_data/"
TRAIN_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_test_split/SURF_Booth_031325+bev-sensor-buffer-zone-c4_infos_test.pkl"
VAL_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_test_split/SURF_Booth_031325+bev-sensor-buffer-zone-c4_infos_test.pkl"
TEST_ANNO_ROOT = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/anno_pkls/ov_test_split/SURF_Booth_031325+bev-sensor-buffer-zone-c4_infos_test.pkl"
ANCHOR_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/_ov_kmeans900_sample100_.npy"
CHECKPOINT_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/sparse4d_tracking_aic25v0.3_moving_classes_iter_60900_v1.1.pth"
FAST_DEV_RUN = 200
tmp_top_dir = tempfile.mkdtemp()

@pytest.fixture
def _base_spec():
    """Base ExperimentConfig with paths pointing to real data."""
    cfg = OmegaConf.structured(ExperimentConfig())

    # Results dir
    os.makedirs(tmp_top_dir, exist_ok=True)
    cfg.results_dir = tmp_top_dir

    # Dataset config using real paths
    cfg.dataset.data_root = DATA_ROOT
    cfg.dataset.train_dataset.ann_file = TRAIN_ANNO_ROOT
    cfg.dataset.val_dataset.ann_file = VAL_ANNO_ROOT
    cfg.dataset.test_dataset.ann_file = TEST_ANNO_ROOT
    cfg.dataset.classes = ['person', 'gr1_t2', 'agility_digit', 'nova_carter', 'transporter', 'forklift', 'pallet']
    cfg.dataset.batch_size = 1
    cfg.dataset.num_workers = 0
    cfg.dataset.use_h5_file_for_rgb = True
    cfg.dataset.use_h5_file_for_depth = True
    cfg.model.head.instance_bank.anchor = ANCHOR_PATH
    cfg.model.head.deformable_model.use_camera_embed = True
    cfg.train.pretrained_model_path = CHECKPOINT_PATH
    cfg.dataset.train_dataset.sequences_split_num = 100
    OmegaConf.resolve(cfg)
    return cfg

@pytest.fixture
def _train_spec(_base_spec):
    cfg = _base_spec.copy()
    cfg.train.num_epochs = 1
    cfg.train.optim.lr = 1e-6
    cfg.train.checkpoint_interval = 1
    cfg.train.validation_interval = 1
    return cfg

@pytest.fixture
def _eval_spec(_base_spec):
    cfg = _base_spec.copy()
    cfg.evaluate.checkpoint = CHECKPOINT_PATH
    return cfg

@pytest.fixture
def _infer_spec(_base_spec):
    cfg = _base_spec.copy()
    cfg.inference.checkpoint = CHECKPOINT_PATH
    return cfg

@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.train
def test_trainer_fit(_train_spec):
    """Test trainer.fit() using real data paths."""
    print(_train_spec)
    dm = Sparse4DDataModule(_train_spec)
    pt_model = Sparse4DPlModel(_train_spec)
    pretrained_path = _train_spec.train.pretrained_model_path
    if pretrained_path:
        print(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        pt_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully attempted to load weights into Sparse4DPlModel from {pretrained_path}")

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      accelerator='auto',
                      default_root_dir=_train_spec.results_dir,
                      max_epochs=1,
                      num_sanity_val_steps=0,
                      fast_dev_run=FAST_DEV_RUN)

    trainer.fit(pt_model, dm)


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
    monkeypatch.setattr(train_script, "Sparse4DPlModel", lambda experiment_config: object())
    monkeypatch.setattr(train_script, "LearningRateMonitor", lambda **kwargs: object())
    monkeypatch.setattr(train_script, "Trainer", _Trainer)

    train_script.run_experiment(cfg, key="")

    assert captured["max_epochs"] == -1
    assert captured["max_steps"] == 6
    assert captured["ckpt_path"] == "/tmp/epoch_1.pth"


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.evaluate
def test_trainer_evaluate(_eval_spec):
    """Test trainer.test() using real data paths."""
    dm = Sparse4DDataModule(_eval_spec)
    pt_model = Sparse4DPlModel(_eval_spec)
    # Get checkpoint path from evaluate config
    pretrained_path = _eval_spec.evaluate.checkpoint

    if pretrained_path:
        print(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        miss, unexpected = pt_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully attempted to load weights into Sparse4DPlModel from {pretrained_path}")
        print(f"Missing keys: {miss}")
        print(f"Unexpected keys: {unexpected}")
    else:
        print("Warning: No evaluate checkpoint path found in the spec. Running test with initialized weights.")

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      accelerator='auto',
                      default_root_dir=_eval_spec.results_dir,
                      num_sanity_val_steps=0,
                      fast_dev_run=FAST_DEV_RUN)

    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.inference
def test_trainer_inference(_infer_spec):
    """Test trainer.test() using real data paths."""
    dm = Sparse4DDataModule(_infer_spec)
    pt_model = Sparse4DPlModel(_infer_spec)

    # Get checkpoint path from evaluate config
    pretrained_path = _infer_spec.inference.checkpoint
    print(f"pretrained_path: {pretrained_path}")

    if pretrained_path:
        print(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        pt_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully attempted to load weights into Sparse4DPlModel from {pretrained_path}")
    else:
        print("Warning: No evaluate checkpoint path found in the spec. Running test with initialized weights.")

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      accelerator='auto',
                      default_root_dir=_infer_spec.results_dir,
                      num_sanity_val_steps=0,
                      fast_dev_run=FAST_DEV_RUN)

    trainer.predict(pt_model, dm)
