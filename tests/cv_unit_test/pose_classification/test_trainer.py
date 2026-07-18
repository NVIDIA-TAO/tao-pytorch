
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pose Classification Trainer Unit Tests
"""
import os
import pytest
import tempfile
import numpy as np
import pickle

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.pose_classification.dataloader.pl_pc_data_module import PCDataModule
from nvidia_tao_pytorch.config.pose_classification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.model.pl_pc_model import PoseClassificationModel

from pytorch_lightning import Trainer


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
nvidia_data_path = os.path.join(tmp_top_dir, "nvidia_data.npy")
openpose_data_path = os.path.join(tmp_top_dir, "openpose_data.npy")
label_path = os.path.join(tmp_top_dir, "test_label.pkl")
out_path = os.path.join(tmp_top_dir, "inference.txt")
FAST_DEV_RUN = 2


@pytest.fixture
def _test_dir():
    check_and_create(tmp_top_dir)

    test_data = np.random.rand(3, 3, 300, 34, 1).astype(np.float32)
    np.save(file=nvidia_data_path,
            arr=test_data, allow_pickle=False)

    test_data = np.random.rand(3, 3, 300, 18, 1).astype(np.float32)
    np.save(file=openpose_data_path,
            arr=test_data, allow_pickle=False)

    test_label = [["a", "b", "c"], [0, 1, 2]]
    with open(label_path, "wb") as f:
        pickle.dump(test_label, f, protocol=4)


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.results_dir = tmp_top_dir
    experiment_config.train.results_dir = tmp_top_dir

    experiment_config.dataset.train_dataset.data_path = nvidia_data_path
    experiment_config.dataset.train_dataset.label_path = label_path
    experiment_config.dataset.val_dataset.data_path = nvidia_data_path
    experiment_config.dataset.val_dataset.label_path = label_path
    experiment_config.dataset.label_map = {"a": 0, "b": 1, "c": 2}
    experiment_config.dataset.num_classes = 3
    experiment_config.dataset.batch_size = 2

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.results_dir = tmp_top_dir

    experiment_config.evaluate.results_dir = tmp_top_dir
    experiment_config.evaluate.test_dataset.data_path = nvidia_data_path
    experiment_config.evaluate.test_dataset.label_path = label_path

    experiment_config.dataset.label_map = {"a": 0, "b": 1, "c": 2}
    experiment_config.dataset.num_classes = 3
    experiment_config.dataset.batch_size = 2

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.results_dir = tmp_top_dir

    experiment_config.inference.results_dir = tmp_top_dir
    experiment_config.inference.test_dataset.data_path = nvidia_data_path
    experiment_config.inference.output_file = out_path

    experiment_config.dataset.label_map = {"a": 0, "b": 1, "c": 2}
    experiment_config.dataset.num_classes = 3
    experiment_config.dataset.batch_size = 2

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.pose_classification
@pytest.mark.train
@pytest.mark.parametrize("graph_layout", ["nvidia", "openpose"])
def test_trainer_fit(_test_dir, _train_spec, graph_layout):

    _train_spec.model.graph_layout = graph_layout
    if graph_layout == 'openpose':
        _train_spec.dataset.random_choose = True
        _train_spec.dataset.random_move = True
        _train_spec.dataset.window_size = 150
        _train_spec.dataset.train_dataset.data_path = openpose_data_path
        _train_spec.dataset.val_dataset.data_path = openpose_data_path

    dm = PCDataModule(_train_spec)
    dm.setup('fit')
    model = PoseClassificationModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      gradient_clip_val=_train_spec.train.grad_clip,
                      fast_dev_run=FAST_DEV_RUN
                      )

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.pose_classification
@pytest.mark.evaluate
@pytest.mark.parametrize("graph_layout", ["nvidia", "openpose"])
def test_trainer_evaluate(_test_dir, _eval_spec, graph_layout):

    _eval_spec.model.graph_layout = graph_layout
    if graph_layout == 'openpose':
        _eval_spec.evaluate.test_dataset.data_path = openpose_data_path

    dm = PCDataModule(_eval_spec)
    dm.setup('test')
    model = PoseClassificationModel(_eval_spec)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN
                      )

    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.pose_classification
@pytest.mark.inference
@pytest.mark.parametrize("graph_layout", ["nvidia", "openpose"])
def test_trainer_infer(_test_dir, _infer_spec, graph_layout):

    _infer_spec.model.graph_layout = graph_layout
    if graph_layout == 'openpose':
        _infer_spec.inference.test_dataset.data_path = openpose_data_path

    dm = PCDataModule(_infer_spec)
    dm.setup('predict')
    model = PoseClassificationModel(_infer_spec)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN
                      )

    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
