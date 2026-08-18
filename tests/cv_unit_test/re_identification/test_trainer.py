# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ReIdentification Trainer Unit Tests
"""
import os
import pytest
import tempfile

from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.re_identification.dataloader.pl_reid_data_module import REIDDataModule
from nvidia_tao_pytorch.config.re_identification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel

from pytorch_lightning import Trainer


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
bbox_train = os.path.join(tmp_top_dir, "test_data", "bounding_box_train")
bbox_test = os.path.join(tmp_top_dir, "test_data", "bounding_box_test")
query = os.path.join(tmp_top_dir, "test_data", "query")
gt_query = os.path.join(tmp_top_dir, "test_data", "gt_query")
FAST_DEV_RUN = 2

# TODO: make a larger test dataset for training and testing to work

@pytest.fixture
def _test_dir():
    os.makedirs(tmp_top_dir, exist_ok=True)
    os.system(f'tar -xvf /home/scratch.metropolis2/tao_ci/tao_pytorch/data/re_identification/test_data.tar.xz --directory {tmp_top_dir}')


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.results_dir = tmp_top_dir
    experiment_config.train.results_dir = tmp_top_dir

    experiment_config.dataset.train_dataset_dir = bbox_train
    experiment_config.dataset.query_dataset_dir = query
    experiment_config.dataset.test_dataset_dir = bbox_test
    experiment_config.dataset.num_instances = 8
    experiment_config.dataset.num_classes = 1
    experiment_config.dataset.batch_size = 8
    experiment_config.dataset.num_workers = 1
    experiment_config.dataset.val_batch_size = 2

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.evaluate.query_dataset = gt_query
    experiment_config.evaluate.test_dataset = bbox_test

    experiment_config.dataset.num_workers = 1
    experiment_config.dataset.val_batch_size = 2

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    experiment_config.inference.query_dataset = gt_query
    experiment_config.inference.test_dataset = bbox_test
    experiment_config.inference.output_file = os.path.join(tmp_top_dir, "inference.json")

    experiment_config.dataset.num_workers = 1
    experiment_config.dataset.val_batch_size = 2

    yield experiment_config


# @pytest.mark.cv_unit
# @pytest.mark.re_identification
# @pytest.mark.train
# @pytest.mark.parametrize("backbone", ["resnet_50", "swin_tiny_patch4_window7_224"])
# def test_trainer_fit(_test_dir, _train_spec, backbone):

#     _train_spec.model.backbone = backbone

#     if "swin" in backbone:
#         _train_spec.model.stride_size = [16, 16]
#         _train_spec.model.reduce_feat_dim = True
#         _train_spec.model.no_margin = True
#         _train_spec.model.label_smooth = True

#     dm = REIDDataModule(_train_spec)
#     dm.setup('fit')
#     model = ReIdentificationModel(_train_spec, prepare_for_training=True)

#     trainer = Trainer(devices=_train_spec.train.num_gpus,
#                       num_nodes=_train_spec.train.num_nodes,
#                       precision='16-mixed',
#                       gradient_clip_val=_train_spec.train.grad_clip,
#                       default_root_dir=_train_spec.results_dir,
#                       fast_dev_run=FAST_DEV_RUN,
#                       limit_val_batches=0
#                       )

#     # Test train
#     trainer.fit(model, dm)


# @pytest.mark.cv_unit
# @pytest.mark.re_identification
# @pytest.mark.evaluate
# @pytest.mark.parametrize("backbone", ["resnet_50", "swin_tiny_patch4_window7_224"])
# def test_trainer_evaluate(_test_dir, _eval_spec, backbone):

#     _eval_spec.model.backbone = backbone

#     if "swin" in backbone:
#         _eval_spec.model.stride_size = [16, 16]
#         _eval_spec.model.reduce_feat_dim = True
#         _eval_spec.model.no_margin = True
#         _eval_spec.model.label_smooth = True

#     dm = REIDDataModule(_eval_spec)
#     dm.setup('test')
#     model = ReIdentificationModel(_eval_spec, prepare_for_training=False)

#     trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
#                       default_root_dir=_eval_spec.results_dir,
#                       fast_dev_run=FAST_DEV_RUN,
#                       limit_val_batches=0
#                       )

#     trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.re_identification
@pytest.mark.inference
@pytest.mark.parametrize("backbone", ["resnet_50", "swin_tiny_patch4_window7_224"])
def test_trainer_infer(_test_dir, _infer_spec, backbone):

    _infer_spec.model.backbone = backbone

    if "swin" in backbone:
        _infer_spec.model.stride_size = [16, 16]
        _infer_spec.model.reduce_feat_dim = True
        _infer_spec.model.no_margin = True
        _infer_spec.model.label_smooth = True

    dm = REIDDataModule(_infer_spec)
    dm.setup('predict')
    model = ReIdentificationModel(_infer_spec, prepare_for_training=False)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN
                      )

    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
