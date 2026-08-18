# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optical Inspection Trainer Unit Tests
"""
import os
import pytest
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.optical_inspection.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel

from pytorch_lightning import Trainer


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
csv_file = os.path.join(tmp_top_dir, 'test_data.csv')
SAMPLES = 10
BATCH_SIZE = 2
IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112
NUM_INPUT = 4
# FAST_DEV_RUN won't work for inference since it needs a full epoch to be run
FAST_DEV_RUN = 2


@pytest.fixture
def _test_dir():
    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_test_dir = os.path.join(tmp_top_dir, "test")
    tmp_golden_dir = os.path.join(tmp_top_dir, "golden")

    check_and_create(tmp_test_dir)
    check_and_create(tmp_golden_dir)
    test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    lighting = ['LowAngleLight', 'SolderLight', 'UniformLight', 'WhiteLight']
    labels = ['PASS', 'MISSING']
    total_samples = SAMPLES
    comp_name = 'test'
    csv_data = []
    for sample in range(total_samples):
        for label in labels:
            for light in lighting:
                save_light = comp_name + '_' + light + '.jpg'
                im.save(os.path.join(tmp_test_dir, save_light))
                im.save(os.path.join(tmp_golden_dir, save_light))
                csv_data.append({
                    'input_path': 'test',
                    'golden_path': 'golden',
                    'label': label,
                    'object_name': comp_name
                })

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    yield csv_file


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.train_dataset.csv_path = csv_file
    experiment_config.dataset.validation_dataset.csv_path = csv_file
    experiment_config.dataset.train_dataset.images_dir = tmp_top_dir
    experiment_config.dataset.validation_dataset.images_dir = tmp_top_dir
    experiment_config.dataset.num_input = NUM_INPUT
    experiment_config.dataset.image_width = IMAGE_WIDTH
    experiment_config.dataset.image_height = IMAGE_HEIGHT
    experiment_config.dataset.input_map = {'LowAngleLight': 0,
                                            'SolderLight': 1,
                                            'UniformLight': 2,
                                            'WhiteLight': 3
                                            }
    experiment_config.dataset.concat_type = 'grid'
    experiment_config.dataset.grid_map = {'x': 2, 'y': 2}
    experiment_config.dataset.image_ext = '.jpg'
    experiment_config.dataset.batch_size = BATCH_SIZE

    experiment_config.results_dir = tmp_top_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.test_dataset.csv_path = csv_file
    experiment_config.dataset.test_dataset.images_dir = tmp_top_dir
    experiment_config.dataset.num_input = NUM_INPUT
    experiment_config.dataset.image_width = IMAGE_WIDTH
    experiment_config.dataset.image_height = IMAGE_HEIGHT
    experiment_config.dataset.input_map = {'LowAngleLight': 0,
                                            'SolderLight': 1,
                                            'UniformLight': 2,
                                            'WhiteLight': 3
                                            }
    experiment_config.dataset.concat_type = 'grid'
    experiment_config.dataset.grid_map = {'x': 2, 'y': 2}
    experiment_config.dataset.image_ext = '.jpg'
    experiment_config.dataset.batch_size = BATCH_SIZE

    experiment_config.results_dir = tmp_top_dir

    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.num_nodes = 1

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.infer_dataset.csv_path = csv_file
    experiment_config.dataset.infer_dataset.images_dir = tmp_top_dir
    experiment_config.dataset.num_input = NUM_INPUT
    experiment_config.dataset.image_width = IMAGE_WIDTH
    experiment_config.dataset.image_height = IMAGE_HEIGHT
    experiment_config.dataset.input_map = {'LowAngleLight': 0,
                                            'SolderLight': 1,
                                            'UniformLight': 2,
                                            'WhiteLight': 3
                                            }
    experiment_config.dataset.concat_type = 'grid'
    experiment_config.dataset.grid_map = {'x': 2, 'y': 2}
    experiment_config.dataset.image_ext = '.jpg'
    experiment_config.dataset.batch_size = BATCH_SIZE

    experiment_config.results_dir = tmp_top_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.num_nodes = 1

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
@pytest.mark.train
def test_trainer_fit(_test_dir, _train_spec):

    dm = OIDataModule(_train_spec)
    dm.setup('fit')
    model = OpticalInspectionModel(_train_spec, dm)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      default_root_dir=_train_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN
                      )
    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
@pytest.mark.evaluate
def test_trainer_evaluate(_test_dir, _eval_spec):

    dm = OIDataModule(_eval_spec)
    dm.setup('test')
    model = OpticalInspectionModel(_eval_spec, dm)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN
                      )

    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.optical_inspection
@pytest.mark.inference
def test_trainer_infer(_test_dir, _infer_spec):

    dm = OIDataModule(_infer_spec)
    dm.setup('predict')
    model = OpticalInspectionModel(_infer_spec, dm)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir
                      )

    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
