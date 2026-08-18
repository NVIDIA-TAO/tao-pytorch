# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StyleGAN-XL Model Unit Tests
"""
import os
import pytest
from omegaconf import OmegaConf
import torch
import tempfile

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel


BATCH_SIZE = 2
UP_FACTOR_LIST = [2, 4]
HEAD_LAYERS_LIST = [3, 7]
STEM_RESOLUTION = 128


@pytest.fixture
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model.generator.superres=False
    experiment_config.dataset.common.img_resolution=STEM_RESOLUTION
    experiment_config.model.generator.stem.resolution=STEM_RESOLUTION
    yield experiment_config


def save_tmp_stem_checkpoint(model_save_path, model):
    stem_checkpoint = {'state_dict': model.state_dict()}
    torch.save(stem_checkpoint, model_save_path)


@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
def test_stylegan_model(_test_exp_spec, superres):
    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm = None)

    if (superres==True):
        tmp_obj = tempfile.TemporaryDirectory()
        root_dir = tmp_obj.name
        for i in range(len(UP_FACTOR_LIST)):
            # Create a temp dir to store temp checkpoint for super resolution stylegan
            # Save temp checkpoint
            model_save_path = os.path.join(root_dir, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm = None)

        # Clean up created temp dir
        tmp_obj.cleanup()
