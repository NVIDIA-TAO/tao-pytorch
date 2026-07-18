# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: hitting a CUDA error when running all tests

# import shutil
# from nvidia_tao_pytorch.cv.segformer.utils.config import MMSegmentationConfig
# import pytest
# from omegaconf import OmegaConf
# import os
# import numpy as np
# from PIL import Image

# from mmengine.config import Config
# from mmengine.runner import Runner
# from mmengine.dist import get_rank, get_world_size

# from nvidia_tao_pytorch.config.segformer.default_config import SFModelConfig, ExperimentConfig, seg_class
# from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create, check_and_delete

# # Triggers build of custom modules
# from nvidia_tao_pytorch.cv.segformer.model import * # noqa pylint: disable=W0401, W0614
# from nvidia_tao_pytorch.cv.segformer.dataloader import * # noqa pylint: disable=W0401, W0614

# tmp_top_dir = "tests/cv_unit_test/segformer/tmp_test_data_dir/"
# tmp_results_dir = "tests/cv_unit_test/segformer/tmp_results/"

# @pytest.fixture
# def _tmp_img_dir():
#     if not os.path.exists(tmp_top_dir):
#         os.makedirs(tmp_top_dir)
#     tmp_img_dir = os.path.join(tmp_top_dir, "images")
#     check_and_create(tmp_img_dir)
#     test_data = np.random.rand(512, 512, 3) * 255
#     test_data = test_data.astype(np.uint8)
#     im = Image.fromarray(test_data)
#     # This and the mask need to have the same filename
#     im.save(os.path.join(tmp_img_dir, "test.png"))
#     return tmp_img_dir

# @pytest.fixture
# def _tmp_mask_dir():
#     if not os.path.exists(tmp_top_dir):
#         os.makedirs(tmp_top_dir)
#     tmp_mask_dir = os.path.join(tmp_top_dir, "masks")
#     check_and_create(tmp_mask_dir)
#     mask_data = np.zeros((512, 512))
#     verts = np.array([[0, 10], 
#                       [10, 0], 
#                       [10, 20]])
#     row_indices = verts[:, 0]
#     col_indices = verts[:, 1]
#     mask_data[row_indices, col_indices] = 1
#     # print(mask_data)
#     mask_data = mask_data.astype(np.uint8)
#     mask = Image.fromarray(mask_data)
#     mask.save(os.path.join(tmp_mask_dir, "test.png"))
#     return tmp_mask_dir

# @pytest.fixture
# def _test_exp_spec(_tmp_img_dir, _tmp_mask_dir):
#     experiment_config = OmegaConf.structured(ExperimentConfig())

#     experiment_config["dataset"]["data_root"] = ""
#     experiment_config["dataset"]["train_dataset"]["img_dir"] = [_tmp_img_dir]
#     experiment_config["dataset"]["train_dataset"]["ann_dir"] = [_tmp_mask_dir]
#     experiment_config["dataset"]["val_dataset"]["img_dir"] = _tmp_img_dir
#     experiment_config["dataset"]["val_dataset"]["ann_dir"] = _tmp_mask_dir
#     experiment_config["dataset"]["test_dataset"]["img_dir"] = _tmp_img_dir
#     experiment_config["dataset"]["test_dataset"]["ann_dir"] = _tmp_mask_dir
#     experiment_config["dataset"]["batch_size"] = 1
#     experiment_config["dataset"]["workers_per_gpu"] = 8

#     palette = [seg_class()]
#     palette_cfg = OmegaConf.create(palette)
#     experiment_config["dataset"]["palette"] = palette_cfg

#     experiment_config["train"]["max_iters"] = 2
#     experiment_config["train"]["validate"] = True

#     yield experiment_config

# @pytest.mark.cv_unit
# @pytest.mark.parametrize("backbone",
#                          [("mit_b0"),
#                           ("mit_b1"),
#                           ("mit_b2"),
#                           ("mit_b3"),
#                           ("mit_b4"),
#                           ("mit_b5"),
#                           ("fan_tiny_8_p4_hybrid"),
#                           ("fan_large_16_p4_hybrid"),
#                           ("fan_small_12_p4_hybrid"),
#                           ("fan_base_16_p4_hybrid")
#                           ])
# def test_model_train(_test_exp_spec, backbone):

#     _test_exp_spec["model"]["backbone"]["type"] = backbone
#     # @sean there's some funny business going on with how mmengine stores its results
#     # nothing is breaking, but maybe worth a cleanup in the future
#     _test_exp_spec["results_dir"] = f"{tmp_results_dir}_{backbone}"

#     mmseg_config = MMSegmentationConfig(_test_exp_spec, phase="train")
#     train_cfg = mmseg_config.updated_config

#     train_cfg["work_dir"] = f"{tmp_results_dir}_{backbone}"

#     # Converts dict to cfg
#     # (This is necessary due to a bug in mmseg for model.test_cfg which errors if it's a dict)
#     train_cfg = Config(train_cfg)

#     # @sean for some reason these are necessary when using pytest, but not when running normally
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(0)
#     if "RANK" not in os.environ:
#         os.environ['RANK'] = str(get_rank())
#     if "WORLD_SIZE" not in os.environ:
#         os.environ['WORLD_SIZE'] = str(get_world_size())
#     if "MASTER_PORT" not in os.environ:
#         os.environ['MASTER_PORT'] = str(_test_exp_spec["train"]["exp_config"]["MASTER_PORT"])
#     if "MASTER_ADDR" not in os.environ:
#         os.environ['MASTER_ADDR'] = _test_exp_spec["train"]["exp_config"]["MASTER_ADDR"]

#     runner = Runner.from_cfg(train_cfg)
#     runner.train()

#     # check_and_delete(_tmp_top_dir)
#     # check_and_delete(_tmp_results_dir)

# @pytest.fixture(scope="session", autouse=True)
# def cleanup(request):
#     """Cleanup a testing directory once we are finished."""
#     def remove_test_dir():
#         shutil.rmtree(tmp_top_dir)
#         shutil.rmtree(tmp_results_dir)
#     request.addfinalizer(remove_test_dir)
