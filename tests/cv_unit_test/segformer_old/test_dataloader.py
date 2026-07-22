# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# import os
# import pytest
# import pickle
# import numpy as np
# from omegaconf import OmegaConf
# from PIL import Image
# import datetime
# from nvidia_tao_pytorch.cv.segformer.utils import get_root_logger
# from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create, check_and_delete
# from nvidia_tao_pytorch.config.segformer.default_config import SFDatasetExpConfig, seg_class
# from nvidia_tao_pytorch.cv.segformer.dataloader.segformer_dm import SFDataModule
# from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataloader
# from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataset

# tmp_top_dir = "tests/cv_unit_test/segformer/tmp_test_data_dir/"

# @pytest.fixture
# def _test_dir():
#     if not os.path.exists(tmp_top_dir):
#         os.makedirs(tmp_top_dir)
#     tmp_img_dir = os.path.join(tmp_top_dir, "images")
#     tmp_mask_dir = os.path.join(tmp_top_dir, "masks")
#     check_and_create(tmp_img_dir)
#     check_and_create(tmp_mask_dir)
#     assert(os.path.isdir(tmp_img_dir))
#     test_data = np.random.rand(320, 320, 3) * 255
#     test_data = test_data.astype(np.uint8)
#     mask_data = np.zeros((320, 320))
#     verts = np.array([[0, 2], [2, 0], [2, 4]])
#     row_indices = verts[:, 0]
#     col_indices = verts[:, 1]
#     mask_data[row_indices, col_indices] = 1
#     mask_data = mask_data.astype(np.uint8)
#     im = Image.fromarray(test_data)
#     im.save(os.path.join(tmp_img_dir, "test.jpg"))
#     mask = Image.fromarray(test_data)
#     mask.save(os.path.join(tmp_mask_dir, "test.png"))
#     log_file = os.path.join(tmp_top_dir, 'log_train_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

#     yield tmp_top_dir, log_file
#     check_and_delete(tmp_top_dir)

# @pytest.fixture
# def _test_logger():
#     check_and_create(tmp_top_dir)
#     log_file = os.path.join(tmp_top_dir, 'log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
#     logger = get_root_logger(log_file, "INFO")
#     yield logger

# @pytest.fixture
# def _test_data_spec():
#     images_path = os.path.join(tmp_top_dir, "images")
#     masks_path = os.path.join(tmp_top_dir, "masks")
#     sc = seg_class()
#     palette = [sc]
#     cfg = OmegaConf.create(palette)
#     data_config = OmegaConf.structured(SFDatasetExpConfig())
#     data_config.palette = cfg
#     data_config["palette"] = cfg
#     data_config["train_dataset"]["img_dir"] = [images_path]
#     data_config["train_dataset"]["ann_dir"] = [masks_path]
#     data_config["val_dataset"]["img_dir"] = images_path
#     data_config["val_dataset"]["ann_dir"] = masks_path
#     data_config["data_root"] = ""

#     yield data_config


# @pytest.mark.cv_unit
# def test_build_dataloader(_test_dir, _test_logger, _test_data_spec):
#     # Get the default spec for the rest of the parameters
#     dm = SFDataModule(_test_data_spec, 1, 49, _test_logger, phase="train")
#     dm.setup()
#     dataset = [build_dataset(dm.train_data, dm.default_args)]
#     dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
#     data_loaders = [
#         build_dataloader(
#             dataset,
#             dm.samples_per_gpu,
#             dm.workers_per_gpu,
#             dm.num_gpus,
#             dist=True,
#             seed=dm.seed,
#             drop_last=True) for ds in dataset
#     ]
#     check_and_delete(tmp_top_dir)
