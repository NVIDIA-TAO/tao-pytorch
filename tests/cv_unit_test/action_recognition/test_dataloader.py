# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
import shutil
from PIL import Image
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.action_recognition.default_config import ARAugmentationConfig, ARModelConfig
from nvidia_tao_pytorch.cv.action_recognition.dataloader.build_data_loader import (list_dataset, get_clips_list,
                                                                build_joint_augmentation_pipeline,
                                                                build_single_augmentation_pipeline,
                                                                build_single_sampler,
                                                                build_joint_sampler,
                                                                build_dataloader)
from nvidia_tao_pytorch.cv.action_recognition.dataloader.frame_sampler import (random_interval_sample,
                                                            random_consecutive_sample,
                                                            test_interval_sample,
                                                            test_consecutive_sample,
                                                            joint_random_interval_sample,
                                                            joint_random_consecutive_sample,
                                                            joint_test_interval_sample,
                                                            joint_test_consecutive_sample)


@pytest.fixture
def _test_dir():
    img = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.uint8)
    seq_len = 32
    tmp_im = [Image.fromarray(img) for _ in range(seq_len)]
    tmp_top_dir = "tmp_test_data_dir/"
    a_video_dir = os.path.join(tmp_top_dir, "a", "001", "rgb")
    b_video_dir = os.path.join(tmp_top_dir, "b", "002", "rgb")
    if not os.path.exists(a_video_dir):
        os.makedirs(a_video_dir)
    if not os.path.exists(b_video_dir):
        os.makedirs(b_video_dir)
    for idx, img in enumerate(tmp_im):
        img.save(os.path.join(a_video_dir, str(idx) + ".png"))
    for idx, img in enumerate(tmp_im):
        img.save(os.path.join(b_video_dir, str(idx) + ".png"))

    yield tmp_top_dir
    shutil.rmtree(tmp_top_dir)


@pytest.fixture
def _test_sample_dict(_test_dir):
    sample_dict = list_dataset(_test_dir)
    yield sample_dict


@pytest.fixture
def _test_spec(_test_dir):
    aug_config = OmegaConf.structured(ARAugmentationConfig())
    model_config = OmegaConf.structured(ARModelConfig())
    spec = {"augmentation_config": aug_config,
            "model": model_config}
    yield spec


@pytest.mark.cv_unit
def test_list_dataset(_test_dir):
    sample_dict = list_dataset(_test_dir)
    assert len(sample_dict.keys()) == 2


@pytest.mark.cv_unit
@pytest.mark.parametrize("seq_len, eval_mode, sampler_strategy, sample_rate, assert_len",
                         [(32, "all", "consecutive", 1, 2),
                          (16, "all", "consecutive", 2, 4),
                          (1, "conv", "consecutive", 1, 2),
                          (1, "all", "consecutive", 1, 64),
                          (1, "all", "random_interval", 1, 64),
                          (1, "conv", "random_interval", 1, 64)])
def test_get_clips_list(_test_sample_dict, seq_len, eval_mode,
                        sampler_strategy, sample_rate, assert_len):
    sample_clips_path, _ = get_clips_list(_test_sample_dict,
                                          seq_len,
                                          eval_mode,
                                          sampler_strategy,
                                          sample_rate)

    assert len(sample_clips_path) == assert_len


@pytest.mark.cv_unit
@pytest.mark.parametrize("train_crop_type, scales, horizontal_flip_prob, val_center_crop,\
                         crop_smaller_edge, dataset_mode",
                         [("random_crop", [1], 0.5, True, 256, "train"),
                          ("multi_scale_crop", [0.8, 1], 0.5, True, 256, "train"),
                          ("no_crop", [1], 0.0, True, 256, "train"),
                          ("no_crop", [1], 0.0, False, 256, "val"),
                          ("no_crop", [1], 0.0, True, 256, "val")])
def test_build_joint_augmentation_pipeline(_test_spec, train_crop_type, scales,
                                           horizontal_flip_prob, val_center_crop,
                                           crop_smaller_edge, dataset_mode
                                           ):
    aug_config = _test_spec["augmentation_config"]
    aug_config.train_crop_type = train_crop_type
    aug_config.scales = scales
    aug_config.horizontal_flip_prob = horizontal_flip_prob
    aug_config.val_center_crop = val_center_crop
    aug_config.crop_smaller_edge = crop_smaller_edge
    build_joint_augmentation_pipeline((224, 224), aug_config, dataset_mode)


@pytest.mark.cv_unit
@pytest.mark.parametrize("train_crop_type, scales, horizontal_flip_prob, val_center_crop,\
                         crop_smaller_edge, dataset_type, dataset_mode",
                         [("random_crop", [1], 0.5, True, 256, "rgb", "train"),
                          ("multi_scale_crop", [0.8, 1], 0.5, True, 256, "of", "train"),
                          ("no_crop", [1], 0.0, True, 256, "rgb", "train"),
                          ("no_crop", [1], 0.0, False, 256, "of", "val"),
                          ("no_crop", [1], 0.0, True, 256, "rgb", "val")])
def test_build_single_augmentation_pipeline(_test_spec, train_crop_type, scales,
                                            horizontal_flip_prob, val_center_crop,
                                            crop_smaller_edge, dataset_type, dataset_mode):
    aug_config = _test_spec["augmentation_config"]
    aug_config.train_crop_type = train_crop_type
    aug_config.scales = scales
    aug_config.horizontal_flip_prob = horizontal_flip_prob
    aug_config.val_center_crop = val_center_crop
    aug_config.crop_smaller_edge = crop_smaller_edge
    build_single_augmentation_pipeline((224, 224), aug_config, dataset_type, dataset_mode)


@pytest.mark.cv_unit
@pytest.mark.parametrize("sampler_strategy, sample_rate, all_frames_3d",
                         [("consecutive", 1, False),
                          ("random_interval", 1, False),
                          ("consecutive", 1, True)])
def test_build_single_sampler(sampler_strategy, sample_rate, all_frames_3d):
    build_single_sampler(sampler_strategy, sample_rate, all_frames_3d)


@pytest.mark.cv_unit
@pytest.mark.parametrize("sampler_strategy, sample_rate, all_frames_3d",
                         [("consecutive", 1, False),
                          ("random_interval", 1, False),
                          ("consecutive", 1, True)])
def test_build_joint_sampler(sampler_strategy, sample_rate, all_frames_3d):
    build_joint_sampler(sampler_strategy, sample_rate, all_frames_3d)


@pytest.mark.cv_unit
@pytest.mark.parametrize("dataset_mode, input_type, eval_mode, model_type",
                         [("train", "2d", "center", "of"),
                          ("val", "3d", "conv", "rgb"),
                          ("val", "3d", "all", "joint"),
                          ("inf", "3d", "center", "rgb")])
def test_build_dataloader(_test_spec, _test_sample_dict, dataset_mode,
                          input_type, eval_mode, model_type):
    aug_config = _test_spec["augmentation_config"]
    model_config = _test_spec["model"]
    model_config.model_type = model_type
    build_dataloader(sample_dict=_test_sample_dict,
                     model_config=model_config,
                     output_shape=(224, 224),
                     label_map={"a": 0, "b": 1},
                     augmentation_config=aug_config,
                     dataset_mode=dataset_mode,
                     input_type=input_type,
                     eval_mode=eval_mode)


@pytest.mark.cv_unit
@pytest.mark.parametrize("max_cnt, seq_len, assert_len",
                         [(32, 16, 16),
                          (8, 16, 16)])
def test_random_interval_sample(max_cnt, seq_len, assert_len):
    sample_ids = list(random_interval_sample(max_cnt, seq_len))

    assert len(sample_ids) == assert_len


@pytest.mark.cv_unit
@pytest.mark.parametrize("max_cnt, seq_len, sample_rate, assert_len",
                         [(32, 16, 1, 16),
                          (8, 16, 1, 16),
                          (32, 16, 4, 16)])
def test_random_consecutive_sample(max_cnt, seq_len, sample_rate, assert_len):
    sample_ids = list(random_consecutive_sample(max_cnt, seq_len, sample_rate))

    assert len(sample_ids) == assert_len


@pytest.mark.cv_unit
def test_test_interval_sample():
    sample_ids = list(test_interval_sample(32, 16))
    expected_ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    assert expected_ids == sample_ids


@pytest.mark.cv_unit
def test_test_consecutive_sample():
    sample_ids = list(test_consecutive_sample(32, 16, 2))
    expected_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    assert sample_ids == expected_ids

    sample_ids = list(test_consecutive_sample(16, 16, 1, True))
    expected_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert sample_ids == expected_ids

    sample_ids = list(test_consecutive_sample(32, 16, 1))
    expected_ids = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    assert sample_ids == expected_ids


@pytest.mark.cv_unit
@pytest.mark.parametrize("max_cnt, rgb_seq_len, of_seq_len",
                         [(32, 16, 16),
                          (32, 4, 16),
                          (32, 16, 4)])
def test_joint_random_interval_sample(max_cnt, rgb_seq_len, of_seq_len):
    rgb_ids, of_ids = joint_random_interval_sample(max_cnt, rgb_seq_len, of_seq_len)
    assert len(rgb_ids) == rgb_seq_len
    assert len(of_ids) == of_seq_len


@pytest.mark.cv_unit
@pytest.mark.parametrize("max_cnt, rgb_seq_len, of_seq_len",
                         [(32, 16, 16),
                          (32, 4, 16),
                          (32, 16, 4)])
def test_joint_random_consecutive_sample(max_cnt, rgb_seq_len, of_seq_len):
    rgb_ids, of_ids = joint_random_consecutive_sample(max_cnt, rgb_seq_len, of_seq_len)
    assert len(rgb_ids) == rgb_seq_len
    assert len(of_ids) == of_seq_len


@pytest.mark.cv_unit
def test_joint_test_interval_sample():
    rgb_ids, of_ids = joint_test_interval_sample(32, 4, 16)
    expected_rgb_ids = [5, 13, 21, 29]
    expected_of_ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    assert list(rgb_ids) == expected_rgb_ids
    assert list(of_ids) == expected_of_ids

    rgb_ids, of_ids = joint_test_interval_sample(32, 16, 4)
    expected_of_ids = [5, 13, 21, 29]
    expected_rgb_ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    assert list(rgb_ids) == expected_rgb_ids
    assert list(of_ids) == expected_of_ids


@pytest.mark.cv_unit
def test_joint_test_consecutive_sample():
    rgb_ids, of_ids = joint_test_consecutive_sample(32, 4, 16)
    expected_rgb_ids = [14, 15, 16, 17]
    expected_of_ids = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    assert list(rgb_ids) == expected_rgb_ids
    assert list(of_ids) == expected_of_ids

    rgb_ids, of_ids = joint_test_consecutive_sample(32, 16, 4)
    expected_of_ids = [14, 15, 16, 17]
    expected_rgb_ids = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    assert list(rgb_ids) == expected_rgb_ids
    assert list(of_ids) == expected_of_ids
