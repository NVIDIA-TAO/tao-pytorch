# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.re_identification.default_config import ReIDModelConfig, ReIDTrainExpConfig
from nvidia_tao_pytorch.cv.re_identification.model.build_nn_model import build_model


@pytest.fixture
def _test_experiment_spec():
    model_config = OmegaConf.structured(ReIDModelConfig())
    train_config = OmegaConf.structured(ReIDTrainExpConfig())
    experiment_config = {"model": model_config, "train": train_config}
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth",
                         [("resnet_18", 2048, "triplet", False, True, True),
                          ("resnet_34", 64, "center", True, False, False),
                          ("resnet_50", 32, "triplet_center", True, True, True)
                         ])
def test_resnet_model(_test_experiment_spec, backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].feat_dim = feat_dim
    _test_experiment_spec["model"].metric_loss_type = metric_loss_type
    _test_experiment_spec["model"].with_center_loss = with_center_loss
    _test_experiment_spec["model"].with_flip_feature = with_flip_feature
    _test_experiment_spec["model"].label_smooth = label_smooth
    build_model(_test_experiment_spec, 751)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type",
                         [("swin_base_patch4_window7_224", 1024, "triplet_center", True, True, True, True, "softmax"),
                          ("swin_base_patch4_window7_224", 1024, "triplet_center", True, True, True, True, "arcface"),
                          ("swin_base_patch4_window7_224", 1024, "triplet", True, True, True, True, "cosface"),
                          ("swin_base_patch4_window7_224", 1024, "triplet", True, True, True, True, "amsoftmax"),
                          ("swin_base_patch4_window7_224", 1024, "triplet", True, True, True, True, "circle"),
                         ])
def test_swin_base_model(_test_experiment_spec, backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].feat_dim = feat_dim
    _test_experiment_spec["model"].metric_loss_type = metric_loss_type
    _test_experiment_spec["model"].with_center_loss = with_center_loss
    _test_experiment_spec["model"].with_flip_feature = with_flip_feature
    _test_experiment_spec["model"].label_smooth = label_smooth
    _test_experiment_spec["model"].cos_layer = cos_layer
    _test_experiment_spec["model"].id_loss_type = id_loss_type
    build_model(_test_experiment_spec, 751)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type",
                         [("swin_small_patch4_window7_224", 768, "triplet_center", True, True, True, True, "softmax"),
                          ("swin_small_patch4_window7_224", 768, "triplet_center", True, True, True, True, "arcface"),
                          ("swin_small_patch4_window7_224", 768, "triplet", True, True, True, True, "cosface"),
                          ("swin_small_patch4_window7_224", 768, "triplet", True, True, True, True, "amsoftmax"),
                          ("swin_small_patch4_window7_224", 768, "triplet", True, True, True, True, "circle"),
                         ])
def test_swin_small_model(_test_experiment_spec, backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].feat_dim = feat_dim
    _test_experiment_spec["model"].metric_loss_type = metric_loss_type
    _test_experiment_spec["model"].with_center_loss = with_center_loss
    _test_experiment_spec["model"].with_flip_feature = with_flip_feature
    _test_experiment_spec["model"].label_smooth = label_smooth
    _test_experiment_spec["model"].cos_layer = cos_layer
    _test_experiment_spec["model"].id_loss_type = id_loss_type
    build_model(_test_experiment_spec, 751)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type",
                         [("swin_tiny_patch4_window7_224", 768, "triplet_center", True, True, True, True, "softmax"),
                          ("swin_tiny_patch4_window7_224", 768, "triplet_center", True, True, True, True, "arcface"),
                          ("swin_tiny_patch4_window7_224", 768, "triplet", True, True, True, True, "cosface"),
                          ("swin_tiny_patch4_window7_224", 768, "triplet", True, True, True, True, "amsoftmax"),
                          ("swin_tiny_patch4_window7_224", 768, "triplet", True, True, True, True, "circle"),
                         ])
def test_swin_tiny_model(_test_experiment_spec, backbone, feat_dim, metric_loss_type, with_center_loss, with_flip_feature, label_smooth, cos_layer, id_loss_type):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].feat_dim = feat_dim
    _test_experiment_spec["model"].metric_loss_type = metric_loss_type
    _test_experiment_spec["model"].with_center_loss = with_center_loss
    _test_experiment_spec["model"].with_flip_feature = with_flip_feature
    _test_experiment_spec["model"].label_smooth = label_smooth
    _test_experiment_spec["model"].cos_layer = cos_layer
    _test_experiment_spec["model"].id_loss_type = id_loss_type
    build_model(_test_experiment_spec, 751)
