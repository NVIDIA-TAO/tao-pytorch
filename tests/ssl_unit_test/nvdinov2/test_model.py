# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVDINOv2 Model Unit Tests
"""
import pytest
from omegaconf import OmegaConf
import torch

from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel

BATCH_SIZE = 2
IMAGE_CHANNEL = 3
N_GLOBAL_CROPS = 1
GLOBAL_CROPS_SIZE = 224
N_LOCAL_CROPS = 1
LOCAL_CROPS_SIZE = 98
PATCH_SIZE = 14
TEACHER_TEMPERATURE = 0.995

@pytest.fixture
def _test_batch():
    torch.manual_seed(47)
    batch = {}
    batch["global_crops"] = torch.randn(BATCH_SIZE * N_GLOBAL_CROPS, IMAGE_CHANNEL, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE)
    batch["local_crops"] = torch.randn(BATCH_SIZE * N_LOCAL_CROPS, IMAGE_CHANNEL, LOCAL_CROPS_SIZE, LOCAL_CROPS_SIZE)
    batch["global_masks"] = torch.zeros((BATCH_SIZE * N_GLOBAL_CROPS, GLOBAL_CROPS_SIZE // PATCH_SIZE * GLOBAL_CROPS_SIZE // PATCH_SIZE), dtype=torch.bool)
    batch["global_masks_indices"] = batch["global_masks"].flatten().nonzero().flatten()
    batch["global_masks_weight"] = (1 / batch["global_masks"].float().sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(batch["global_masks"])[batch["global_masks"]]
    yield batch

@pytest.mark.ssl_unit
def test_nvdionv2_model(_test_batch):
    experiment_config = OmegaConf.structured(ExperimentConfig())

    model = DinoV2PlModel(experiment_config)
    model.to(torch.float16).train().cuda()

    global_crops = _test_batch["global_crops"].to(torch.float16).cuda()
    local_crops = _test_batch["local_crops"].to(torch.float16).cuda()
    global_masks = _test_batch["global_masks"].cuda()
    global_masks_indices = _test_batch["global_masks_indices"].cuda()
    global_masks_weight = _test_batch["global_masks_weight"].to(torch.float16).cuda()
    with torch.no_grad():
        teacher_dino_centered, teacher_ibot_centered, _ = model.teacher_forward(
            global_crops=global_crops,
            global_masks_indices=global_masks_indices,
            teacher_temperature=TEACHER_TEMPERATURE,
        )
        loss = model.student_forward(
            global_crops=global_crops,
            global_masks=global_masks,
            global_masks_indices=global_masks_indices,
            global_masks_weight=global_masks_weight,
            local_crops=local_crops,
            teacher_dino_centered=teacher_dino_centered,
            teacher_ibot_centered=teacher_ibot_centered,
        )
