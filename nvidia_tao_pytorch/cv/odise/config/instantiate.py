# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from detectron2.config import instantiate


def instantiate_odise(cfg):
    """Instantiate model from config file."""
    backbone = instantiate(cfg.backbone)

    cfg.sem_seg_head.pixel_decoder.input_shape = {
        k: v for k, v in backbone.output_shape().items() if k in ["res2", "res3", "res4", "res5"]
    }
    cfg.sem_seg_head.input_shape = {
        k: v for k, v in backbone.output_shape().items() if k in ["res2", "res3", "res4", "res5"]
    }
    cfg.backbone = backbone
    model = instantiate(cfg)

    return model
