# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" DINOv2 ViT Model Module """

from functools import partial

from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import vit_large_patch14_dinov2_swiglu


vit_model_dict = {
    'vit_large_nvdinov2': partial(vit_large_patch14_dinov2_swiglu, num_classes=0)
}
