# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" NVCLIP Model Module """

from nvidia_tao_pytorch.cv.segformer_old.model.backbones.vision_transformer.vit_adapter import (
    TIMMTransformerWrapper,
    OpenCLIPTransformerWrapper,
    ViTAdapter
)
from nvidia_tao_core.config.classification_pyt.model_params_mapping import map_clip_model_cfg
import torch.nn as nn
import open_clip as OpenCLIP
from mmseg.registry import MODELS


@MODELS.register_module()
class vit_base_nvclip_16_siglip(nn.Module):
    """ViT-Base NVCLIP model.."""

    def __init__(self, out_indices=[0, 1, 2, 3], resolution=(1024, 1024), init_cfg=None, **kwargs):
        """ViT-Base NVCLIP model.

        Args:
            out_indices (list, optional): List of block indices to return as feature.. Defaults to [0, 1, 2, 3].
            resolution (tuple, optional): flag to indicate if activation checkpoint is used.. Defaults to (1024, 1024).
            init_cfg (dict, optional): initial config. Defaults to None.
        """
        super().__init__()

        if init_cfg and init_cfg["checkpoint"]:
            pretrained_backbone_ckp = init_cfg["checkpoint"]
        else:
            pretrained_backbone_ckp = None

        clip_vitb16 = OpenCLIP.create_model("ViT-B-16-SigLIP", pretrained=pretrained_backbone_ckp)
        timm_vit_model = clip_vitb16.visual.trunk

        # setup for dynamic input
        timm_vit_model.patch_embed.strict_img_size = False

        self.nvclip_vit_adapter = ViTAdapter(vit_model=TIMMTransformerWrapper(timm_vit_model),
                                             conv_inplane=56,
                                             n_points=4,
                                             drop_path_rate=0.4,
                                             deform_num_heads=16,
                                             init_values=1e-5,
                                             interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                                             cffn_ratio=0.25,
                                             deform_ratio=0.5,
                                             out_indices=out_indices,
                                             resolution=resolution[0])

    def forward(self, x):
        """Forward function"""
        return self.nvclip_vit_adapter(x)


@MODELS.register_module()
class vit_huge_nvclip_14_siglip(nn.Module):
    """ViT-Base NVCLIP model."""

    def __init__(self, out_indices=[0, 1, 2, 3], resolution=(1024, 1024), init_cfg=None, **kwargs):
        """ViT-huge NVCLIP model.

        Args:
            out_indices (list, optional): List of block indices to return as feature.. Defaults to [0, 1, 2, 3].
            resolution (tuple, optional): input resolution. Defaults to (1024, 1024).
            init_cfg (dict, optional): initial config. Defaults to None.
        """
        super().__init__()

        # Handle customized clip config
        OpenCLIP.factory._MODEL_CONFIGS["ViT-H-14-SigLIP-CLIPA-224"] = map_clip_model_cfg["ViT-H-14-SigLIP-CLIPA-224"]

        if init_cfg and init_cfg["checkpoint"]:
            pretrained_backbone_ckp = init_cfg["checkpoint"]
        else:
            pretrained_backbone_ckp = None

        clip_vith14 = OpenCLIP.create_model("ViT-H-14-SigLIP-CLIPA-224", pretrained=pretrained_backbone_ckp)
        timm_vit_model = clip_vith14.visual

        self.nvclip_vit_adapter = ViTAdapter(vit_model=OpenCLIPTransformerWrapper(timm_vit_model),
                                             conv_inplane=56,
                                             n_points=4,
                                             drop_path_rate=0.4,
                                             deform_num_heads=16,
                                             init_values=1e-5,
                                             interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
                                             cffn_ratio=0.25,
                                             deform_ratio=0.5,
                                             out_indices=out_indices,
                                             resolution=resolution[0])

    def forward(self, x):
        """Forward function"""
        return self.nvclip_vit_adapter(x)
