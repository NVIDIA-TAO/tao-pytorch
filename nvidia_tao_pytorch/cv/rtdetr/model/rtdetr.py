# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" RT-DETR model. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.radio import radio_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.utils import rtdetr_parser, ptm_adapter


class RTDETR(nn.Module):
    """RT-DETR Module."""

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, frozen_fm_cfg=None, export=False):
        """Init function."""
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.frozen_fm_cfg = frozen_fm_cfg
        self.export = export
        if frozen_fm_cfg and frozen_fm_cfg.enabled:
            if "radio" in frozen_fm_cfg.backbone:
                model_name = frozen_fm_cfg.backbone

                self.frozen_radio = radio_model_dict[model_name][0](resolution=self.encoder.eval_spatial_size)
                # Freeze the backbone.
                self.frozen_radio.eval()
                for p in self.frozen_radio.parameters():
                    p.requires_grad = False
                # Load the pretrained weights.
                msg = self.frozen_radio.load_state_dict(
                    load_pretrained_weights(
                        frozen_fm_cfg.checkpoint,
                        parser=rtdetr_parser,
                        ptm_adapter=ptm_adapter
                    ),
                    strict=False,
                )
                if get_global_rank() == 0:
                    logger.info(f"Loaded frozen FM model from {frozen_fm_cfg.checkpoint} with message: {msg}")
                self.frozen_radio.float()
                self.frozen_radio.cuda()
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                for _, param in self.frozen_radio.named_parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError("The backbone of the frozen FM must be `radio` for now.")

    def forward(self, x, targets=None):
        """Forward function."""
        if self.frozen_fm_cfg and self.frozen_fm_cfg.enabled:
            b, _, h, w = x.shape
            assert h % 32 == 0 and w % 32 == 0, "The height and width of the input must be divisible by 32."
            x_norm = TF.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            h_down, w_down = h // 2, w // 2
            x_down = F.interpolate(x_norm, size=[h_down, w_down])
            with torch.no_grad():
                summary, spatial_features = self.frozen_radio.forward_pre_logits(x_down)
            spatial_features = spatial_features.view(b, int(h_down // 16), int(w_down // 16), -1).permute(0, 3, 1, 2)
            spatial_features = self.maxpool(spatial_features)

        feats = self.backbone.forward_feature_pyramid(x)
        if self.frozen_fm_cfg and self.frozen_fm_cfg.enabled:
            feats.append(spatial_features)
            x, proj_feats = self.encoder(feats)
            x = x[:-1]
            x = self.decoder(x, targets, summary.view(b, 1, -1))
        else:
            x, proj_feats = self.encoder(feats)
            x = self.decoder(x, targets)
        if not self.export:
            x['bb_feats'] = feats
            x['srcs'] = proj_feats

        return x

    def deploy(self):
        """Convert to deploy mode."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
