# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Backbone modules. """

import numpy as np
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2.resnet import resnet_50
from nvidia_tao_pytorch.cv.grounding_dino.model.swin_transformer import swin_model_dict
from nvidia_tao_pytorch.cv.grounding_dino.model.utils import grounding_dino_parser, ptm_adapter


class BackboneBase(nn.Module):
    """BackboneBase class."""

    def __init__(self,
                 model_name: str,
                 backbone: nn.Module,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_indices: list,
                 export: bool):
        """Initialize the Backbone Base Class.

        Args:
            model_name (str): backbone model name.
            backbone (nn.Module): backbone torch module.
            train_backbone (bool): flag whether we want to train the backbone or not.
            num_channels (int): channel size.
            return_interm_indices (list): list of layer indices to reutrn as backbone features.
            export (bool): flag to indicate whehter exporting to onnx or not.
        """
        super().__init__()
        self.export = export
        self.model_name = model_name
        self.body = backbone
        self.num_channels = num_channels
        self.return_interm_indices = return_interm_indices

    def forward(self, input_tensors):
        """Forward function for Backboone base.

        Args:
            input_tensors (torch.Tensor): input tensor.

        Returns:
            out (torch.Tensor): output tensor.
        """
        if self.export:
            batch_shape = input_tensors.shape
            dtype = input_tensors.dtype
            device = input_tensors.device
            # when exporting, the input shape is fixed and no padding mask is needed.
            masks = torch.zeros((batch_shape[0], 1, batch_shape[2], batch_shape[3]), dtype=dtype, device=device)
            input_tensor = input_tensors
        else:
            masks = input_tensors[:, 3:4]
            input_tensor = input_tensors[:, :3]

        xs = self.body.forward_feature_pyramid(input_tensor)

        out: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            mask = F.interpolate(masks.float(), size=x.shape[-2:])
            mask = mask.to(torch.bool)
            out[name] = (x, mask)
        return out


class Backbone(BackboneBase):
    """Backbone for Grounding DINO."""

    def __init__(self, name: str,
                 pretrained_backbone_path: Optional[str],
                 train_backbone: bool,
                 return_interm_indices: list,
                 export: bool,
                 activation_checkpoint: bool):
        """Initialize the Backbone Class.

        Args:
            pretrained_backbone_path (str): optional path to the pretrained backbone.
            train_backbone (bool): flag whether we want to train the backbone or not.
            return_interm_indices (list): list of layer indices to reutrn as backbone features.
            export (bool): flag to indicate whehter exporting to onnx or not.
            activation_checkpoint (bool): flag to indicate whether to run activation checkpointing during training.

        Raises:
            ValueError: If return_interm_indices does not have valid range or has duplicate index.
            NotImplementedError: If invalid backbone name was provided.
        """
        return_interm_indices = np.array(return_interm_indices)
        if not np.logical_and(return_interm_indices >= 0, return_interm_indices <= 4).all():
            raise ValueError(f"Invalid range for return_interm_indices. "
                             f"Provided return_interm_indices is {return_interm_indices}.")
        if len(np.unique(return_interm_indices)) != len(return_interm_indices):
            raise ValueError(f"Duplicate index in the provided return_interm_indices: {return_interm_indices}")

        supported_arch = list(swin_model_dict.keys()) + ['resnet_50']

        freeze_at = None
        freeze_norm = False
        if not train_backbone:
            freeze_at = "all"
        elif train_backbone and name.startswith('resnet'):
            freeze_at = [0]

        if name == 'resnet_50':
            backbone = resnet_50(
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                export=export,
            )
            num_channels_all = np.array([256, 512, 1024, 2048])
            num_channels = num_channels_all[return_interm_indices]
        elif 'swin' in name:
            if name not in swin_model_dict:
                raise NotImplementedError(f"{name} is not supported Swin backbone. "
                                          f"Supported architecutres: {swin_model_dict.keys()}")
            backbone = swin_model_dict[name](
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                activation_checkpoint=activation_checkpoint
            )
            num_channels_all = np.array(backbone.num_features)
            num_channels = num_channels_all[return_interm_indices]
        else:
            raise NotImplementedError(f"Backbone {name} is not implemented. Supported architectures {supported_arch}")

        if pretrained_backbone_path:
            checkpoint = load_pretrained_weights(
                pretrained_backbone_path,
                parser=grounding_dino_parser,
                ptm_adapter=ptm_adapter
            )
            _tmp_st_output = backbone.load_state_dict(checkpoint, strict=False)
            if get_global_rank() == 0:
                logging.info("Loaded pretrained weights from %s \n %s", pretrained_backbone_path, _tmp_st_output)

        super().__init__(name, backbone, train_backbone, num_channels, return_interm_indices, export)
