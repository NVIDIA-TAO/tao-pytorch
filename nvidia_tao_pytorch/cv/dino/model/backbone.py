# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Backbone modules. """

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from typing import Optional
from typing import Dict, List

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import (
    interpolate_pos_embed, interpolate_patch_embed
)
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.backbone_v2.resnet import resnet_34, resnet_50
from nvidia_tao_pytorch.cv.backbone_v2.vit_codetr import vit_large_codetr
from nvidia_tao_pytorch.cv.dino.model.fan import fan_model_dict
from nvidia_tao_pytorch.cv.dino.model.vision_transformer.vit_adapter import vit_model_dict
from nvidia_tao_pytorch.cv.dino.model.utils import dino_parser, ptm_adapter
from nvidia_tao_pytorch.cv.grounding_dino.model.swin_transformer import swin_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.efficientvit import (
    efficientvit_b0, efficientvit_b1, efficientvit_b2, efficientvit_b3
)

# ViTDet backbones that include SFP neck and produce all feature levels.
_VITDET_MODEL_DICT = {
    'vit_large_codetr': vit_large_codetr,
}

efficientvit_model_dict = {
    'efficientvit_b0': efficientvit_b0,
    'efficientvit_b1': efficientvit_b1,
    'efficientvit_b2': efficientvit_b2,
    'efficientvit_b3': efficientvit_b3,
}


class BackboneBase(nn.Module):
    """BackboneBase class."""

    def __init__(self,
                 model_name,
                 backbone: nn.Module,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_indices: list,
                 export: bool,
                 missing_keys: list):
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

        # Handling timm/efficientvit cases
        if isinstance(xs, list):
            new_xs = {}
            for i, x in enumerate(xs):
                new_xs[f'layer{i}'] = x
            xs = new_xs

        out: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            mask = F.interpolate(masks.float(), size=x.shape[-2:])
            mask = mask.to(torch.bool)
            out[name] = (x, mask)
        return out


class Backbone(BackboneBase):
    """Backbone for DINO."""

    def __init__(self, name: str,
                 pretrained_backbone_path: Optional[str],
                 train_backbone: bool,
                 resolution: int,
                 return_interm_indices: list,
                 dilation: bool,
                 export: bool,
                 activation_checkpoint: bool):
        """Initialize the Backbone Class.

        Args:
            pretrained_backbone_path (str): optional path to the pretrained backbone.
            train_backbone (bool): flag whether we want to train the backbone or not.
            resolution (int): input resolution for ViT models.
            return_interm_indices (list): list of layer indices to reutrn as backbone features.
            dilation (bool): flag whether we can to use dilation or not.
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

        supported_arch = list(fan_model_dict.keys()) + \
            list(vit_model_dict.keys()) + \
            list(swin_model_dict.keys()) + \
            list(_VITDET_MODEL_DICT.keys()) + \
            ["resnet_34", "resnet_50"] + \
            ['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3']
        # TODO: use core utils to load pretrained weights
        pretrained_backbone_ckp = None
        if pretrained_backbone_path:
            pretrained_backbone_ckp = load_pretrained_weights(
                pretrained_backbone_path,
                parser=dino_parser,
                ptm_adapter=ptm_adapter
            )

        freeze_at = None
        freeze_norm = False
        if not train_backbone:
            freeze_at = "all"
        elif train_backbone and name.startswith('resnet'):
            freeze_at = [0]

        if name == 'resnet_34':
            backbone = resnet_34(
                num_classes=0,
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                export=export,
            )
            num_channels = np.array([64, 128, 256, 512])[return_interm_indices]
        elif name == 'resnet_50':
            backbone = resnet_50(
                num_classes=0,
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                export=export,
            )
            num_channels = np.array([256, 512, 1024, 2048])[return_interm_indices]
        elif name.startswith('fan'):
            if name not in fan_model_dict:
                raise NotImplementedError(f"{name} is not supported FAN backbone. "
                                          f"Supported architecutres: {fan_model_dict.keys()}")
            backbone = fan_model_dict[name](
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                export=export
            )
            num_channels = np.array(backbone.out_channels)[return_interm_indices]
        elif name.startswith('efficientvit'):
            backbone = efficientvit_model_dict[name](
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                out_indices=return_interm_indices,
                export=export
            )
            num_channels = np.array(backbone.out_channels)[return_interm_indices - 1]
        elif name in _VITDET_MODEL_DICT:
            backbone = _VITDET_MODEL_DICT[name](
                activation_checkpoint=activation_checkpoint,
            )
            num_channels = np.array(backbone.num_features)
        elif name.startswith('vit'):
            if name not in vit_model_dict:
                raise NotImplementedError(f"{name} is not supported ViT-Adapter backbone. "
                                          f"Supported architecutres: {vit_model_dict.keys()}")

            pretrained_backbone_ckp = interpolate_vit_checkpoint(checkpoint=pretrained_backbone_ckp,
                                                                 target_patch_size=16,
                                                                 target_resolution=resolution)

            backbone = vit_model_dict[name](
                out_indices=return_interm_indices,
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                resolution=resolution,
                activation_checkpoint=activation_checkpoint
            )
            num_channels = np.array([backbone.embed_dim] * len(return_interm_indices))
        elif 'swin' in name:
            if name not in swin_model_dict:
                raise NotImplementedError(f"{name} is not supported Swin backbone. "
                                          f"Supported architecutres: {swin_model_dict.keys()}")
            backbone = swin_model_dict[name](
                out_indices=return_interm_indices,
                freeze_at=freeze_at, freeze_norm=freeze_norm,
                activation_checkpoint=activation_checkpoint
            )
            num_channels = np.array(backbone.num_features)[return_interm_indices]
        else:
            raise NotImplementedError(f"Backbone {name} is not implemented. Supported architectures {supported_arch}")

        missing_keys = None
        if pretrained_backbone_ckp:
            _tmp_st_output = backbone.load_state_dict(pretrained_backbone_ckp, strict=False)
            missing_keys = list(_tmp_st_output[0])
            for n, parameter in backbone.named_parameters():
                if n in missing_keys:
                    parameter.requires_grad_(True)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logging.info(f"{_tmp_st_output}")

        super().__init__(name, backbone, train_backbone, num_channels, return_interm_indices, export, missing_keys)


class Joiner(nn.Sequential):
    """Joiner Class."""

    def __init__(self, backbone):
        """Initialize the Joiner Class.

        Args:
            backbone (nn.Module): backbone module.

        """
        super().__init__(backbone)
        self.num_channels = backbone.num_channels

    def forward(self, input_tensors):
        """Forward function for Joiner to prepare the backbone output into transformer input format.

        Args:
            input_tensors (torch.Tensor): input tensor.

        Returns:
            out (List[Tensor]): list of tensor (feature vectors from backbone).

        """
        xs = self[0](input_tensors)
        out: List[torch.Tensor] = []
        for _, x in sorted(xs.items()):
            out.append(x)
        return out


def interpolate_vit_checkpoint(checkpoint, target_patch_size, target_resolution):
    """ Interpolate ViT backbone position embedding and patch embedding

    Args:
        checkpoint: pretrained ViT checkpoint
        target_patch_size: target patch size to interpolate to. ex: 14, 16, etc
        target_resolution: target image size to interpolate to. ex: 224, 512, 518, etc

    Returns:
        interpolated model checkpoints

    """
    if checkpoint is None:
        return checkpoint

    logging.info("Do ViT pretrained backbone interpolation")
    # interpolate patch embedding
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)

    # interpolate pos embedding
    checkpoint = interpolate_pos_embed(checkpoint_model=checkpoint,
                                       new_resolution=target_resolution,
                                       new_patch_size=target_patch_size)
    return checkpoint
