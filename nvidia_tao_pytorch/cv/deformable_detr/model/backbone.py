# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Backbone modules. """

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import Optional, Dict
import numpy as np

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.cv.deformable_detr.model.resnet import resnet50
from nvidia_tao_pytorch.cv.deformable_detr.model.gc_vit import gc_vit_model_dict


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed."""

    def __init__(self, n, eps=1e-5):
        """Initialize the FrozenBatchNorm2d Class.

        Args:
            n (int): num_features from an expected input of size
            eps (float): a value added to the denominator for numerical stability.
        """
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Load paremeters from state dict. """
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """Forward function: move reshapes to the beginning to make it fuser-friendly.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output of Frozen Batch Norm.
        """
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        rv = self.running_var.view(1, -1, 1, 1)
        rm = self.running_mean.view(1, -1, 1, 1)
        eps = self.eps
        scale = 1 / (rv + eps).sqrt()
        scale = w * scale
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """BackboneBase class."""

    def __init__(self, model_name, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list, export: bool):
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

        if model_name == 'resnet_50':
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)

            return_layers = {}
            # 4 scale: {'layer2': '1', 'layer3': '2', 'layer4': '3'}
            # 5 scale: {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
            for layer_index in return_interm_indices:
                return_layers.update({"layer{}".format(layer_index + 1): "{}".format(layer_index)})
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        elif 'fan' in model_name or 'gc_vit' in model_name:
            for name, parameter in backbone.named_parameters():
                if not train_backbone:
                    parameter.requires_grad_(False)
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

        xs = self.body(input_tensor)

        out: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            mask = F.interpolate(masks.float(), size=x.shape[-2:])
            mask = mask.to(torch.bool)
            out[name] = (x, mask)
        return out


class Backbone(BackboneBase):
    """Backbone for D-DETR."""

    def __init__(self, name: str,
                 pretrained_backbone_path: Optional[str],
                 train_backbone: bool,
                 return_interm_indices: list,
                 dilation: bool,
                 export: bool,
                 activation_checkpoint: bool):
        """Initialize the Backbone Class.

        Args:
            pretrained_backbone_path (str): optional path to the pretrained backbone.
            train_backbone (bool): flag whether we want to train the backbone or not.
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

        if name == 'resnet_50':
            if export:
                _norm_layer = nn.BatchNorm2d
            else:
                _norm_layer = FrozenBatchNorm2d

            backbone = resnet50(norm_layer=_norm_layer,
                                replace_stride_with_dilation=[False, False, dilation])
            num_channels_all = np.array([256, 512, 1024, 2048])
            num_channels = num_channels_all[return_interm_indices]
        elif 'gc_vit' in name:
            if name not in gc_vit_model_dict:
                raise NotImplementedError(f"{name} is not supported GCViT backbone. "
                                          f"Supported architecutres: {gc_vit_model_dict.keys()}")
            backbone = gc_vit_model_dict[name](out_indices=return_interm_indices,
                                               activation_checkpoint=activation_checkpoint)
            num_channels_all = np.array(backbone.num_features)
            num_channels = num_channels_all[return_interm_indices]
        else:
            supported_arch = list(gc_vit_model_dict.keys()) + ["resnet_50"]
            raise NotImplementedError(f"Backbone {name} is not implemented. Supported architectures {supported_arch}")

        if pretrained_backbone_path:
            checkpoint = load_pretrained_weights(pretrained_backbone_path)
            _tmp_st_output = backbone.load_state_dict(checkpoint, strict=False)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logging.info(f"{_tmp_st_output}")

        super().__init__(name, backbone, train_backbone, num_channels, return_interm_indices, export)


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
        return list(xs.values())
