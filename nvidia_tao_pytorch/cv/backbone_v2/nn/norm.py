# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

"""Norm layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(nn.Module):
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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Load paremeters from state dict."""
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm 2D."""

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        data_format="channels_last",
    ):
        """Initialize the LayerNorm2d Class.

        Args:
            normalized_shape (int): The input shape from an expected input of size.
            eps (float): A value added to the denominator for numerical stability. Default: `1e-5`.
            elementwise_affine (bool): If `True`, this module has learnable per-element affine parameters. Default:
                `True`.
            bias (bool): If `True`, adds a learnable bias to the output. Default: `True`.
            data_format (str): The data format of the input tensor. Default: `channels_last`.
        """
        super().__init__(normalized_shape, eps, elementwise_affine, bias)
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"{self.data_format} is not supported. Supported formats are: channels_last, channels_first"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Both data formats route through ``F.layer_norm`` so ONNX export
        emits a single ``LayerNormalization`` op.
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        # channels_first: permute to channels-last, layer_norm, permute back.
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(),
            self.normalized_shape, self.weight, self.bias, self.eps,
        ).permute(0, 3, 1, 2).contiguous()


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim):
        """Initialize the Global Response Normalization Class.

        Args:
            dim (int): The number of features in the input tensor.
        """
        super().__init__()
        # Initialize with 4D shape [1, 1, 1, dim]
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
