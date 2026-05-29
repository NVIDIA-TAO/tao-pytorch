# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet Classification model builder"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import interpolate_patch_embed, interpolate_pos_embed
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import DINOV2
from nvidia_tao_pytorch.cv.visual_changenet.backbone.dino_v2 import vit_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.fan import fan_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.radio import radio_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.utils import ptm_adapter, visual_changenet_parser
from nvidia_tao_pytorch.cv.visual_changenet.backbone.vit_adapter import vit_adapter_model_dict, ViTAdapter
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet_utils import (
    MLP,
    conv_diff,
    resize,
)


class SingleHeadAttention(nn.Module):
    """
    Single Head Attention Module
    """

    def __init__(self, embed_dim):
        """
        Initialize Single Head Attention Module.

        Args:
            embed_dim: Dimension of the input embeddings.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Compute the scaling factor and register it as a buffer.
        scaling = 1.0 / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.register_buffer("scaling", scaling)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [L, B, D]
        Returns:
            Output tensor of shape [L, B, D]
        """
        x = x.transpose(0, 1)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q * self.scaling
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, V)

        output = self.W_o(attn_output)

        output = output.transpose(0, 1)
        return output


class ChangeNetClassifyDecoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 feature_strides=[2, 4, 8, 16], model_name='FANHybrid',
                 output_shape=[128, 128], embed_dec=30, learnable_difference_modules=4, num_golden=1):
        """
        Initialize Visual ChangeNetClassifyDecoder.

        Args:
            input_transform (str): Input transformation strategy.
            in_index (list): Indices of input features to use.
            align_corners (bool): Align corners flag for resizing.
            in_channels (list): Input channels for each feature level.
            embedding_dim (int): Embedding dimension for decoder.
            output_nc (int): Number of output classes.
            feature_strides (list): Feature stride for each level.
            model_name (str): Backbone model name.
            output_shape (list): Output shape of the model.
            embed_dec (int): Embedding dimension for the final classifier.
            learnable_difference_modules (int): Number of decoder difference modules for Architecture 2.
            num_golden (int): Number of golden sample.
        """
        super(ChangeNetClassifyDecoder, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels), f"Feature strides {feature_strides} is not equal to the in_channels {in_channels}"
        assert min(feature_strides) == feature_strides[0], f"First feature stride from {feature_strides} is not the minimum {min(feature_strides)}"

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.model_name = model_name
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.learnable_difference_modules = learnable_difference_modules
        assert num_golden > 0, "Number of golden images must be greater than 0"
        self.num_golden = num_golden

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # Multi-golden Fusion Modules
        if num_golden > 1:
            self.attn_c1 = SingleHeadAttention(self.embedding_dim)
            self.attn_c2 = SingleHeadAttention(self.embedding_dim)
            self.attn_c3 = SingleHeadAttention(self.embedding_dim)
            self.attn_c4 = SingleHeadAttention(self.embedding_dim)
            self.position_embedding_c1 = nn.Parameter(torch.zeros(1, self.embedding_dim, (output_shape[0] // self.feature_strides[0] // 7), (output_shape[1] // self.feature_strides[0]) // 7))
            self.position_embedding_c2 = nn.Parameter(torch.zeros(1, self.embedding_dim, (output_shape[0] // self.feature_strides[1] // 7), (output_shape[1] // self.feature_strides[1]) // 7))
            self.position_embedding_c3 = nn.Parameter(torch.zeros(1, self.embedding_dim, (output_shape[0] // self.feature_strides[2] // 7), (output_shape[1] // self.feature_strides[2]) // 7))
            self.position_embedding_c4 = nn.Parameter(torch.zeros(1, self.embedding_dim, (output_shape[0] // self.feature_strides[3] // 7), (output_shape[1] // self.feature_strides[3]) // 7))
            self.bn1 = nn.BatchNorm2d(self.embedding_dim)
            self.bn2 = nn.BatchNorm2d(self.embedding_dim)
            self.bn3 = nn.BatchNorm2d(self.embedding_dim)
            self.bn4 = nn.BatchNorm2d(self.embedding_dim)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(nn.Conv2d(in_channels=self.embedding_dim * learnable_difference_modules,
                                                   out_channels=self.embedding_dim, kernel_size=1),
                                         nn.BatchNorm2d(self.embedding_dim)
                                         )

        self.active = nn.Sigmoid()
        # Image level classification

        self.dim_output = output_shape[0] // self.feature_strides[0]
        self.dim_output1 = output_shape[1] // self.feature_strides[0]
        self.in_channel_dim = self.embedding_dim * self.dim_output * self.dim_output1

        self.classifier = nn.Sequential(
            nn.Linear(self.in_channel_dim, embed_dec),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dec),
            nn.Linear(embed_dec, embed_dec),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dec),
            nn.Linear(embed_dec, self.output_nc)
        )

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        """
        Forward pass of the Visual ChangeNetClassifyDecoder.
        """
        # Transforming encoder features (select layers)
        # Just takes the 4 encoder feature maps as give in comment below
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        # img1 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        # MLP decoder on C1-C4
        B, _, _, _ = c4_1.shape
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(B, -1, c4_1.shape[2], c4_1.shape[3])
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(B, -1, c3_1.shape[2], c3_1.shape[3])
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(B, -1, c2_1.shape[2], c2_1.shape[3])
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(B, -1, c1_1.shape[2], c1_1.shape[3])

        B, C, H, W = _c1_1.size()
        N = self.num_golden

        if self.num_golden > 1:
            # Difference modules
            x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

            # img2 features
            c1_2, c2_2, c3_2, c4_2 = x_2
            _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(N * B, -1, c4_2.shape[2], c4_2.shape[3])
            _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(N * B, -1, c3_2.shape[2], c3_2.shape[3])
            _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(N * B, -1, c2_2.shape[2], c2_2.shape[3])
            _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(N * B, -1, c1_2.shape[2], c1_2.shape[3])

            _c4_1 = _c4_1.unsqueeze(0).repeat(N, 1, 1, 1, 1).view(N * B, -1, c4_1.shape[2], c4_1.shape[3])
            _c3_1 = _c3_1.unsqueeze(0).repeat(N, 1, 1, 1, 1).view(N * B, -1, c3_1.shape[2], c3_1.shape[3])
            _c2_1 = _c2_1.unsqueeze(0).repeat(N, 1, 1, 1, 1).view(N * B, -1, c2_1.shape[2], c2_1.shape[3])
            _c1_1 = _c1_1.unsqueeze(0).repeat(N, 1, 1, 1, 1).view(N * B, -1, c1_1.shape[2], c1_1.shape[3])

            _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
            if self.feature_strides[-2] == self.feature_strides[-1]:
                _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + _c4
            else:
                _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
            _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
            _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

            # Diff features
            _c4 = _c4.view(N, B, -1, _c4.shape[2], _c4.shape[3])
            _c3 = _c3.view(N, B, -1, _c3.shape[2], _c3.shape[3])
            _c2 = _c2.view(N, B, -1, _c2.shape[2], _c2.shape[3])
            _c1 = _c1.view(N, B, -1, _c1.shape[2], _c1.shape[3])

            # Diff features aggregation
            N, B, C, H, W = _c1.size()

            # Add position embedding, share the same position embedding for in the window (window size = 7)
            # B * C * N * H * W
            _c4 = _c4.permute(1, 2, 0, 3, 4) + self.position_embedding_c4.unsqueeze(2).repeat(1, 1, 1, 7, 7)
            _c3 = _c3.permute(1, 2, 0, 3, 4) + self.position_embedding_c3.unsqueeze(2).repeat(1, 1, 1, 7, 7)
            _c2 = _c2.permute(1, 2, 0, 3, 4) + self.position_embedding_c2.unsqueeze(2).repeat(1, 1, 1, 7, 7)
            _c1 = _c1.permute(1, 2, 0, 3, 4) + self.position_embedding_c1.unsqueeze(2).repeat(1, 1, 1, 7, 7)

            # Reshape for window attention
            # B * C * N * H * W  -> B * C * N * H//win * win * W//win * win
            _c4 = _c4.reshape(B, C, N, H // 8 // 7, 7, W // 8 // 7, 7)
            _c3 = _c3.reshape(B, C, N, H // 4 // 7, 7, W // 4 // 7, 7)
            _c2 = _c2.reshape(B, C, N, H // 2 // 7, 7, W // 2 // 7, 7)
            _c1 = _c1.reshape(B, C, N, H // 7, 7, W // 7, 7)

            # B * C * N * H//win * win * W//win * win  -> (B*win*win) * C * (N*H//win*W//win)
            _c4 = _c4.permute(0, 4, 6, 1, 2, 3, 5).reshape(B * 7 * 7, C, -1)
            _c3 = _c3.permute(0, 4, 6, 1, 2, 3, 5).reshape(B * 7 * 7, C, -1)
            _c2 = _c2.permute(0, 4, 6, 1, 2, 3, 5).reshape(B * 7 * 7, C, -1)
            _c1 = _c1.permute(0, 4, 6, 1, 2, 3, 5).reshape(B * 7 * 7, C, -1)

            # (B*win*win) * C * (N*H//win*W//win) -> (N*H//win*W//win) * (B*win*win) * C
            _c4 = _c4.permute(2, 0, 1)
            _c3 = _c3.permute(2, 0, 1)
            _c2 = _c2.permute(2, 0, 1)
            _c1 = _c1.permute(2, 0, 1)

            # Window attention + skip connection
            _c4 = _c4 + self.attn_c4(_c4)
            _c3 = _c3 + self.attn_c3(_c3)
            _c2 = _c2 + self.attn_c2(_c2)
            _c1 = _c1 + self.attn_c1(_c1)

            # Reshape to original shape
            # (N*H//win*W//win) * (B*win*win)  * C
            _c4, _ = _c4.view(N, H // 8 // 7, W // 8 // 7, B, 7, 7, C).max(dim=0)
            _c4 = _c4.permute(2, 5, 0, 3, 1, 4).reshape(B, C, H // 8, W // 8)
            _c3, _ = _c3.view(N, H // 4 // 7, W // 4 // 7, B, 7, 7, C).max(dim=0)
            _c3 = _c3.permute(2, 5, 0, 3, 1, 4).reshape(B, C, H // 4, W // 4)
            _c2, _ = _c2.view(N, H // 2 // 7, W // 2 // 7, B, 7, 7, C).max(dim=0)
            _c2 = _c2.permute(2, 5, 0, 3, 1, 4).reshape(B, C, H // 2, W // 2)
            _c1, _ = _c1.view(N, H // 7, W // 7, B, 7, 7, C).max(dim=0)
            _c1 = _c1.permute(2, 5, 0, 3, 1, 4).reshape(B, C, H, W)

            _c1 = self.bn1(_c1)
            _c2 = self.bn2(_c2)
            _c3 = self.bn3(_c3)
            _c4 = self.bn4(_c4)

            _c4_up = resize(_c4, size=[H, W], mode='bilinear', align_corners=False)
            _c3_up = resize(_c3, size=[H, W], mode='bilinear', align_corners=False)
            _c2_up = resize(_c2, size=[H, W], mode='bilinear', align_corners=False)
            difference_modules = tuple([_c4_up, _c3_up, _c2_up, _c1][:self.learnable_difference_modules])
            # Linear Fusion of difference feature map from all scales
            _c = self.linear_fuse(torch.cat(difference_modules, dim=1))

        else:
            x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

            # img2 features
            c1_2, c2_2, c3_2, c4_2 = x_2
            _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(B, -1, c4_2.shape[2], c4_2.shape[3])
            _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(B, -1, c3_2.shape[2], c3_2.shape[3])
            _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(B, -1, c2_2.shape[2], c2_2.shape[3])
            _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(B, -1, c1_2.shape[2], c1_2.shape[3])

            _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
            if self.feature_strides[-2] == self.feature_strides[-1]:
                _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + _c4
            else:
                _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
            _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
            _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

            _c4_up = resize(_c4, size=[H, W], mode='bilinear', align_corners=False)
            _c3_up = resize(_c3, size=[H, W], mode='bilinear', align_corners=False)
            _c2_up = resize(_c2, size=[H, W], mode='bilinear', align_corners=False)

            difference_modules = tuple([_c4_up, _c3_up, _c2_up, _c1][:self.learnable_difference_modules])
            # Linear Fusion of difference image from all scales
            _c = self.linear_fuse(torch.cat(difference_modules, dim=1))
        # For Visual ChangeNet Classifier
        _c_flatten = _c.reshape(_c.size(0), -1)
        output = self.classifier(_c_flatten)

        return output


# Visual ChangeNetClassify:
class ChangeNetClassify(nn.Module):
    """
    Visual ChangeNetClassify model for change classification using a Transformer Encoder and Decoder.

    Args:
        input_nc (int): Number of input channels (default is 3).
        output_nc (int): Number of output classes (default is 2).
        embed_dim (int): Embedding dimension for the Transformer Encoder (default is 256).
        model (str): Backbone model name (default is 'fan_tiny_8_p4_hybrid_256').
        embed_dims (list): List of embedding dimensions for the Transformer Decoder (default is [128, 256, 384, 384]).
        feature_strides (list): List of feature strides for the Transformer Decoder (default is [4, 8, 16, 16]).
        in_index (list): List of input indices for the Transformer Decoder (default is [0, 1, 2, 3]).
        output_shape: Image input shape to the model as output by the dataloader after augmentation
        embedding_vectors: For architecture 1, the dimension for the last embedding vector for each input image for computing eulidean distance
        embed_dec: For architecture 2, the embedding dimension of the output MLP.
        feat_downsample: Flag to indicate whether to use feature downsampling for the transformer block for FAN-Hybrid
        learnable_difference_modules (int): Number of decoder difference modules for Architecture 2
        pretrained_backbone_path (str): Path to the pre-trained backbone weights.
        activation_checkpoint (bool): Enable activation checkpointing
        freeze_backbone: Flag to freeze backbone weights during training.
        return_interm_indices (list): list of layer indices to reutrn as backbone features.
        use_summary_token (bool): Use summary token of backone.
        num_golden (int): Number of golden sample.
        export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=2,
        embed_dim=256,
        model="fan_tiny_8_p4_hybrid_256",
        embed_dims=[128, 256, 384, 384],
        feature_strides=[4, 8, 16, 16],
        in_index=[0, 1, 2, 3],
        difference_module="learnable",
        output_shape=[128, 128],
        embedding_vectors=5,
        embed_dec=30,
        feat_downsample=False,
        return_interm_indices=[0, 1, 2, 3],
        learnable_difference_modules=4,
        pretrained_backbone_path=None,
        activation_checkpoint=False,
        freeze_backbone=False,
        use_summary_token=True,
        num_golden=1,
        export=False,
    ):
        """Initialize Visual ChangeNetSegment class"""
        super(ChangeNetClassify, self).__init__()

        # Index 4 is not part of the backbone but taken from index 3 with conv 3x3 stride 2
        return_interm_indices = [r for r in return_interm_indices if r != 4]
        return_interm_indices = np.array(return_interm_indices)
        if not np.logical_and(return_interm_indices >= 0, return_interm_indices <= 4).all():
            raise ValueError(f"Invalid range for return_interm_indices. "
                             f"Provided return_interm_indices is {return_interm_indices}.")

        if len(np.unique(return_interm_indices)) != len(return_interm_indices):
            raise ValueError(f"Duplicate index in the provided return_interm_indices: {return_interm_indices}")

        # Transformer Encoder
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.model_name = model
        self.embed_dims = embed_dims
        self.difference_module = difference_module
        self.use_summary_token = use_summary_token
        self.num_golden = num_golden

        freeze_at = None
        if freeze_backbone:
            if pretrained_backbone_path is None:
                raise ValueError("You shouldn't freeze a model without specifying pretrained_backbone_path")
            freeze_at = "all"

        if get_global_rank() == 0:
            logger.info(f"Number of output classes: {output_nc}")

        if 'fan' in self.model_name:
            assert (output_shape[0] % feature_strides[-1] == 0) and (output_shape[1] % feature_strides[-1] == 0), 'Input image size must be a multiple of 16'
            assert num_golden == 1, f"Multiple golden samples is not supported for backbone [{self.model_name}]"
            self.backbone = fan_model_dict[self.model_name](
                feat_downsample=feat_downsample,
                freeze_at=freeze_at,
                export=export,
            )
        elif 'radio' in self.model_name:
            assert output_shape[0] == output_shape[1], 'ViT Backbones only support square input image where input_width == input_height'
            if self.difference_module == 'learnable':
                self.backbone = vit_adapter_model_dict[self.model_name](
                    out_indices=return_interm_indices,
                    resolution=output_shape[0],
                    activation_checkpoint=activation_checkpoint,
                    use_summary_token=use_summary_token,
                    freeze_at=freeze_at,
                    export=export,
                )
            elif self.difference_module == 'euclidean':
                self.backbone = radio_model_dict[self.model_name](
                    resolution=[224, 224],
                    freeze_at=freeze_at,
                    export=export,
                )
        elif 'vit' in self.model_name:
            assert output_shape[0] == output_shape[1], 'ViT Backbones only support square input image where input_width == input_height'
            if self.difference_module == 'learnable':
                self.backbone = vit_adapter_model_dict[self.model_name](
                    out_indices=return_interm_indices,
                    resolution=output_shape[0],
                    activation_checkpoint=activation_checkpoint,
                    freeze_at=freeze_at,
                    export=export,
                )
            elif self.difference_module == 'euclidean':
                self.backbone = vit_model_dict[self.model_name](freeze_at=freeze_at, export=export)
        else:
            raise NotImplementedError('Bacbkbone name [%s] is not supported' % self.model_name)

        if pretrained_backbone_path:
            state_dict = load_pretrained_weights(
                pretrained_backbone_path,
                parser=visual_changenet_parser,
                ptm_adapter=ptm_adapter,
            )
            if isinstance(self.backbone, ViTAdapter):
                state_dict = interpolate_vit_checkpoint(
                    checkpoint=state_dict,
                    target_patch_size=16,
                    target_resolution=output_shape[0],
                )
            elif isinstance(self.backbone, DINOV2):
                state_dict = interpolate_vit_checkpoint(
                    checkpoint=state_dict,
                    target_patch_size=14,
                    target_resolution=518,
                )
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            if get_global_rank() == 0:
                logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logger.info(f"{msg}")

        if self.difference_module == 'learnable':
            # Transformer Decoder
            self.decoder = ChangeNetClassifyDecoder(input_transform='multiple_select', in_index=in_index, align_corners=False,
                                                    in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                                    feature_strides=feature_strides, model_name=self.model_name,
                                                    output_shape=output_shape, embed_dec=embed_dec,
                                                    learnable_difference_modules=learnable_difference_modules, num_golden=self.num_golden)

        elif self.difference_module == 'euclidean':
            assert num_golden == 1, "Multiple golden samples is not supported for Difference module [euclidean]"
            self.dim_output = output_shape[0] // feature_strides[-1]
            self.dim_output1 = output_shape[1] // feature_strides[-1]
            self.fc_ip_dim = self.embed_dims[-1] * self.dim_output * self.dim_output1
            if 'radio' in self.model_name:
                self.fc_ip_dim = self.embed_dims[-1] * len(self.backbone.radio.radio.summary_idxs)
            elif 'vit' in self.model_name:
                self.fc_ip_dim = self.embed_dims[-1]

            self.embedding = embedding_vectors
            self.fc1 = nn.Sequential(
                nn.Linear(self.fc_ip_dim, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, self.embedding))

        else:
            raise NotImplementedError('Difference module [%s] is not supported' % self.difference_module)

    def forward_once(self, x):
        """
        Forward pass of the Visual ChangeNetClassify model for a single image when using Euclidean Distance metric

        Args:
            x (torch.Tensor): Input tensor for the image input.

        Returns:
            torch.Tensor: Embedding vector for the given input image
        """
        if 'fan' in self.model_name:
            output = self.backbone.forward_feature_pyramid(x)[-1]
        elif 'vit' in self.model_name:
            output = self.backbone(x)
        output = output.reshape(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, x1, x2):
        """
        Forward pass of the ChangeNetClassify model.

        Args:
            x1 (torch.Tensor): Input tensor for the first image input.
            x2 (torch.Tensor): Input tensor for the second image input.

        Returns:
            torch.Tensor: Returns the output tensor (For Arch1: two embedding vectors for 2 input images, Arch2: single output vector of dimension Batch_size x num_classes)
        """
        if 'vit' in self.model_name:
            assert x1.shape[2] == x1.shape[3], f"{self.model_name} backbone only supports square input images. " \
                "Please make sure the input height and width are equal, use an N x N grid to combine multiple lighting conditions."
            if self.difference_module == 'learnable':
                assert x1.shape[2] % 32 == 0 and x1.shape[3] % 32 == 0, "Input image size must be a multiple of 32 for ViT-Adapter"
        if self.difference_module == 'euclidean':
            # Classifier after last FM with eucledian distance:
            output1 = self.forward_once(x1)
            output2 = self.forward_once(x2)
            output = F.pairwise_distance(output1, output2)

        elif self.difference_module == 'learnable':
            if self.num_golden == 1:
                [fx1, fx2] = [self.backbone.forward_feature_pyramid(x1), self.backbone.forward_feature_pyramid(x2)]
                out_decoder = self.decoder(fx1, fx2)
            else:
                fx1 = self.backbone.forward_feature_pyramid(x1)
                B, N, C, H, W = x2.shape

                # Only compute the gradient for the first golden sample
                fx2s = self.backbone.forward_feature_pyramid(x2[:, 0])
                with torch.no_grad():
                    x2_no_grad = x2[:, 1:]
                    x2_no_grad = x2_no_grad.permute(1, 0, 2, 3, 4)
                    fx2s_no_grad = self.backbone.forward_feature_pyramid(x2_no_grad.reshape((N - 1) * B, C, H, W))
                for i in range(len(fx2s)):
                    fx2s[i] = torch.cat((fx2s[i].unsqueeze(0), fx2s_no_grad[i].view(N - 1, B, fx2s_no_grad[i].shape[1], fx2s_no_grad[i].shape[2], fx2s_no_grad[i].shape[3])), dim=0)
                    fx2s[i] = fx2s[i].view(N * B, fx2s_no_grad[i].shape[1], fx2s_no_grad[i].shape[2],  fx2s_no_grad[i].shape[3])
                out_decoder = self.decoder(fx1, fx2s)
            output = out_decoder
        else:
            raise NotImplementedError('Only option 1 and 2 are supported')
        return output


def build_model(experiment_config,
                export=False):
    """ Build Visual ChangeNet classification model according to configuration

    Args:
        experiment_config: experiment configuration.
        export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset.classify

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    feat_downsample = model_config.backbone.feat_downsample
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    channels_map = {"fan_tiny_8_p4_hybrid": [128, 256, 192, 192],
                    "fan_large_16_p4_hybrid": [128, 256, 480, 480],
                    "fan_small_12_p4_hybrid": [128, 256, 384, 384],
                    "fan_base_16_p4_hybrid": [128, 256, 448, 448],
                    "vit_large_nvdinov2": [1024, 1024, 1024, 1024],
                    "c_radio_p1_vit_huge_patch16_224_mlpnorm": [1280, 1280, 1280, 1280],
                    "c_radio_p2_vit_huge_patch16_224_mlpnorm": [1280, 1280, 1280, 1280],
                    "c_radio_p3_vit_huge_patch16_224_mlpnorm": [1280, 1280, 1280, 1280],
                    "c_radio_v2_vit_base_patch16_224": [768, 768, 768, 768],
                    "c_radio_v2_vit_large_patch16_224": [1024, 1024, 1024, 1024],
                    "c_radio_v2_vit_huge_patch16_224": [1280, 1280, 1280, 1280]
                    }

    if backbone in channels_map:
        embed_dims = channels_map[backbone]
        model_config.decode_head.in_channels = embed_dims
    else:
        raise NotImplementedError('Backbone name [%s] is not supported' % backbone)

    embed_dim = model_config.decode_head.decoder_params['embed_dim']
    feature_strides = model_config.decode_head.feature_strides
    in_index = model_config.decode_head.in_index
    use_summary_token = model_config.decode_head.use_summary_token
    num_golden = dataset_config.num_golden

    num_classes = dataset_config.num_classes
    image_width = dataset_config.image_width
    image_height = dataset_config.image_height
    num_input = dataset_config.num_input
    concat_type = dataset_config.concat_type
    grid_map = dataset_config.grid_map
    embedding_vectors = model_config.classify.embedding_vectors
    embed_dec = model_config.classify.embed_dec

    learnable_difference_modules = model_config.classify.learnable_difference_modules
    assert 1 <= learnable_difference_modules <= 4, "Visual ChangeNet only supports learnable difference modules in the range [1,4]"
    difference_module = model_config.classify.difference_module

    output_shape = [image_height, image_width]
    if concat_type == 'linear':
        output_shape = [image_height, image_width * num_input]
    elif concat_type == 'grid':
        output_shape = [image_height * grid_map.y, image_width * grid_map.x]

    return ChangeNetClassify(
        embed_dim=embed_dim,
        model=backbone,
        output_nc=num_classes,
        input_nc=3,
        embed_dims=embed_dims,
        feature_strides=feature_strides,
        in_index=in_index,
        output_shape=output_shape,
        embedding_vectors=embedding_vectors,
        embed_dec=embed_dec,
        feat_downsample=feat_downsample,
        learnable_difference_modules=learnable_difference_modules,
        difference_module=difference_module,
        pretrained_backbone_path=pretrained_backbone_path,
        freeze_backbone=freeze_backbone,
        use_summary_token=use_summary_token,
        num_golden=num_golden,
        export=export,
    )


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

    if get_global_rank() == 0:
        logger.info("Do ViT pretrained backbone interpolation")
    # interpolate patch embedding
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)

    # interpolate pos embedding
    checkpoint = interpolate_pos_embed(checkpoint_model=checkpoint,
                                       new_resolution=target_resolution,
                                       new_patch_size=target_patch_size)
    return checkpoint
