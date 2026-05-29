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

"""Visual ChangeNet Segmentation model builder"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import interpolate_patch_embed, interpolate_pos_embed
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import FrozenBatchNorm2d
from nvidia_tao_pytorch.cv.visual_changenet.backbone.fan import fan_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.backbone.utils import ptm_adapter, visual_changenet_parser
from nvidia_tao_pytorch.cv.visual_changenet.backbone.vit_adapter import vit_adapter_model_dict
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet_utils import (
    MLP,
    ConvLayer,
    ResidualBlock,
    UpsampleConvLayer,
    conv_diff,
    make_prediction,
    resize,
)


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(
        self,
        input_transform="multiple_select",
        in_index=[0, 1, 2, 3],
        align_corners=True,
        in_channels=[32, 64, 128, 256],
        embedding_dim=64,
        output_nc=2,
        decoder_softmax=False,
        feature_strides=[2, 4, 8, 16],
        export=False,
        model_name="FANHybrid",
    ):
        """Initialize DecoderTransformer_v3 class

        Args:
            input_transform (str): Input transformation strategy.
            in_index (list): Indices of input features to use.
            align_corners (bool): Align corners flag for resizing.
            in_channels (list): Input channels for each feature level.
            embedding_dim (int): Embedding dimension for decoder.
            output_nc (int): Number of output classes.
            decoder_softmax (bool): Flag to indicate whether to use softmax for predictions.
            feature_strides (list): Feature stride for each level.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.
            model_name (str): Backbone model name.
        """
        super(DecoderTransformer_v3, self).__init__()
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
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.model_name = model_name

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

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

        if export:
            self._freeze_bn_norm(self)

    def _freeze_bn_norm(self, m: nn.Module):
        """Recursively freeze the batch normalization layers in the given module.

        Replaces all BatchNorm2d layers with FrozenBatchNorm2d to prevent
        their statistics from being updated during training.

        Args:
            m (nn.Module): Module to process

        Returns:
            nn.Module: Module with frozen batch normalization layers
        """
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_bn_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

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
        Forward pass of the Visual ChangeNetSegment decoder.
        """
        # Transforming encoder features (select layers)
        # Just takes the 4 encoder feature maps as give in comment below
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        # MLP decoder on C1-C4
        n, _, _, _ = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        if self.feature_strides[-1] == self.feature_strides[-2]:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + _c4
        else:
            _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")

        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


# Visual ChangeNetSegment:
class ChangeNetSegment(nn.Module):
    """
    Visual ChangeNetSegment model for semantic segmentation using a Transformer Encoder and Decoder.

    Args:
        input_nc (int): Number of input channels (default is 3).
        output_nc (int): Number of output classes (default is 2).
        decoder_softmax (bool): Whether to apply softmax to the decoder output (default is False).
        embed_dim (int): Embedding dimension for the Transformer Encoder (default is 256).
        model (str): Backbone model name (default is 'fan_tiny_8_p4_hybrid_256').
        img_size (int): Input image size (default is 256).
        embed_dims (list): List of embedding dimensions for the Transformer Decoder (default is [128, 256, 384, 384]).
        feature_strides (list): List of feature strides for the Transformer Decoder (default is [4, 8, 16, 16]).
        in_index (list): List of input indices for the Transformer Decoder (default is [0, 1, 2, 3]).
        feat_downsample: Flag to indicate whether to use feature downsampling for the transformer block for FAN-Hybrid
        learnable_difference_modules (int): Number of decoder difference modules for Architecture 2
        pretrained_backbone_path (str): Path to the pre-trained backbone weights.
        activation_checkpoint (bool): Enable activation checkpointing
        freeze_backbone: Flag to freeze backbone weights during training.
        return_interm_indices (list): list of layer indices to reutrn as backbone features.
        export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=2,
        decoder_softmax=False,
        embed_dim=256,
        model="fan_tiny_8_p4_hybrid_256",
        img_size=256,
        embed_dims=[128, 256, 384, 384],
        feature_strides=[4, 8, 16, 16],
        in_index=[0, 1, 2, 3],
        feat_downsample=False,
        pretrained_backbone_path=None,
        return_interm_indices=[0, 1, 2, 3],
        activation_checkpoint=False,
        freeze_backbone=False,
        use_summary_token=True,
        export=False,
    ):
        """Initialize Visual ChangeNetSegment class"""
        super(ChangeNetSegment, self).__init__()

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

        freeze_at = None
        if freeze_backbone:
            if pretrained_backbone_path is None:
                raise ValueError("You shouldn't freeze a model without specifying pretrained_backbone_path")
            freeze_at = "all"
        if get_global_rank() == 0:
            logger.info(f"Number of output classes: {output_nc}")
        assert img_size % feature_strides[-1] == 0, f"Input image size must be a multiple of {feature_strides[-1]}"

        if 'fan' in self.model_name:
            self.backbone = fan_model_dict[self.model_name](
                img_size=img_size,
                feat_downsample=feat_downsample,
                freeze_at=freeze_at,
                export=export,
            )
        elif 'radio' in self.model_name:
            assert img_size % 32 == 0, "Input image resolution must be a multiple of 32 for ViT-Adapter"
            self.backbone = vit_adapter_model_dict[self.model_name](
                out_indices=return_interm_indices,
                resolution=img_size,
                activation_checkpoint=activation_checkpoint,
                use_summary_token=use_summary_token,
                freeze_at=freeze_at,
                export=export,
            )
        elif 'vit' in self.model_name:
            assert img_size % 32 == 0, "Input image resolution must be a multiple of 32 for ViT-Adapter"
            self.backbone = vit_adapter_model_dict[self.model_name](
                out_indices=return_interm_indices,
                resolution=img_size,
                activation_checkpoint=activation_checkpoint,
                freeze_at=freeze_at,
                export=export,
            )
        else:
            raise NotImplementedError('Bacbkbone name [%s] is not supported' % self.model_name)

        if pretrained_backbone_path:
            pretrained_backbone_ckp = load_pretrained_weights(
                pretrained_backbone_path,
                parser=visual_changenet_parser,
                ptm_adapter=ptm_adapter,
            )
            if "vit" in self.model_name and "radio" not in self.model_name:
                pretrained_backbone_ckp = interpolate_vit_checkpoint(
                    checkpoint=pretrained_backbone_ckp,
                    target_patch_size=16,
                    target_resolution=img_size,
                )
            msg = self.backbone.load_state_dict(pretrained_backbone_ckp, strict=False)
            if get_global_rank() == 0:
                logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
                logger.info(f"{msg}")

        # Transformer Decoder
        self.decoder = DecoderTransformer_v3(
            input_transform="multiple_select",
            in_index=in_index,
            align_corners=False,
            in_channels=self.embed_dims,
            embedding_dim=self.embedding_dim,
            output_nc=output_nc,
            decoder_softmax=decoder_softmax,
            feature_strides=feature_strides,
            export=export,
            model_name=self.model_name,
        )

    def forward(self, x1, x2):
        """
        Forward pass of the Visual ChangeNetSegment model.

        Args:
            x1 (torch.Tensor): Input tensor for the first image input.
            x2 (torch.Tensor): Input tensor for the second image input.

        Returns:
            torch.Tensor: Output tensor representing the segmentation map.
        """
        [fx1, fx2] = [self.backbone.forward_feature_pyramid(x1), self.backbone.forward_feature_pyramid(x2)]
        out_decoder = self.decoder(fx1, fx2)
        return out_decoder


def build_model(experiment_config,
                export=False):
    """ Build changenet model according to configuration

    Args:
        experiment_config: experiment configuration.
        export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset.segment

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
        raise NotImplementedError('Bacbkbone name [%s] is not supported' % backbone)

    embed_dim = model_config.decode_head.decoder_params['embed_dim']
    feature_strides = model_config.decode_head.feature_strides
    in_index = model_config.decode_head.in_index
    use_summary_token = model_config.decode_head.use_summary_token

    num_classes = dataset_config.num_classes
    img_size = dataset_config.img_size

    return ChangeNetSegment(
        embed_dim=embed_dim,
        model=backbone,
        output_nc=num_classes,
        img_size=img_size,
        input_nc=3,
        decoder_softmax=False,
        embed_dims=embed_dims,
        feature_strides=feature_strides,
        in_index=in_index,
        feat_downsample=feat_downsample,
        pretrained_backbone_path=pretrained_backbone_path,
        freeze_backbone=freeze_backbone,
        use_summary_token=use_summary_token,
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
