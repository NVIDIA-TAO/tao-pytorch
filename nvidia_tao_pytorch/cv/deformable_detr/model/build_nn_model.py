# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build nn module model."""

import torch.nn as nn

from nvidia_tao_pytorch.cv.deformable_detr.model.backbone import Backbone, Joiner
from nvidia_tao_pytorch.cv.deformable_detr.model.position_encoding import PositionEmbeddingSine, PositionEmbeddingSineExport
from nvidia_tao_pytorch.cv.deformable_detr.model.deformable_transformer import DeformableTransformer
from nvidia_tao_pytorch.cv.deformable_detr.model.deformable_detr_base import DeformableDETR


class DDModel(nn.Module):
    """Deformable DETR model module."""

    def __init__(self,
                 num_classes=4,
                 hidden_dim=256,
                 pretrained_backbone_path=None,
                 backbone='resnet_50',
                 train_backbone=True,
                 num_feature_levels=4,
                 return_interm_indices=[1, 2, 3, 4],
                 nheads=8,
                 enc_layers=6,
                 dec_layers=6,
                 dim_feedforward=1024,
                 dec_n_points=4,
                 enc_n_points=4,
                 num_queries=300,
                 aux_loss=True,
                 with_box_refine=True,
                 dilation=False,
                 dropout_ratio=0.3,
                 export=False,
                 export_format='onnx',
                 activation_checkpoint=True):
        """Initialize D-DETR Model.

        Args:
            num_classes (int): number of classes for the model.
            hidden_dim (int): size of the hidden dimension.
            pretrained_backbone_path (str): pretrained backbone path.
                If not provided, train from scratch.
            backbone (str): type of backbone architecture.
            train_backbone (bool): whether to train backbone or not.
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            return_interm_indices (list): indices of feature level to use.
            nheads (int): number of heads.
            enc_layers (int): number of encoder layers.
            dec_layers (int): number of decoder layers.
            dim_feedforward (int): dimension of the feedforward layer.
            dec_n_points (int): number of reference points in the decoder.
            enc_n_points (int): number of reference points in the encoder.
            num_queries (int): number of queries to be used in D-DETR encoder-decoder.
            aux_loss (bool): flag to indicate if auxiliary loss is used.
            with_box_refine (bool): flag to indicate if iterative box refinement is used.
            dilation (bool): flag to indicate if dilation is used (only for ResNet).
            dropout_ratio (float): probability for the dropout layer.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            export_format (str): format for exporting (e.g. 'onnx' or 'xdl')
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
        """
        super(__class__, self).__init__()  # pylint:disable=undefined-variable

        # build positional encoding. only support PositionEmbeddingSine
        if export:
            position_embedding = PositionEmbeddingSineExport(hidden_dim // 2, normalize=True)
        else:
            position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # build backbone
        if num_feature_levels != len(return_interm_indices):
            raise ValueError(f"num_feature_levels: {num_feature_levels} does not match the size of "
                             f"return_interm_indices: {return_interm_indices}")

        # Index 4 is not part of the backbone but taken from index 3 with conv 3x3 stride 2
        return_interm_indices = [r for r in return_interm_indices if r != 4]
        backbone_only = Backbone(backbone,
                                 pretrained_backbone_path,
                                 train_backbone,
                                 return_interm_indices,
                                 dilation,
                                 export,
                                 activation_checkpoint)

        # Keep joiner for backward compatibility
        joined_backbone = Joiner(backbone_only)

        # build tranformer
        transformer = DeformableTransformer(d_model=hidden_dim,
                                            nhead=nheads,
                                            num_encoder_layers=enc_layers,
                                            num_decoder_layers=dec_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout_ratio,
                                            activation="relu",
                                            return_intermediate_dec=True,
                                            num_feature_levels=num_feature_levels,
                                            dec_n_points=dec_n_points,
                                            enc_n_points=enc_n_points,
                                            export=export,
                                            export_format=export_format,
                                            activation_checkpoint=activation_checkpoint)

        # build deformable detr model
        self.model = DeformableDETR(joined_backbone,
                                    position_embedding,
                                    transformer,
                                    num_classes=num_classes,
                                    num_queries=num_queries,
                                    num_feature_levels=num_feature_levels,
                                    aux_loss=aux_loss,
                                    with_box_refine=with_box_refine,
                                    export=export)

    def forward(self, x):
        """model forward function"""
        x = self.model(x)
        return x


def build_model(experiment_config,
                export=False):
    """ Build deformable detr model according to configuration.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag to indicate onnx export.

    Returns:
        model (nn.Module): D-DETR model.
    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    num_classes = dataset_config.num_classes
    backbone = model_config.backbone
    dropout_ratio = model_config.dropout_ratio
    hidden_dim = model_config.hidden_dim
    num_feature_levels = model_config.num_feature_levels
    return_interm_indices = model_config.return_interm_indices
    nheads = model_config.nheads
    enc_layers = model_config.enc_layers
    dec_layers = model_config.dec_layers
    dim_feedforward = model_config.dim_feedforward
    dec_n_points = model_config.dec_n_points
    enc_n_points = model_config.enc_n_points
    num_queries = model_config.num_queries
    aux_loss = model_config.aux_loss
    with_box_refine = model_config.with_box_refine
    dilation = model_config.dilation
    train_backbone = model_config.train_backbone
    pretrained_backbone = model_config.pretrained_backbone_path
    activation_checkpoint = experiment_config.train.activation_checkpoint
    export_format = experiment_config.export.format

    model = DDModel(num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    pretrained_backbone_path=pretrained_backbone,
                    backbone=backbone,
                    train_backbone=train_backbone,
                    num_feature_levels=num_feature_levels,
                    return_interm_indices=return_interm_indices,
                    nheads=nheads,
                    enc_layers=enc_layers,
                    dec_layers=dec_layers,
                    dim_feedforward=dim_feedforward,
                    dec_n_points=dec_n_points,
                    enc_n_points=enc_n_points,
                    num_queries=num_queries,
                    aux_loss=aux_loss,
                    with_box_refine=with_box_refine,
                    dilation=dilation,
                    dropout_ratio=dropout_ratio,
                    export=export,
                    export_format=export_format,
                    activation_checkpoint=activation_checkpoint)
    return model
