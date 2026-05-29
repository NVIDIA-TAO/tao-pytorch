# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build nn module model."""

import torch.nn as nn

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.rtdetr.model.hybrid_encoder import HybridEncoder
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr_decoder import RTDETRTransformer
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr import RTDETR
from nvidia_tao_pytorch.cv.rtdetr.model.utils import rtdetr_parser, ptm_adapter


class RTDETRModel(nn.Module):
    """RT-DETR model module."""

    def __init__(self,
                 backbone_name='resnet_50',
                 pretrained_backbone=None,
                 train_backbone=True,
                 num_classes=80,
                 out_indices=[1, 2, 3],
                 # Encoder
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0,
                 enc_act='gelu',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1,
                 act='silu',
                 eval_spatial_size=[640, 640],
                 # Decoder
                 feat_channels=[256, 256, 256],
                 num_levels=3,
                 num_queries=300,
                 num_decoder_layers=6,
                 num_denoising=100,
                 eval_idx=-1,
                 multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
                 activation_checkpoint=False,
                 frozen_fm_cfg=None,
                 export=False,
                 ):
        """Initialize RT-DETR Model."""
        super().__init__()
        freeze_at = None
        freeze_norm = False
        if not train_backbone:
            freeze_at = "all"
        elif pretrained_backbone and train_backbone and backbone_name.startswith('resnet'):
            freeze_at = [0]
            freeze_norm = True
        backbone = RTDETR_BACKBONE_REGISTRY.get(backbone_name)(
            out_indices,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            activation_checkpoint=activation_checkpoint,
            export=export,
        )
        in_channels = backbone.out_channels
        if pretrained_backbone:
            state_dict = load_pretrained_weights(
                pretrained_backbone,
                parser=rtdetr_parser,
                ptm_adapter=ptm_adapter
            )
            new_checkpoint = {}
            teacher_model_dict = backbone.state_dict()
            for k in sorted(teacher_model_dict.keys()):
                # k_ckpt = "model." + k
                k_ckpt = k
                v = state_dict.get(k_ckpt, None)
                if v is None:
                    logger.info(f"skip layer: {k}, {k_ckpt} doesn't exist in the pretrained model.")
                    continue
                # Handle PTL format
                # k = k.replace("model.model.", "model.")
                if v.size() == teacher_model_dict[k].size():
                    new_checkpoint[k] = v
                else:
                    # Skip layers that mismatch
                    logger.info(
                        "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                        k, list(v.size()), list(teacher_model_dict[k].size())
                    )
                    new_checkpoint[k] = teacher_model_dict[k]
            msg = backbone.load_state_dict(new_checkpoint, strict=False)
            if get_global_rank() == 0:
                logger.info(f"Loaded pretrained weights from {pretrained_backbone}")
                logger.info(f"incompatible keys: {msg}")

        encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            frozen_fm_cfg=frozen_fm_cfg,
        )

        decoder = RTDETRTransformer(
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_denoising=num_denoising,
            eval_idx=eval_idx,
            eval_spatial_size=eval_spatial_size,
            num_classes=num_classes,
            frozen_fm_cfg=frozen_fm_cfg,
            export=export,
        )

        self.model = RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            multi_scale=multi_scale,
            frozen_fm_cfg=frozen_fm_cfg,
            export=export,
        )

    def forward(self, x, targets=None):
        """model forward function"""
        x = self.model(x, targets)
        return x


def build_model(experiment_config,
                export=False):
    """ Build dino model according to configuration.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag to indicate onnx export.

    Returns:
        model (nn.Module): RT-DETR model.
    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone
    train_backbone = model_config.train_backbone

    pretrained_backbone = model_config.pretrained_backbone_path
    return_interm_indices = model_config.return_interm_indices

    eval_spatial_size = dataset_config.augmentation.eval_spatial_size
    multi_scale = dataset_config.augmentation.multi_scales
    num_classes = dataset_config.num_classes
    num_queries = model_config.num_queries
    hidden_dim = model_config.hidden_dim
    use_encoder_idx = model_config.use_encoder_idx
    nhead = model_config.nheads
    dim_feedforward = model_config.dim_feedforward
    dropout = model_config.dropout_ratio
    pe_temperature = model_config.pe_temperature
    expansion = model_config.expansion
    depth_mult = model_config.depth_mult
    enc_act = model_config.enc_act
    act = model_config.act
    num_levels = model_config.num_feature_levels
    num_encoder_layers = model_config.enc_layers
    num_decoder_layers = model_config.dec_layers
    num_denoising = model_config.dn_number
    feat_channels = model_config.feat_channels
    feat_strides = model_config.feat_strides
    eval_idx = model_config.eval_idx
    frozen_fm_cfg = model_config.frozen_fm

    activation_checkpoint = experiment_config.train.activation_checkpoint

    model = RTDETRModel(
        backbone_name=backbone,
        train_backbone=train_backbone,
        pretrained_backbone=pretrained_backbone,
        out_indices=return_interm_indices,
        num_classes=num_classes,
        eval_spatial_size=eval_spatial_size,
        multi_scale=multi_scale,
        num_queries=num_queries,

        # Encoder
        hidden_dim=hidden_dim,
        use_encoder_idx=use_encoder_idx,
        num_encoder_layers=num_encoder_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        enc_act=enc_act,
        pe_temperature=pe_temperature,
        expansion=expansion,
        depth_mult=depth_mult,
        act=act,
        # Decoder
        feat_channels=feat_channels,
        feat_strides=feat_strides,
        num_levels=num_levels,
        num_decoder_layers=num_decoder_layers,
        num_denoising=num_denoising,
        eval_idx=eval_idx,
        activation_checkpoint=activation_checkpoint,
        # frozen FM
        frozen_fm_cfg=frozen_fm_cfg,
        export=export,
    )
    return model
