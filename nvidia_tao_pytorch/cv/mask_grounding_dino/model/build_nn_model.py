# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""The build nn module model."""

import torch.nn as nn

from nvidia_tao_pytorch.cv.dino.model.position_encoding import PositionEmbeddingSineHW, PositionEmbeddingSineHWExport
from nvidia_tao_pytorch.cv.dino.model.backbone import Joiner

from nvidia_tao_pytorch.cv.grounding_dino.model.backbone import Backbone
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.transformer import Transformer
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.rela import ReLA
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.mask_groundingdino import MaskGroundingDINO


class MaskGDINOModel(nn.Module):
    """GDINO model module."""

    def __init__(self,
                 hidden_dim=256,
                 dropout=0.0,
                 pretrained_backbone_path=None,
                 backbone='swin_tiny_224_1k',
                 train_backbone=True,
                 num_feature_levels=4,
                 nheads=8,
                 enc_layers=6,
                 dec_layers=6,
                 dim_feedforward=1024,
                 dec_n_points=4,
                 enc_n_points=4,
                 num_queries=300,
                 aux_loss=True,
                 dilation=False,
                 export=False,
                 activation_checkpoint=True,
                 return_interm_indices=[1, 2, 3, 4],
                 pre_norm=False,
                 two_stage_type='standard',
                 embed_init_tgt=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_bbox_embed_share=False,
                 two_stage_class_embed_share=False,
                 pe_temperatureH=20,
                 pe_temperatureW=20,
                 transformer_activation='relu',
                 use_text_enhancer=True,
                 use_fusion_layer=True,
                 use_text_cross_attention=True,
                 text_dropout=0.0,
                 fusion_dropout=0.0,
                 fusion_droppath=0.1,

                 use_dn=False,
                 dn_number=100,
                 dn_box_noise_scale=1.0,
                 dn_label_noise_ratio=0.5,

                 text_encoder_type='bert-base-uncased',
                 sub_sentence_present=True,
                 log_scale=None,
                 class_embed_bias=False,
                 max_text_len=256,
                 has_mask=True,
                 num_region_queries=100,
                 ):
        """Initialize Mask Grounding DINO Model.

        Args:
            hidden_dim (int): size of the hidden dimension.
            pretrained_backbone_path (str): pretrained backbone path.
                                            If not provided, train from scratch.
            backbone (str): type of backbone architecture.
            train_backbone (bool): whether to train backbone or not.
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            nheads (int): number of heads.
            enc_layers (int): number of encoder layers.
            dec_layers (int): number of decoder layers.
            dim_feedforward (int): dimension of the feedforward layer.
            dec_n_points (int): number of reference points in the decoder.
            enc_n_points (int): number of reference points in the encoder.
            num_queries (int): number of queries to be used in D-DETR encoder-decoder.
            aux_loss (bool): flag to indicate if auxiliary loss is used.
            dilation (bool): flag to indicate if dilation is used (only for ResNet).
            export (bool): flag to indicate if the current model is being used for ONNX export.
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
            return_interm_indices (list): indices of feature level to use.
            pre_norm (bool): whether to add LayerNorm before the encoder.
            add_channel_attention (bool): whether to add channel attention.
            random_refpoints_xy (bool): whether to randomly initialize reference point embedding.
            two_stage_type (str): type of two stage in DINO.
            two_stage_pat_embed (int): size of the patch embedding for the second stage.
            two_stage_add_query_num (int): size of the target embedding.
            two_stage_learn_wh (bool): add embedding for learnable w and h.
            two_stage_keep_all_tokens (bool): whether to keep all tokens in the second stage.
            embed_init_tgt (bool): whether to add target embedding.
            use_detached_boxes_dec_out (bool): use detached box decoder output in the reference points.
            dec_pred_class_embed_share (bool): whether to share embedding for decoder classification prediction.
            dec_pred_bbox_embed_share (bool): whether to share embedding for decoder bounding box prediction.
            two_stage_bbox_embed_share (bool): whether to share embedding for two stage bounding box.
            two_stage_class_embed_share (bool): whether to share embedding for two stage classification.
            pe_temperatureH (int): the temperature applied to the height dimension of Positional Sine Embedding.
            pe_temperatureW (int): the temperature applied to the width dimension of Positional Sine Embedding.
            num_region_queries (int): number of region queries for ReLA.
        """
        super(__class__, self).__init__()  # pylint:disable=undefined-variable

        # TODO: Update position_embedding in the build stage
        # build positional encoding. only support PositionEmbeddingSine
        if export:
            position_embedding = PositionEmbeddingSineHWExport(hidden_dim // 2,
                                                               temperatureH=pe_temperatureH,
                                                               temperatureW=pe_temperatureW,
                                                               normalize=True)
        else:
            position_embedding = PositionEmbeddingSineHW(hidden_dim // 2,
                                                         temperatureH=pe_temperatureH,
                                                         temperatureW=pe_temperatureW,
                                                         normalize=True)

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
                                 export,
                                 activation_checkpoint)

        # Keep joiner for backward compatibility
        joined_backbone = Joiner(backbone_only)

        # build tranformer
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            num_queries=num_queries,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=True,
            activation=transformer_activation,
            num_feature_levels=num_feature_levels,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            learnable_tgt_init=True,
            # two stage
            two_stage_type=two_stage_type,  # ['no', 'standard', 'early']
            embed_init_tgt=embed_init_tgt,
            use_text_enhancer=use_text_enhancer,
            use_fusion_layer=use_fusion_layer,
            use_checkpoint=activation_checkpoint,
            use_text_cross_attention=use_text_cross_attention,
            text_dropout=text_dropout,
            fusion_dropout=fusion_dropout,
            fusion_droppath=fusion_droppath,
        )

        # build rela
        if num_region_queries > 0:
            rela = ReLA(
                num_queries=num_region_queries,
                hidden_dim=hidden_dim,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dec_layers=dec_layers,
                pre_norm=pre_norm,
                rla_weight=0.1,
                num_feature_levels=3,
                mask_dim=hidden_dim,
                export=export,
            )
        else:
            rela = None

        # build mask groundingdino model
        self.model = MaskGroundingDINO(
            joined_backbone,
            position_embedding,
            transformer,
            rela=rela,
            num_queries=num_queries,
            aux_loss=aux_loss,
            iter_update=True,
            query_dim=4,
            num_feature_levels=num_feature_levels,
            nheads=nheads,
            dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
            two_stage_type=two_stage_type,
            two_stage_bbox_embed_share=two_stage_bbox_embed_share,
            two_stage_class_embed_share=two_stage_class_embed_share,
            text_encoder_type=text_encoder_type,
            sub_sentence_present=sub_sentence_present,
            max_text_len=max_text_len,
            export=export,
            dn_number=dn_number if use_dn else 0,
            dn_box_noise_scale=dn_box_noise_scale,
            dn_label_noise_ratio=dn_label_noise_ratio,
            dn_labelbook_size=max_text_len,
            log_scale=log_scale,
            class_embed_bias=class_embed_bias,
        )

    def forward(self, *args, **kwargs):
        """model forward function"""
        x = self.model(*args, **kwargs)
        return x

    def set_matcher(self, matcher):
        """Set matcher."""
        self.model.matcher = matcher


def build_model(experiment_config,
                export=False):
    """ Build mask grounding dino model according to configuration.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag to indicate onnx export.

    Returns:
        model (nn.Module): Mask GDINO model.
    """
    model_config = experiment_config.model

    backbone = model_config.backbone
    has_mask = model_config.has_mask
    hidden_dim = model_config.hidden_dim
    num_feature_levels = model_config.num_feature_levels
    nheads = model_config.nheads
    enc_layers = model_config.enc_layers
    dec_layers = model_config.dec_layers
    dim_feedforward = model_config.dim_feedforward
    dec_n_points = model_config.dec_n_points
    enc_n_points = model_config.enc_n_points
    num_queries = model_config.num_queries
    aux_loss = model_config.aux_loss
    dilation = model_config.dilation
    train_backbone = model_config.train_backbone
    pretrained_backbone = model_config.pretrained_backbone_path
    dropout_ratio = model_config.dropout_ratio

    # DINO arch specific
    return_interm_indices = model_config.return_interm_indices
    pre_norm = model_config.pre_norm
    two_stage_type = model_config.two_stage_type
    embed_init_tgt = model_config.embed_init_tgt
    pe_temperatureH = model_config.pe_temperatureH
    pe_temperatureW = model_config.pe_temperatureW

    # DN training
    use_dn = model_config.use_dn
    dn_number = model_config.dn_number
    dn_box_noise_scale = model_config.dn_box_noise_scale
    dn_label_noise_ratio = model_config.dn_label_noise_ratio

    activation_checkpoint = experiment_config.train.activation_checkpoint
    text_encoder_type = model_config.text_encoder_type
    max_text_len = model_config.max_text_len
    log_scale = model_config.log_scale
    class_embed_bias = model_config.class_embed_bias

    num_region_queries = model_config.num_region_queries

    model = MaskGDINOModel(
        hidden_dim=hidden_dim,
        pretrained_backbone_path=pretrained_backbone,
        backbone=backbone,
        train_backbone=train_backbone,
        num_feature_levels=num_feature_levels,
        nheads=nheads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,
        num_queries=num_queries,
        aux_loss=aux_loss,
        dilation=dilation,
        export=export,
        activation_checkpoint=activation_checkpoint,
        return_interm_indices=return_interm_indices,
        embed_init_tgt=embed_init_tgt,
        pe_temperatureH=pe_temperatureH,
        pe_temperatureW=pe_temperatureW,

        use_dn=use_dn,
        dn_number=dn_number if use_dn else 0,
        dn_box_noise_scale=dn_box_noise_scale,
        dn_label_noise_ratio=dn_label_noise_ratio,

        dropout=dropout_ratio,
        pre_norm=pre_norm,
        two_stage_type=two_stage_type,
        text_encoder_type=text_encoder_type,
        max_text_len=max_text_len,
        log_scale=log_scale,
        class_embed_bias=class_embed_bias,
        has_mask=has_mask,
        num_region_queries=num_region_queries)
    return model
