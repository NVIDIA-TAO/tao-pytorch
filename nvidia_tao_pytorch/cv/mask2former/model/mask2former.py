# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask2former model."""
from torch import nn
from torch.nn import functional as F
import torch

from nvidia_tao_pytorch.cv.mask2former.model.backbone.swin import D2SwinTransformer
from nvidia_tao_pytorch.cv.mask2former.model.backbone.efficientvit import EfficientViT
from nvidia_tao_pytorch.cv.mask2former.model.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from nvidia_tao_pytorch.cv.mask2former.model.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class Postprocessor(nn.Module):
    """Mask2former postprocessor."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.test_topk_per_image = cfg.model.test_topk_per_image
        self.num_classes = cfg.model.sem_seg_head.num_classes
        self.num_queries = cfg.model.mask_former.num_object_queries

    def batch_panoptic_inference(self, mask_cls, mask_pred):
        """Batched panoptic inference."""
        bs = mask_cls.shape[0]
        scores_per_batch, labels_per_batch = F.softmax(mask_cls, dim=-1)[..., 1:].max(-1)
        mask_pred = mask_pred.sigmoid()
        cur_prob_masks = scores_per_batch.view(bs, -1, 1, 1) * mask_pred
        return cur_prob_masks, mask_pred, scores_per_batch, labels_per_batch

    def batch_semantic_inference(self, mask_cls, mask_pred):
        """Batched semantic inference."""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        # TODO(@yuw): USE the workaround if tensorrt engine fails
        # H, W = mask_pred.shape[-2:]
        # semseg = torch.bmm(mask_cls.permute(0, 2, 1), mask_pred.flatten(2))
        # semseg = semseg.view(-1, self.num_classes, H, W)
        return semseg

    def batch_instance_inference(self, mask_cls, mask_pred):
        """Batched instance inference."""
        # mask_pred is already processed to have the same shape as original input
        B = mask_cls.shape[0]
        scores = F.softmax(mask_cls, dim=-1)[:, :, 1:]
        # [QxC]
        labels = torch.arange(self.num_classes).to(scores.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # [B, QC]
        labels = labels.unsqueeze(0).repeat(B, 1)
        # [B, QC] --> [B,topk], [B,topk]
        scores_per_image, topk_indices = scores.flatten(1, 2).topk(self.test_topk_per_image, sorted=False)
        # [B, topk]
        labels_per_image = labels.gather(1, topk_indices.to(torch.int64))
        topk_indices = topk_indices // self.num_classes  # [B, top_k]

        # [B, Q, H, W] -> [B, topk, H, W]
        # mask_pred = mask_pred.gather(1, torch.tile(topk_indices[..., None, None], (H, W)))
        mask_pred = torch.stack([torch.index_select(mask_pred[i], 0, idx) for i, idx in enumerate(topk_indices)])

        # mask (before sigmoid)
        pred_masks = (mask_pred > 0).float()
        # pred_boxes = torch.zeros((B, self.test_topk_per_image, 4))

        # calculate average mask prob
        # [B, topk]
        mask_scores_per_image = (mask_pred.sigmoid().flatten(2) * pred_masks.flatten(2)).sum(2) / (pred_masks.flatten(2).sum(-1) + 1e-6)
        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image
        return (pred_masks, pred_scores, pred_classes)

    def forward(self, outputs):
        """Forward pass."""
        pred_masks = F.interpolate(
            outputs["pred_masks"],
            size=(self.cfg.export.input_height, self.cfg.export.input_width),
            mode="bilinear",
            align_corners=False,
        )
        if self.cfg.model.mode == 'semantic':
            semseg = self.batch_semantic_inference(
                outputs["pred_logits"], pred_masks)
            return semseg
        elif self.cfg.model.mode == 'instance':
            pred_masks, pred_scores, pred_classes = self.batch_instance_inference(
                outputs["pred_logits"], pred_masks
            )
            return pred_masks, pred_scores, pred_classes
        elif self.cfg.model.mode == 'panoptic':
            return self.batch_panoptic_inference(outputs["pred_logits"], pred_masks)
        else:
            raise ValueError("Only semantic, instance and panoptic modes are supported.")


class MaskFormerHead(nn.Module):
    """Maskformer Head."""

    def __init__(self, cfg, input_shape):
        """Init."""
        super().__init__()
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        self.predictor = self.predictor_init(cfg)

    def pixel_decoder_init(self, cfg, input_shape):
        """Initialize pixel decoder."""
        export = cfg.model.export
        transformer_dropout = cfg.model.mask_former.dropout
        transformer_nheads = cfg.model.mask_former.nheads

        common_stride = cfg.model.sem_seg_head.common_stride
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.model.sem_seg_head.transformer_enc_layers
        conv_dim = cfg.model.sem_seg_head.convs_dim
        mask_dim = cfg.model.sem_seg_head.mask_dim
        transformer_in_features = cfg.model.sem_seg_head.deformable_transformer_encoder_in_features  # ["res3", "res4", "res5"]
        norm = cfg.model.sem_seg_head.norm
        # swin-l: {'res2': {'channel': 192, 'stride': 4},
        # 'res3': {'channel': 384, 'stride': 8},
        # 'res4': {'channel': 768, 'stride': 16},
        # 'res5': {'channel': 1536, 'stride': 32}}
        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
            norm=norm,
            export=export)
        return pixel_decoder

    def predictor_init(self, cfg):
        """Init predictor class."""
        export = cfg.model.export
        in_channels = cfg.model.sem_seg_head.convs_dim
        num_classes = cfg.model.sem_seg_head.num_classes
        mask_dim = cfg.model.sem_seg_head.mask_dim

        hidden_dim = cfg.model.mask_former.hidden_dim
        num_queries = cfg.model.mask_former.num_object_queries
        nheads = cfg.model.mask_former.nheads
        dim_feedforward = cfg.model.mask_former.dim_feedforward
        dec_layers = cfg.model.mask_former.dec_layers - 1
        pre_norm = cfg.model.mask_former.pre_norm
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder(
            in_channels,
            num_classes,
            mask_classification,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            export)
        return predictor

    def forward(self, features, mask=None):
        """Forward."""
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions


class MaskFormerModel(nn.Module):
    """Mask2former Model."""

    def __init__(self, cfg):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)
        self.post_processor = Postprocessor(cfg) if cfg.model.export else None

    def build_backbone(self, cfg):
        """Build backbone."""
        model_type = cfg.model.backbone.type
        if model_type == 'swin':
            swin_depth = {'tiny': [2, 2, 6, 2], 'small': [2, 2, 18, 2], 'base': [2, 2, 18, 2], 'large': [2, 2, 18, 2]}
            swin_heads = {'tiny': [3, 6, 12, 24], 'small': [3, 6, 12, 24], 'base': [4, 8, 16, 32], 'large': [6, 12, 24, 48]}
            swin_dim = {'tiny': 96, 'small': 96, 'base': 128, 'large': 192}
            cfg.model.backbone.swin.depths = swin_depth[cfg.model.backbone.swin.type]
            cfg.model.backbone.swin.num_heads = swin_heads[cfg.model.backbone.swin.type]
            cfg.model.backbone.swin.embed_dim = swin_dim[cfg.model.backbone.swin.type]
            backbone = D2SwinTransformer(cfg)
            self.backbone_feature_shape = backbone.output_shape()
        elif model_type == 'efficientvit':
            backbone = EfficientViT(cfg, export=cfg.model.export)
            self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError('Do not support model type!')
        return backbone

    def forward(self, inputs):
        """Forward."""
        features = self.backbone(inputs)
        outputs = self.sem_seg_head(features)
        if self.post_processor is not None:
            return self.post_processor(outputs)
        return outputs
