# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.solver import WarmupParamScheduler

from nvidia_tao_pytorch.cv.odise.config.common.data.coco_panoptic_semseg import dataloader
from nvidia_tao_pytorch.cv.odise.config.common.optim import AdamW as optimizer
from nvidia_tao_pytorch.cv.odise.data.build import get_openseg_labels
from nvidia_tao_pytorch.cv.odise.modeling.backbone.clip import CLIP
from nvidia_tao_pytorch.cv.odise.modeling.meta_arch.odise import (
    CategoryODISE,
    ODISEMultiScaleMaskedTransformerDecoder,
    PooledMaskEmbed,
    CategoryEmbed,
    PseudoClassEmbed,
)
from nvidia_tao_pytorch.cv.odise.modeling.meta_arch.mask_former_head import MaskFormerHead
from nvidia_tao_pytorch.cv.odise.modeling.criterion import SetCriterion
from nvidia_tao_pytorch.cv.odise.modeling.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.odise.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

from ..common.train import train


# model config
model = L(CategoryODISE)(
    alpha=0.4,
    beta=0.8,
    is_inference=False,
    precision='fp32',
    backbone=L(CLIP)(
        model_name="convnext_large_d_320",
        pretrained="laion2b_s29b_b131k_ft_soup",
        precision="${..precision}"),
    sem_seg_head=L(MaskFormerHead)(
        ignore_value=255,
        num_classes=133,
        pixel_decoder=L(MSDeformAttnPixelDecoder)(
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
        ),
        loss_weight=1.0,
        transformer_in_feature="multi_scale_pixel_decoder",
        transformer_predictor=L(ODISEMultiScaleMaskedTransformerDecoder)(
            precision="${...precision}",
            class_embed=L(PseudoClassEmbed)(num_classes="${..num_classes}"),
            hidden_dim=256,
            post_mask_embed=L(PooledMaskEmbed)(
                hidden_dim="${..hidden_dim}",
                mask_dim="${..mask_dim}",
                projection_dim=768, # 1024 for convnext-xxl
            ),
            in_channels="${..pixel_decoder.conv_dim}",
            mask_classification=True,
            num_classes="${..num_classes}",
            num_queries="${...num_queries}",
            nheads=8,
            dim_feedforward=2048,
            # 9 decoder layers, add one for the loss on learnable query
            dec_layers=9,
            pre_norm=False,
            enforce_input_project=False,
            mask_dim=256,
        ),
    ),
    criterion=L(SetCriterion)(
        num_layers="${..sem_seg_head.transformer_predictor.dec_layers}",
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class="${..class_weight}",
            cost_mask="${..mask_weight}",
            cost_dice="${..dice_weight}",
            num_points="${..num_points}",
        ),
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    category_head=L(CategoryEmbed)(
        clip_model_name="convnext_large_d_320",
        pretrained="laion2b_s29b_b131k_ft_soup",
        labels=L(get_openseg_labels)(dataset="coco_panoptic", prompt_engineered=True),
        projection_dim=-1,  # "${..sem_seg_head.transformer_predictor.post_mask_embed.projection_dim}"
        precision="${..precision}"
    ),
    num_queries=250,
    object_mask_threshold=0.0,
    overlap_threshold=0.8,
    metadata=L(MetadataCatalog.get)(name="coco_2017_train_panoptic_with_sem_seg"),
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    # normalize to [0, 1]
    pixel_mean=[122.7709383, 116.7460125, 104.09373615], # [0.0, 0.0, 0.0],
    pixel_std=[68.5005327, 66.6321579, 70.32316305], # [255.0, 255.0, 255.0],
    # inference
    semantic_on=True,
    instance_on=True,
    panoptic_on=True,
    test_topk_per_image=100,
)

train.max_iter = 92_188
train.grad_clip = 0.01
train.checkpointer.period = 4500

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # assume 100e with batch-size 64 as original LSJ
        # Equivalent to 100 epochs.
        # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
        milestones=[163889, 177546],
        num_updates=184375,
    ),
    # for warmup length we adopted COCO LSJ setting
    warmup_length=500 / 184375,
    warmup_factor=0.067,
)
