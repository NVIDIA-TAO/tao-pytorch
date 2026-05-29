# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from typing import Any, Optional, List, Tuple
from dataclasses import dataclass, field

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class SemanticSegmentationHead:
    """Semantic Segmentation Head config."""

    name: str = STR_FIELD(
        value="OneFormerHead",
        default_value="OneFormerHead",
        description="Name of the semantic segmentation head.",
        display_name="name"
    )
    ignore_value: int = INT_FIELD(
        value=255,
        default_value=255,
        description="Value to ignore in the semantic segmentation head.",
        display_name="ignore value"
    )
    loss_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Loss weight of the semantic segmentation head.",
        display_name="loss weight"
    )
    in_features: List[str] = LIST_FIELD(
        arrList=["res3", "res4", "res5"],
        default_value=["res3", "res4", "res5"],
        description="List of feature names for the semantic segmentation head input.",
        display_name="in features"
    )
    common_stride: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Common stride.",
        display_name="Common stride",
        valid_min=2,
    )
    transformer_enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of transformer encoder layers.",
        display_name="Number of transformer encoder layers.",
        valid_min=1,
        popular="yes",
    )
    convs_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Convolutional layer dimension.",
        display_name="conv layer dim.",
        valid_min=1,
        popular="yes",
    )
    mask_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Mask head dimension.",
        display_name="mask head dim.",
        valid_min=1,
        popular="yes",
    )
    pixel_decoder_name: str = STR_FIELD(
        value="MSDeformAttnPixelDecoder",
        default_value="MSDeformAttnPixelDecoder",
        description="Name of the pixel decoder.",
        display_name="pixel decoder name"
    )
    deformable_transformer_encoder_in_features: List[str] = LIST_FIELD(
        arrList=["res3", "res4", "res5"],
        default_value=["res3", "res4", "res5"],
        description="List of feature names for deformable transformer encoder input.",
        display_name="transformer encoder in_features"
    )
    norm: str = STR_FIELD(
        value="GN",
        description="""Norm layer type.""",
        display_name="norm type"
    )
    num_classes: int = INT_FIELD(
        value=133,
        default_value=133,
        description="Number of classes.",
        display_name="num classes",
    )


@dataclass
class OneFormer:
    """OneFormer config."""

    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_name="hidden dim",
    )
    nheads: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of heads.",
        display_name="nheads",
    )
    dim_feedforward: int = INT_FIELD(
        value=2048,
        default_value=2048,
        description="Dimension of the feedforward network.",
        display_name="dim feedforward",
    )
    enc_layers: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Number of encoder layers.",
        display_name="enc layers",
    )
    dec_layers: int = INT_FIELD(
        value=10,
        default_value=10,
        description="Number of decoder layers.",
        display_name="dec layers",
    )
    pre_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Pre-norm.",
        display_name="pre norm",
    )
    enforce_input_proj: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enforce input projection.",
        display_name="enforce input proj",
    )
    size_divisibility: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Size divisibility.",
        display_name="size divisibility",
    )
    num_object_queries: int = INT_FIELD(
        value=150,
        default_value=150,
        description="Number of object queries.",
        display_name="num object queries",
    )
    train_num_points: int = INT_FIELD(
        value=12544,
        default_value=12544,
        description="Number of training points.",
        display_name="train num points",
    )
    oversample_ratio: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Oversample ratio.",
        display_name="oversample ratio",
    )
    importance_sample_ratio: float = FLOAT_FIELD(
        value=0.75,
        default_value=0.75,
        description="Importance sample ratio.",
        display_name="importance sample ratio",
    )
    mask_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        description="Mask weight.",
        display_name="mask weight",
    )
    dice_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        description="Dice weight.",
        display_name="dice weight",
    )
    class_weight: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        description="Class weight.",
        display_name="class weight",
    )
    no_object_weight: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="No object weight.",
        display_name="no object weight",
    )
    deep_supervision: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Deep supervision.",
        display_name="deep supervision",
    )
    dropout: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="Dropout rate.",
        display_name="dropout",
    )
    transformer_decoder_name: str = STR_FIELD(
        value="ContrastiveMultiScaleMaskedTransformerDecoder",
        default_value="ContrastiveMultiScaleMaskedTransformerDecoder",
        description="Name of the transformer decoder.",
        display_name="transformer decoder name"
    )
    transformer_in_feature: str = STR_FIELD(
        value="multi_scale_pixel_decoder",
        default_value="multi_scale_pixel_decoder",
        description="Name of the transformer input feature.",
        display_name="transformer in feature"
    )
    class_dec_layers: int = INT_FIELD(
        value=2,
        default_value=2,
        description="Number of class decoder layers.",
        display_name="class dec layers",
    )
    contrastive_weight: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Contrastive weight.",
        display_name="contrastive weight",
    )
    contrastive_temperature: float = FLOAT_FIELD(
        value=0.07,
        default_value=0.07,
        description="Contrastive temperature.",
        display_name="contrastive temperature",
    )
    use_task_norm: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use task norm.",
        display_name="use task norm",
    )
    transformer_decoder_name: str = STR_FIELD(
        value="ContrastiveMultiScaleMaskedTransformerDecoder",
        default_value="ContrastiveMultiScaleMaskedTransformerDecoder",
        description="Name of the transformer decoder.",
        display_name="transformer decoder name"
    )
    transformer_in_feature: str = STR_FIELD(
        value="multi_scale_pixel_decoder",
        default_value="multi_scale_pixel_decoder",
        description="Name of the transformer input feature.",
        display_name="transformer in feature"
    )
    num_feature_levels: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of feature levels.",
        display_name="num feature levels",
    )


@dataclass
class TestConfig:
    """Test config."""

    object_mask_threshold: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="""The value of the threshold to be used when
                    filtering out the object mask.""",
        display_name="object mask threshold"
    )
    overlap_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the threshold to be used when
                    evaluating overlap.""",
        display_name="overlap threshold"
    )
    test_topk_per_image: int = INT_FIELD(
        value=100,
        default_value=100,
        description=" keep topk instances per image for instance segmentation.",
        display_name="top k per image",
    )
    semantic_on: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable semantic segmentation.",
        display_name="semantic on"
    )
    instance_on: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable instance segmentation.",
        display_name="instance on"
    )
    panoptic_on: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable panoptic segmentation.",
        display_name="panoptic on"
    )
    detection_on: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable detection.",
        display_name="detect on"
    )


@dataclass
class Swin:
    """Swin config."""

    embed_dim: int = INT_FIELD(
        value=192,
        default_value=192,
        description="Embedding dimension.",
        display_name="embed dim",
    )
    depths: List[int] = LIST_FIELD(
        arrList=[2, 2, 18, 2],
        default_value=[2, 2, 18, 2],
        description="Depths of each stage.",
        display_name="depths",
    )
    num_heads: List[int] = LIST_FIELD(
        arrList=[6, 12, 24, 48],
        default_value=[6, 12, 24, 48],
        description="Number of heads of each stage.",
        display_name="num heads",
    )
    window_size: int = INT_FIELD(
        value=12,
        default_value=12,
        description="Window size.",
        display_name="window size",
    )
    mlp_ratio: float = FLOAT_FIELD(
        value=4.0,
        default_value=4.0,
        description="MLP ratio.",
        display_name="mlp ratio",
    )
    qkv_bias: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="QKV bias.",
        display_name="qkv bias",
    )
    qk_scale: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        display_name="qk scale",
        description="Override default qk scale of head_dim ** -0.5 if set."
    )
    attn_drop_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Attention dropout rate.",
        display_name="attn drop rate",
    )
    drop_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Dropout rate.",
        display_name="drop rate",
    )
    drop_path_rate: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="Drop path rate.",
        display_name="drop path rate",
    )
    ape: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="APE.",
        display_name="ape",
    )
    patch_norm: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Patch normalization.",
        display_name="patch norm",
    )
    patch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Patch size.",
        display_name="patch size",
    )
    pretrain_img_size: int = INT_FIELD(
        value=384,
        default_value=384,
        description="Pretrained image size.",
        display_name="pretrained image size",
    )
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use checkpoint.",
        display_name="use checkpoint",
    )
    out_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        default_value=["res2", "res3", "res4", "res5"],
        description="List of output features.",
        display_name="out features",
    )
    out_indices: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        default_value=[0, 1, 2, 3],
        description="List of output indices.",
        display_name="out indices",
    )
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use checkpoint.",
        display_name="use checkpoint",
    )


@dataclass
class Radio:
    """Radio config."""

    resolution: Tuple[int, int] = LIST_FIELD(
        arrList=[1024, 1024],
        default_value=[1024, 1024],
        description="Resolution of the radio.",
        display_name="resolution"
    )
    backbone: str = STR_FIELD(
        value="vit_base_patch16_224",
        default_value="vit_base_patch16_224",
        description="Name of the radio backbone.",
        display_name="backbone"
    )
    summary_idxs: List[int] = LIST_FIELD(
        arrList=[0, 1, 2],
        default_value=[0, 1, 2],
        description="Summary indices.",
        display_name="summary idxs"
    )
    window_size: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Window size.",
        display_name="window size"
    )
    num_teacher: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Number of teachers.",
        display_name="num teacher"
    )
    cpe_max_size: int = INT_FIELD(
        value=2048,
        default_value=2048,
        description="Maximum size of the cropped positional embedding.",
        display_name="cpe max size"
    )
    register_multiple: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of extra tokens.",
        display_name="register multiple"
    )
    out_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        default_value=["res2", "res3", "res4", "res5"],
        description="List of output features.",
        display_name="out features",
    )
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use checkpoint.",
        display_name="use checkpoint"
    )


@dataclass
class DiNAT:
    """DiNAT backbone config."""

    embed_dim: int = INT_FIELD(
        value=192,
        default_value=192,
        description="Embedding dimension.",
        display_name="embed dim",
    )
    mlp_ratio: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        description="MLP ratio.",
        display_name="mlp ratio",
    )
    depths: List[int] = LIST_FIELD(
        arrList=[3, 4, 18, 5],
        description="Depths of each stage.",
        display_name="depths",
    )
    num_heads: List[int] = LIST_FIELD(
        arrList=[6, 12, 24, 48],
        description="Number of heads of each stage.",
        display_name="num heads",
    )
    kernel_size: int = INT_FIELD(
        value=7,
        default_value=7,
        description="Neighborhood attention kernel size.",
        display_name="kernel size",
    )
    drop_path_rate: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="Drop path rate.",
        display_name="drop path rate",
    )
    dilations: Any = field(
        default=None,
        metadata={
            "display_name": "dilations",
            "description": "Per-layer dilation schedule (list of lists of ints).",
            "value_type": "list",
            "default_value": None,
        },
    )
    out_indices: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        description="List of output stage indices.",
        display_name="out indices",
    )
    out_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        description="List of output features.",
        display_name="out features",
    )
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use gradient checkpointing.",
        display_name="use checkpoint",
    )


@dataclass
class Backbone:
    """Backbone config."""

    name: str = STR_FIELD(
        value="D2SwinTransformer",
        default_value="D2SwinTransformer",
        description="Name of the backbone.",
        display_name="name"
    )
    freeze_at: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Freeze at.",
        display_name="freeze at",
    )
    swin: Swin = DATACLASS_FIELD(
        Swin(),
        description="Swin.",
        display_name="swin",
    )
    dinat: DiNAT = DATACLASS_FIELD(
        DiNAT(),
        description="DiNAT.",
        display_name="dinat",
    )
    radio: Radio = DATACLASS_FIELD(
        Radio(),
        description="Radio.",
        display_name="radio",
    )


@dataclass
class TextEncoder:
    """Text encoder config."""

    context_length: int = INT_FIELD(
        value=77,
        default_value=77,
        description="Context length.",
        display_name="context length",
    )
    vocab_size: int = INT_FIELD(
        value=49408,
        default_value=49408,
        description="Vocabulary size.",
        display_name="vocab size",
    )
    width: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Width.",
        display_name="width",
    )
    num_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of layers.",
        display_name="num layers",
    )
    n_ctx: int = INT_FIELD(
        value=16,
        default_value=16,
        description="Context length.",
        display_name="context length",
    )
    proj_num_layers: int = INT_FIELD(
        value=2,
        default_value=2,
        description="Number of projection layers.",
        display_name="proj num layers",
    )


@dataclass
class OneFormerModelConfig:
    """OneFormer model config."""

    sem_seg_head: SemanticSegmentationHead = DATACLASS_FIELD(
        SemanticSegmentationHead(),
        description="Semantic segmentation head.",
        display_name="sem seg head",
    )
    one_former: OneFormer = DATACLASS_FIELD(
        OneFormer(),
        description="OneFormer.",
        display_name="oneformer",
    )
    text_encoder: TextEncoder = DATACLASS_FIELD(
        TextEncoder(),
        description="Text encoder.",
        display_name="text encoder",
    )
    backbone: Backbone = DATACLASS_FIELD(
        Backbone(),
        description="Backbone.",
        display_name="backbone",
    )
    export: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="export",
        description="A flag to enable export mode."
    )
    test: TestConfig = DATACLASS_FIELD(
        TestConfig(),
        description="Test.",
        display_name="Test configs",
    )
