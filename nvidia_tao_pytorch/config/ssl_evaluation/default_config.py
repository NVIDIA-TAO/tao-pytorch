# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable config dataclasses for the shared embedding-evaluation suite.

These per-metric blocks (``KNNEvalConfig``, ...) are network-agnostic so the SSL
``evaluate`` actions for NV-DINOv2 and MAE (and any future consumer) share one
schema. A network's ``EvaluateConfig`` mixes in :class:`EvalSuiteConfig` to gain
the metric blocks plus ``cache_dir`` alongside its own GPU/checkpoint fields.

Each block's ``enabled`` flag drives ``build_enabled_evaluators`` — only enabled
metrics run. Defaults preserve the paper protocol (KNN k=20, ImageNet-normalized
for backbones whose adapter does not normalize internally).

Lives under ``config/`` (not inside ``core/evaluation``) so importing the config
schema stays torch-light, matching every other ``config/*/default_config.py``.
"""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class KNNEvalConfig:
    """KNN Top-1 classification probe (summary/CLS embedding, cosine, weighted vote)."""

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Whether to run the KNN classification evaluator.",
        display_name="KNN enabled",
    )
    dataset_type: str = STR_FIELD(
        value="image_folder",
        default_value="image_folder",
        valid_options="image_folder,webdataset",
        description="Labeled dataset backend: torchvision ImageFolder layout or WebDataset shards.",
        display_name="dataset type",
    )
    train_root: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the train split (ImageFolder root or WDS shard pattern) used as the KNN index.",
        display_name="train root",
    )
    val_root: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the val split queried against the KNN index.",
        display_name="val root",
    )
    k: int = INT_FIELD(
        value=20,
        default_value=20,
        valid_min=1,
        valid_max="inf",
        description="Number of nearest neighbors for the weighted majority vote (protocol: 20).",
        display_name="k",
    )
    batch_size: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=1,
        valid_max="inf",
        description="Feature-extraction batch size.",
        display_name="batch size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        valid_max="inf",
        description="Dataloader worker processes.",
        display_name="workers",
    )
    crop: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=1,
        valid_max="inf",
        description="Center-crop (and model input) resolution.",
        display_name="crop size",
    )
    resize: Optional[int] = INT_FIELD(
        value=None,
        default_type=None,
        description="Shorter-side resize target; defaults to crop (c-radiov4 KNN parity).",
        display_name="resize",
    )
    interpolation: str = STR_FIELD(
        value="bicubic",
        default_value="bicubic",
        valid_options="bicubic,bilinear,nearest",
        description="Resize interpolation mode.",
        display_name="interpolation",
    )
    imagenet_normalize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "Apply ImageNet mean/std normalization in the dataset. Set True for backbones whose "
            "adapter does NOT normalize (NV-DINOv2); False for those that normalize internally (RADIO)."
        ),
        display_name="ImageNet normalize",
    )
    num_classes: int = INT_FIELD(
        value=1000,
        default_value=1000,
        valid_min=1,
        valid_max="inf",
        description="Number of classes (ImageNet-1k = 1000); required for the WebDataset backend.",
        display_name="num classes",
    )
    label_key: str = STR_FIELD(
        value="cls",
        default_value="cls",
        description="Per-sample label field for the WebDataset backend (RADIO shards use 'id').",
        display_name="WDS label key",
    )
    max_train_samples: Optional[int] = INT_FIELD(
        value=None,
        default_type=None,
        description="Optional cap on the train index size (deterministic subset) for quick sanity checks.",
        display_name="max train samples",
    )
    amp: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use bfloat16 autocast for feature extraction.",
        display_name="amp",
    )
    use_faiss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "Use FAISS for the offline KNN search when available; set False to force the "
            "brute-force matmul fallback (numerically identical) — useful if faiss-gpu is "
            "unstable in the container (it can segfault during IndexFlatIP search)."
        ),
        display_name="use faiss",
    )
    cache_tag: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Tag for the on-disk feature cache filename; defaults to the network name.",
        display_name="cache tag",
    )


@dataclass
class SegEvalConfig:
    """Segmentation linear-probe (BNHead) mIoU. Disabled by default (80k-iter train)."""

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to run the segmentation linear-probe evaluator.",
        display_name="segmentation enabled",
    )
    dataset: str = STR_FIELD(
        value="ade20k",
        default_value="ade20k",
        valid_options="ade20k,voc,cityscapes",
        description="Segmentation dataset.",
        display_name="dataset",
    )
    root: str = STR_FIELD(
        value="",
        default_value="",
        description="Dataset root directory.",
        display_name="root",
    )
    total_iters: int = INT_FIELD(
        value=80000,
        default_value=80000,
        valid_min=1,
        valid_max="inf",
        description="Head training iterations (protocol: 80000).",
        display_name="total iters",
    )
    batch_size: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Per-GPU train batch size (paper: 2; total 16 across 8 GPUs).",
        display_name="batch size",
    )
    num_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        valid_max="inf",
        description="Dataloader workers.",
        display_name="workers",
    )
    crop_size: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=1,
        valid_max="inf",
        description="Square train crop size.",
        display_name="crop size",
    )
    pad_divisor: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        valid_max="inf",
        description="Pad images to this divisor (defaults to backbone patch size).",
        display_name="pad divisor",
    )
    base_lr: float = FLOAT_FIELD(
        value=1e-3,
        default_value=1e-3,
        description="AdamW base LR for the head (protocol: 1e-3).",
        display_name="base lr",
    )
    warmup_steps: int = INT_FIELD(
        value=1500,
        default_value=1500,
        valid_min=0,
        valid_max="inf",
        description="Linear warmup steps before poly decay (protocol: 1500).",
        display_name="warmup steps",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="AdamW weight decay (protocol: 0).",
        display_name="weight decay",
    )
    val_every: int = INT_FIELD(
        value=2000,
        default_value=2000,
        valid_min=1,
        valid_max="inf",
        description="Validate every N iters during training.",
        display_name="val every",
    )
    tiling: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use overlapping-tile (full-res) inference at val time.",
        display_name="tiling",
    )
    tile_size: int = INT_FIELD(
        value=512, default_value=512, valid_min=1, valid_max="inf",
        description="Tile size for tiled inference.", display_name="tile size",
    )
    tile_stride: int = INT_FIELD(
        value=256, default_value=256, valid_min=1, valid_max="inf",
        description="Stride between tiles.", display_name="tile stride",
    )
    tile_batch_size: int = INT_FIELD(
        value=8, default_value=8, valid_min=1, valid_max="inf",
        description="Backbone tile chunk size.", display_name="tile batch size",
    )
    max_val_samples: int = INT_FIELD(
        value=0, default_value=0, valid_min=0, valid_max="inf",
        description="Evaluate only the first N val samples (0=all); fast non-canonical signal.",
        display_name="max val samples",
    )
    feature_cache_dir: Optional[str] = STR_FIELD(
        value=None, default_type=None,
        description="Pre-extracted dense feature cache dir; backbone bypassed when set.",
        display_name="feature cache dir",
    )
    imagenet_normalize: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="ImageNet-normalize in the dataset (True for NV-DINOv2; False for RADIO).",
        display_name="ImageNet normalize",
    )
    compute_small_object_iou: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="Also report ADE20K small-object mIoU.",
        display_name="small-object IoU",
    )
    amp: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="Use bfloat16 autocast for backbone feature extraction.",
        display_name="amp",
    )


@dataclass
class RetrievalEvalConfig:
    """Image-to-image retrieval (Recall@k / mAP / NDCG@k). Disabled by default."""

    enabled: bool = BOOL_FIELD(
        value=False, default_value=False,
        description="Whether to run the retrieval evaluator.",
        display_name="retrieval enabled",
    )
    dataset_type: str = STR_FIELD(
        value="image_folder", default_value="image_folder",
        valid_options="image_folder,webdataset",
        description="Labeled dataset backend.", display_name="dataset type",
    )
    root: str = STR_FIELD(
        value="", default_value="",
        description="Gallery (and default query) set root.", display_name="root",
    )
    query_root: Optional[str] = STR_FIELD(
        value=None, default_type=None,
        description="Optional separate query set; defaults to the gallery (self-match excluded).",
        display_name="query root",
    )
    batch_size: int = INT_FIELD(
        value=128, default_value=128, valid_min=1, valid_max="inf",
        description="Feature-extraction batch size.", display_name="batch size",
    )
    num_workers: int = INT_FIELD(
        value=8, default_value=8, valid_min=0, valid_max="inf",
        description="Dataloader workers.", display_name="workers",
    )
    crop: int = INT_FIELD(
        value=224, default_value=224, valid_min=1, valid_max="inf",
        description="Center-crop (model input) size.", display_name="crop size",
    )
    resize: Optional[int] = INT_FIELD(
        value=None, default_type=None,
        description="Shorter-side resize; defaults to crop.", display_name="resize",
    )
    interpolation: str = STR_FIELD(
        value="bicubic", default_value="bicubic", valid_options="bicubic,bilinear,nearest",
        description="Resize interpolation.", display_name="interpolation",
    )
    imagenet_normalize: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="ImageNet-normalize in the dataset (True for NV-DINOv2; False for RADIO).",
        display_name="ImageNet normalize",
    )
    num_classes: int = INT_FIELD(
        value=1000, default_value=1000, valid_min=1, valid_max="inf",
        description="Number of classes (WebDataset backend).", display_name="num classes",
    )
    label_key: str = STR_FIELD(
        value="cls", default_value="cls",
        description="WebDataset per-sample label field.", display_name="WDS label key",
    )
    k_values: List[int] = LIST_FIELD(
        arrList=[1, 5, 10],
        default_value=[1, 5, 10],
        description="k values for Recall@k and NDCG@k.",
        display_name="k values",
    )
    amp: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="Use bfloat16 autocast for feature extraction.", display_name="amp",
    )


@dataclass
class EvalSuiteConfig:
    """Mixin adding the shared evaluation-suite blocks to a network EvaluateConfig."""

    cache_dir: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Directory for cached feature tensors (extract-once). Point at scratch; None disables.",
        display_name="cache dir",
    )
    knn: KNNEvalConfig = DATACLASS_FIELD(
        KNNEvalConfig(),
        description="K-Nearest-Neighbors classification probe configuration.",
    )
    segmentation: SegEvalConfig = DATACLASS_FIELD(
        SegEvalConfig(),
        description="Segmentation linear-probe (BNHead) configuration.",
    )
    retrieval: RetrievalEvalConfig = DATACLASS_FIELD(
        RetrievalEvalConfig(),
        description="Image-to-image retrieval configuration.",
    )
