# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""VideoCLIP experiment configuration."""

from dataclasses import dataclass
from typing import Any, List, Optional

from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    GenTrtEngineConfig,
    TrainConfig,
    TrtConfig,
)
from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    UNION_FIELD,
)


# =============================================================================
# Model Config
# =============================================================================
@dataclass
class VideoCLIPModelConfig:
    """VideoCLIP (InternVideo2-CLIP) model configuration."""

    type: str = STR_FIELD(
        value="internvideo2-clip-l14",
        default_value="internvideo2-clip-l14",
        description="CLIP model type. "
                    "C-RADIO: c-radio_v3-h, c-radio_v3-l, c-radio_v3-b, c-radio_v3-g; "
                    "SigLIP2: siglip2-so400m-patch16-naflex (NaFlex), siglip2-so400m-patch14-224, "
                    "siglip2-so400m-patch14-384, siglip2-so400m-patch16-256, "
                    "siglip2-so400m-patch16-384, siglip2-so400m-patch16-512; "
                    "OpenCLIP: ViT-L-14-SigLIP-CLIPA-224, ViT-L-14-SigLIP-CLIPA-336, "
                    "ViT-H-14-SigLIP-CLIPA-224; "
                    "InternVideo2: internvideo2-clip-l14.",
        display_name="Model Type",
    )
    adaptor_name: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Text adaptor for C-RADIO models (ignored for other model types). "
                    "'siglip' (SigLIP2 text encoder) or 'clip' (DFN CLIP text encoder). "
                    "When None, defaults to 'siglip' at runtime.",
        display_name="Adaptor Name",
    )
    freeze_vision_encoder: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, freeze vision encoder weights during training.",
        display_name="Freeze Vision Encoder",
    )
    freeze_text_encoder: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, freeze text encoder weights during training.",
        display_name="Freeze Text Encoder",
    )
    image_size: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Input image resolution for training transforms. "
                    "Common values: 224 (InternVideo2-CLIP L14 / RADIO / OpenCLIP), "
                    "256 (SigLIP2-so400m), 384 (SigLIP2-g). "
                    "Must be a multiple of the model's patch size (typically 14 or 16).",
        display_name="Image Size",
    )
    init_logit_scale: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Override for the initial logit scale (log-space). "
                    "When None, automatically set from train.loss_type: "
                    "2.3026 (SigLIP) or 2.6592 (CLIP). "
                    "Set manually only with caution, as incorrect values "
                    "can destabilize training.",
        display_name="Initial Logit Scale",
    )
    init_logit_bias: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Override for the initial logit bias. "
                    "When None, automatically set from train.loss_type: "
                    "-10.0 (SigLIP) or 0.0 (CLIP). "
                    "Set manually only with caution, as incorrect values "
                    "can destabilize training.",
        display_name="Initial Logit Bias",
    )
    canonicalize_text: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Apply text canonicalization (lowercase + punctuation removal) "
                    "before tokenization. Set to True to match Google big_vision/SigLIP "
                    "zero-shot classification preprocessing. Set to False (default) to "
                    "preserve punctuation, which is better for retrieval tasks and "
                    "matches original CLIP/OpenCLIP behavior.",
        display_name="Canonicalize Text",
    )
    # Base-model initialization has two supported forms:
    # 1. Component weights: set internvideo2clip_hf_id (vision + CLIP head) and
    #    text_encoder (MobileCLIP), with optional explicit component overrides.
    # 2. Complete weights: set pretrained_ckpt; component sources are ignored.
    # Set every weight-source field to null only for random initialization.
    pretrained_ckpt: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional local .pth holding the COMPLETE InternVideo2-CLIP "
                    "model state_dict. If set, it is loaded last and overrides all "
                    "component weights (vision/text/clip_head). Must also be null for "
                    "the all-null train-from-scratch configuration.",
        display_name="Pretrained Checkpoint",
    )
    internvideo2clip_hf_id: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="HuggingFace repo id for InternVideo2-CLIP assets (provides the "
                    "vision encoder and CLIP head). Resolved via the ambient HF "
                    "cache (set HF_HOME). "
                    "Suggested value: 'OpenGVLab/InternVideo2_distillation_models'. "
                    "Leave null together with vision_encoder, text_encoder, "
                    "clip_head, and pretrained_ckpt (all null) to train from scratch "
                    "/ random initialization.",
        display_name="InternVideo2CLIP HF ID",
    )
    vision_encoder: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Vision encoder source: a local file OR an HF repo id. If unset, "
                    "resolved from internvideo2clip_hf_id (suggested: "
                    "'OpenGVLab/InternVideo2_distillation_models'). Null with all "
                    "other weight-source fields null => train from scratch.",
        display_name="Vision Encoder",
    )
    text_encoder: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Text encoder (MobileCLIP) source: a local file OR an HF repo "
                    "id. Required when pretrained_ckpt is not set (MobileCLIP is not "
                    "in the IV2CLIP repo). Suggested: MobileCLIP 'mobileclip_blt.pt' "
                    "(local path or HF id). Leave null only for the all-null "
                    "train-from-scratch configuration.",
        display_name="Text Encoder",
    )
    clip_head: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="CLIP alignment head source: a local file OR an HF repo id. If "
                    "unset, resolved from internvideo2clip_hf_id (suggested: "
                    "'OpenGVLab/InternVideo2_distillation_models'). Applied last and "
                    "overwrites overlapping vision/text keys. Null with all other "
                    "weight-source fields null => train from scratch.",
        display_name="CLIP Head",
    )
    num_frames: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Number of video frames for InternVideo2 video-text inputs.",
        display_name="Number of Frames",
    )
    use_flash_attn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable upstream InternVideo2 FlashAttention in the vision "
                    "encoder (use_flash_attn). Requires flash-attn in the env.",
        display_name="Use FlashAttention",
    )
    use_fused_rmsnorm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable upstream InternVideo2 fused RMSNorm "
                    "(use_fused_rmsnorm). Requires the flash-attn "
                    "dropout_layer_norm CUDA extension.",
        display_name="Use Fused RMSNorm",
    )
    use_fused_mlp: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable upstream InternVideo2 fused MLP (use_fused_mlp). "
                    "Requires the flash-attn fused_dense_lib CUDA extension.",
        display_name="Use Fused MLP",
    )


# =============================================================================
# PEFT Config
# =============================================================================
@dataclass
class VideoCLIPLoRATargetConfig:
    """LoRA configuration for a single encoder tower (vision or text).

    Controls which transformer blocks and attention modules receive
    low-rank adapters. When enabled, only the LoRA parameters in the
    selected blocks are trainable; all other backbone parameters are frozen.
    """

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable LoRA adaptation for this encoder tower.",
        display_name="Enabled",
    )
    target_modules: List[str] = LIST_FIELD(
        arrList=["qkv", "proj"],
        default_value=["qkv", "proj"],
        description="Attention-projection leaf module names to wrap with LoRA "
                    "within each adapted block. Defaults are for the "
                    "InternVideo2-CLIP vision tower (fused 'qkv' + 'proj'); the "
                    "MobileCLIP text tower uses 'qkv_proj' + 'out_proj' (set on "
                    "the peft.text default). Matching is by the leaf module name.",
        display_name="Target Modules",
    )
    num_last_blocks: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=0,
        description="Number of final transformer blocks to adapt. "
                    "0 means adapt all blocks.",
        display_name="Number of Last Blocks",
    )
    rank: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="LoRA rank (low-rank dimension).",
        display_name="Rank",
    )
    alpha: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="LoRA alpha scaling factor. Effective scale = alpha / rank.",
        display_name="Alpha",
    )
    dropout: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        valid_max=1.0,
        description="Dropout applied to LoRA input.",
        display_name="Dropout",
    )


@dataclass
class VideoCLIPPEFTConfig:
    """Parameter-efficient fine-tuning configuration for CLIP.

    When enabled, injects LoRA adapters into the selected encoder towers
    and freezes all non-LoRA backbone parameters. Disabled by default
    so existing training behavior is unchanged.
    """

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable PEFT mode. When False, training uses standard "
                    "full fine-tuning (existing behavior).",
        display_name="Enabled",
    )
    method: str = STR_FIELD(
        value="lora",
        default_value="lora",
        valid_options="lora",
        description="PEFT method. Currently only 'lora' is supported.",
        display_name="Method",
    )
    vision: VideoCLIPLoRATargetConfig = DATACLASS_FIELD(
        VideoCLIPLoRATargetConfig(),
        description="LoRA configuration for the vision encoder "
                    "(InternVideo2: target_modules 'qkv', 'proj').",
    )
    text: VideoCLIPLoRATargetConfig = DATACLASS_FIELD(
        VideoCLIPLoRATargetConfig(
            target_modules=["qkv_proj", "out_proj"]
        ),
        description="LoRA configuration for the text encoder "
                    "(MobileCLIP: target_modules 'qkv_proj', 'out_proj').",
    )


# =============================================================================
# Regularization Config
# =============================================================================
@dataclass
class VideoCLIPRegularizationConfig:
    """Geometry-preserving regularization for domain adaptation.

    When enabled, creates a frozen teacher copy of the pretrained model
    and adds preservation losses that constrain how much the student
    embeddings drift from the teacher during fine-tuning.
    """

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable preservation regularization. When False, "
                    "only the contrastive loss is used (existing behavior).",
        display_name="Enabled",
    )
    embedding_mse_weight: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        description="Weight for MSE loss between student and teacher embeddings.",
        display_name="Embedding MSE Weight",
    )
    cosine_weight: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        description="Weight for cosine preservation loss between "
                    "student and teacher embeddings.",
        display_name="Cosine Weight",
    )
    similarity_weight: float = FLOAT_FIELD(
        value=0.10,
        default_value=0.10,
        valid_min=0.0,
        description="Weight for similarity matrix preservation loss "
                    "(MSE between student and teacher image-text similarity matrices).",
        display_name="Similarity Weight",
    )


# =============================================================================
# Dataset Config
# =============================================================================
@dataclass
class VideoCLIPAugmentationConfig:
    """Data augmentation configuration for VideoCLIP training.

    To disable augmentations:
        - scale: [1.0, 1.0]       -> disables random resize crop scaling
        - color_jitter: []        -> disables color jitter
        - grayscale: 0.0          -> disables grayscale
    """

    scale: List[float] = LIST_FIELD(
        arrList=[0.4, 1.0],
        default_value=[0.4, 1.0],
        description="Scale range [min, max] for random resized crop. Set to [1.0, 1.0] to disable.",
        display_name="Scale Range",
    )
    color_jitter: List[float] = LIST_FIELD(
        arrList=[0.8, 0.32, 0.32, 0.32, 0.08],
        default_value=[0.8, 0.32, 0.32, 0.32, 0.08],
        description="Color jitter [prob, brightness, contrast, saturation, hue]. Set to [] to disable.",
        display_name="Color Jitter",
    )
    grayscale: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability of grayscale conversion. Set to 0.0 to disable.",
        display_name="Grayscale",
    )


@dataclass
class VideoCLIPVideoTextConfig:
    """Video-text metadata configuration."""

    metadata: Any = UNION_FIELD(
        value=None,
        union_types=["list"],
        default_value=None,
        description="List of video-text JSON/JSONL metadata file paths whose "
                    "records are concatenated (each file must be a JSON array / "
                    "JSONL). A single file may be given as a one-element list; a "
                    "bare string path is still accepted for backward compatibility.",
        display_name="Metadata",
    )
    format: str = STR_FIELD(
        value="auto",
        default_value="auto",
        valid_options="auto",
        description="Metadata format. Leave 'auto' (the standard): the canonical "
                    "schema is the nested per-video chunks layout, and legacy "
                    "shapes (flat rows / MSR-VTT) are still auto-detected. Not a "
                    "client-facing choice.",
        display_name="Format",
    )
    data_root: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Root directory used to resolve relative/remapped video paths.",
        display_name="Data Root",
    )
    split: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional split filter, such as train or test.",
        display_name="Split",
    )
    path_prefix_mapping: dict = DICT_FIELD(
        {},
        description="Optional mapping from original path prefixes to local prefixes.",
        display_name="Path Prefix Mapping",
    )
    caption_fields: List[str] = LIST_FIELD(
        arrList=["caption"],
        default_value=["caption"],
        description="Metadata fields used as caption candidates.",
        display_name="Caption Fields",
    )
    caption_mode: str = STR_FIELD(
        value="first",
        default_value="first",
        valid_options="first,random,all,one_per_field",
        description="Caption selection strategy. 'first': always use the first "
                    "caption (captions[0]). 'random': sample one caption per "
                    "epoch from the flat pool. 'all' (train-only): explode each "
                    "chunk into one entry per caption so every caption is "
                    "trained each epoch. 'one_per_field' (train-only): explode "
                    "one entry per caption field, sampling one caption from "
                    "within that field each epoch (equal weight per field). "
                    "Exploded entries share the chunk's idx (multi-positive); "
                    "'all'/'one_per_field' require train.loss_type="
                    "internvideo2_vtc and are ignored for val/eval.",
        display_name="Caption Mode",
    )
    idx_mode: str = STR_FIELD(
        value="sample_id",
        default_value="sample_id",
        valid_options="sample_id,video_id,category,field",
        description="How to build InternVideo2 VTC positive ids.",
        display_name="IDX Mode",
    )
    idx_field: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Metadata field used when idx_mode='field'.",
        display_name="IDX Field",
    )
    task_type: str = STR_FIELD(
        value="retrieval",
        default_value="retrieval",
        valid_options="retrieval,classification",
        description="Dataset task type. 'classification' groups samples by "
                    "their category label (e.g. Vad-R1 anomaly_type) for "
                    "supervised-contrastive training; 'retrieval' uses "
                    "instance/video grouping. Governs how the dataset class "
                    "derives the category and default idx grouping.",
        display_name="Task Type",
    )
    anomaly_only: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="For Vad-R1 chunks, keep only anomaly chunks.",
        display_name="Anomaly Only",
    )
    num_frames: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Number of frames to sample from each video clip.",
        display_name="Number of Frames",
    )
    relevance_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional path to an explicit-relevance eval-query file "
                    "(e.g. a frozen domain_test.json with a 'queries' list of "
                    "{query, chunk_id, slice, relevant_clip_ids}). When set on "
                    "the val source, evaluation scores those text queries "
                    "against this dataset as the shared gallery, per slice, "
                    "using the per-query relevant_clip_ids. When null, "
                    "evaluation uses the default idx-grouped retrieval.",
        display_name="Relevance File",
    )


@dataclass
class VideoCLIPDataLoaderConfig:
    """Base dataloader configuration shared by train and validation."""

    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Batch size per GPU.",
        display_name="Batch Size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Number of data loading worker processes.",
        display_name="Number of Workers",
    )


@dataclass
class VideoCLIPTrainDataConfig(VideoCLIPDataLoaderConfig):
    """Training data configuration with additional options for dataset type."""

    type: str = STR_FIELD(
        value="video_text",
        default_value="video_text",
        valid_options="video_text",
        description="Dataset type. Only 'video_text' (video-text retrieval "
                    "metadata) is supported; the legacy image-text 'custom' and "
                    "'wds' types were removed.",
        display_name="Dataset Type",
    )
    video_text: Optional[VideoCLIPVideoTextConfig] = DATACLASS_FIELD(
        VideoCLIPVideoTextConfig(),
        description="Video-text configuration (used when type='video_text').",
    )
    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Training batch size per GPU.",
        display_name="Batch Size",
    )


@dataclass
class VideoCLIPValDataConfig(VideoCLIPDataLoaderConfig):
    """Validation data configuration for retrieval evaluation."""

    type: str = STR_FIELD(
        value="video_text",
        default_value="video_text",
        valid_options="video_text",
        description="Validation dataset type. Only 'video_text' is supported; "
                    "the legacy image-text 'custom' type was removed.",
        display_name="Dataset Type",
    )
    video_text: Optional[VideoCLIPVideoTextConfig] = DATACLASS_FIELD(
        VideoCLIPVideoTextConfig(),
        description="Video-text validation configuration.",
    )


@dataclass
class VideoCLIPInferenceDataConfig(VideoCLIPDataLoaderConfig):
    """Inference corpus configuration (the gallery embedded/searched).

    Owns the inference video corpus directly so the dataloader is built from
    ``dataset.inference`` (no copying into ``dataset.val``).
    """

    type: str = STR_FIELD(
        value="video_text",
        default_value="video_text",
        valid_options="video_text",
        description="Inference corpus type (video_text metadata).",
        display_name="Dataset Type",
    )
    video_text: Optional[VideoCLIPVideoTextConfig] = DATACLASS_FIELD(
        VideoCLIPVideoTextConfig(),
        description="Video-text corpus configuration for inference.",
    )


@dataclass
class VideoCLIPMetricsConfig:
    """Evaluation metric configuration for video_clip validation/test."""

    mode: str = STR_FIELD(
        value="retrieval",
        default_value="retrieval",
        valid_options="retrieval,classification",
        description="Evaluation metric mode. 'retrieval' reports N-to-N "
                    "multi-relevant mAP / recall@k / hit@k; 'classification' "
                    "reports category-level MAP / MRR / macro P-R-F1 / top-K "
                    "(cosmos-embed1 parity).",
        display_name="Evaluation Mode",
    )
    exclude_categories: List[str] = LIST_FIELD(
        arrList=["Normal", "Abnormal"],
        default_value=["Normal", "Abnormal"],
        description="Category names excluded from classification MAP/MRR and "
                    "macro averages (case-insensitive).",
        display_name="Excluded Categories",
    )


@dataclass
class VideoCLIPDatasetConfig:
    """Dataset configuration for VideoCLIP training and evaluation."""

    train: VideoCLIPTrainDataConfig = DATACLASS_FIELD(
        VideoCLIPTrainDataConfig(),
        description="Training dataset configuration.",
    )
    val: VideoCLIPValDataConfig = DATACLASS_FIELD(
        VideoCLIPValDataConfig(),
        description="Validation dataset configuration.",
    )
    metrics: VideoCLIPMetricsConfig = DATACLASS_FIELD(
        VideoCLIPMetricsConfig(),
        description="Evaluation metric configuration (metric mode, excluded categories).",
    )
    inference: VideoCLIPInferenceDataConfig = DATACLASS_FIELD(
        VideoCLIPInferenceDataConfig(),
        description="Inference corpus configuration (gallery to embed/search).",
    )
    augmentation: VideoCLIPAugmentationConfig = DATACLASS_FIELD(
        VideoCLIPAugmentationConfig(),
        description="Data augmentation configuration.",
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Pin memory in DataLoader for faster GPU transfer.",
        display_name="Pin Memory",
    )
    seed: int = INT_FIELD(
        value=42,
        default_value=42,
        description="Random seed for data loading and shuffling.",
        display_name="Random Seed",
    )


# =============================================================================
# Training Config
# =============================================================================
@dataclass
class VideoCLIPOptimConfig:
    """Optimizer configuration for VideoCLIP training."""

    optimizer_type: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        valid_options="adamw,lamb",
        description="Optimizer type: 'adamw' (AdamW) or 'lamb' (LAMB).",
        display_name="Optimizer Type",
    )
    vision_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Learning rate for the vision encoder.",
        display_name="Vision LR",
    )
    text_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Learning rate for the text encoder.",
        display_name="Text LR",
    )
    weight_decay: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Weight decay (L2 regularization) coefficient.",
        display_name="Weight Decay",
    )
    betas: List[float] = LIST_FIELD(
        arrList=[0.9, 0.95],
        default_value=[0.9, 0.95],
        description="Adam/LAMB beta parameters [beta1, beta2] for momentum.",
        display_name="Betas",
    )
    eps: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        valid_min=0,
        description="Epsilon for numerical stability.",
        display_name="Epsilon",
    )
    warmup_steps: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=0,
        description="Number of linear warmup steps for learning rate.",
        display_name="Warmup Steps",
    )
    scheduler: str = STR_FIELD(
        value="cosine",
        default_value="cosine",
        valid_options="cosine,constant,linear",
        description="LR schedule after warmup: "
                    "'cosine' (cosine decay to 0), "
                    "'constant' (hold at base LR), "
                    "'linear' (linear decay to 0).",
        display_name="LR Scheduler",
    )


@dataclass
class VideoCLIPTrainConfig(TrainConfig):
    """VideoCLIP training configuration."""

    optim: VideoCLIPOptimConfig = DATACLASS_FIELD(
        VideoCLIPOptimConfig(),
        description="Optimizer configuration with per-tower learning rates.",
    )
    loss_type: str = STR_FIELD(
        value="internvideo2_vtc",
        default_value="internvideo2_vtc",
        valid_options="siglip,clip,internvideo2_vtc",
        description="Contrastive loss function: 'siglip' (sigmoid), "
                    "'clip' (softmax), or 'internvideo2_vtc' "
                    "(InternVideo2 video-text contrastive loss with idx positives).",
        display_name="Loss Type",
    )
    precision: str = STR_FIELD(
        value="fp16",
        default_value="fp16",
        valid_options="fp16,fp32,bf16",
        description="Training precision: fp16 (mixed), fp32 (full), or bf16 (bfloat16).",
        display_name="Precision",
    )
    grad_clip_norm: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Maximum gradient norm for clipping. Set to None to disable.",
        display_name="Gradient Clip Norm",
    )
    grad_checkpointing: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable gradient checkpointing to reduce memory at cost of speed.",
        display_name="Gradient Checkpointing",
    )
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        default_value="ddp",
        valid_options="ddp,fsdp",
        description="Distributed training strategy: 'ddp' or 'fsdp' (fully sharded).",
        display_name="Distributed Strategy",
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to pretrained model checkpoint for fine-tuning.",
        display_name="Pretrained Model Path",
    )
    val_check_interval: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Run validation every N training steps. If None, validates at end of epoch.",
        display_name="Validation Check Interval",
    )


# =============================================================================
# Inference/Eval Config
# =============================================================================
@dataclass
class VideoCLIPSearchConfig:
    """Text->video search / retrieval ranking over the extracted embeddings.

    Shared by the evaluate and inference tasks. Disabled by default: when
    ``enabled`` is False (and inference ``mode`` is not 'retrieval') the task
    only extracts embeddings. When active, the task ranks, for each text query,
    the most similar video clips and writes the retrieval results +
    ``similarity_stats.json``. Dataset-agnostic: operates only on the generic
    video/text embeddings.
    """

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Run text->video search over the extracted embeddings.",
        display_name="Enable Search",
    )
    search_metric: str = STR_FIELD(
        value="cosine",
        default_value="cosine",
        valid_options="cosine,knn",
        description="Ranking metric: 'cosine' (cosine similarity, higher is "
                    "closer) or 'knn' (Euclidean/L2 distance, smaller is "
                    "closer).",
        display_name="Search Metric",
    )
    normalize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="L2-normalize embeddings before scoring. On normalized "
                    "vectors cosine and knn give the same ranking.",
        display_name="Normalize Embeddings",
    )
    top_k: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        description="Number of video clips returned per text query.",
        display_name="Top K",
    )


@dataclass
class VideoCLIPRunConfig:
    """Shared run configuration for the evaluate and inference tasks.

    Holds the fields common to both: checkpoint, output/GPU settings, batch
    sizing, the video-text source, embeddings-cache controls, and the shared
    search/retrieval sub-config. The evaluate and inference configs extend this
    base with their task-specific fields.
    """

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Required path to a trained model checkpoint (.ckpt or .pth) "
                    "for evaluation or inference.",
        display_name="Checkpoint Path",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save inference/evaluation results.",
        display_name="Results Directory",
    )
    num_gpus: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Number of GPUs to use.",
        display_name="Number of GPUs",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="List of GPU device IDs to use.",
        display_name="GPU IDs",
    )
    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Batch size per GPU.",
        display_name="Batch Size",
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Number of data loading worker processes.",
        display_name="Number of Workers",
    )
    video_text: Optional[VideoCLIPVideoTextConfig] = DATACLASS_FIELD(
        VideoCLIPVideoTextConfig(),
        description="Video-text source configuration.",
    )
    video_embeddings_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Explicit path for the video embeddings .h5. If it exists "
                    "it is reused (generation skipped); if set but missing it "
                    "is generated there. Defaults to "
                    "<results_dir>/video_embeddings.h5.",
        display_name="Video Embeddings File",
    )
    text_embeddings_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Explicit path for the text embeddings .h5 (reusable "
                    "query/label cache). Reused if present, generated if "
                    "missing. Defaults to <results_dir>/text_embeddings.h5.",
        display_name="Text Embeddings File",
    )
    overwrite_embeddings: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Regenerate embeddings even if a cached file is present.",
        display_name="Overwrite Embeddings",
    )
    search: VideoCLIPSearchConfig = DATACLASS_FIELD(
        VideoCLIPSearchConfig(),
        description="Search / retrieval ranking over the embeddings.",
    )


@dataclass
class VideoCLIPEvaluateConfig(VideoCLIPRunConfig):
    """Evaluation task configuration (retrieval / classification metrics).

    Extends the shared run config with evaluate-only fields. The evaluation
    metric mode and excluded categories live under ``dataset.metrics``.
    """

    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to TensorRT engine for TRT-based evaluation/inference.",
        display_name="TRT Engine Path",
    )
    text_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file with prompts for text embedding extraction.",
        display_name="Text File",
    )


# =============================================================================
# Inference Config
# =============================================================================
@dataclass
class VideoCLIPQueryConfig:
    """Inline ad-hoc inference queries (texts and/or video files)."""

    input_texts: List[str] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Inline text queries to embed / search with.",
        display_name="Input Texts",
    )
    input_videos: List[str] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Inline video file paths to embed / search with.",
        display_name="Input Videos",
    )
    text_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Optional file with one text prompt per line (convenience "
                    "for large query lists; merged with input_texts).",
        display_name="Text File",
    )
    num_frames: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Frames sampled per inline video query.",
        display_name="Query Num Frames",
    )


@dataclass
class VideoCLIPInferenceConfig(VideoCLIPRunConfig):
    """Inference task configuration for the video_clip model.

    Extends the shared run config (checkpoint, results_dir, GPU/batch settings,
    embeddings-cache controls, and the ``search`` sub-config) with the inference
    ``mode`` and inline ``query``.

    ``mode='embeddings'`` extracts embeddings for whatever sources are present
    (corpus -> video_embeddings.h5, text queries -> text_embeddings.h5, video
    queries -> query_video_embeddings.h5). ``mode='retrieval'`` embeds the
    queries and the ``dataset.inference`` corpus and writes top-k matches
    (ranking controlled by ``search``) to retrieval_results.json. Corpus
    extraction is multi-GPU (DDP) and cache-aware.
    """

    mode: str = STR_FIELD(
        value="embeddings",
        default_value="embeddings",
        valid_options="embeddings,retrieval",
        description="'embeddings' extracts embeddings; 'retrieval' ranks the "
                    "corpus against the queries and writes top-k matches "
                    "(see the 'search' sub-config).",
        display_name="Inference Mode",
    )
    query: VideoCLIPQueryConfig = DATACLASS_FIELD(
        VideoCLIPQueryConfig(),
        description="Inline ad-hoc text/video queries.",
    )


# =============================================================================
# Export Config
# =============================================================================
@dataclass
class VideoCLIPExportConfig:
    """ONNX export configuration for VideoCLIP models."""

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to trained model checkpoint (.ckpt or .pth). "
                    "If null, exports directly from HuggingFace pretrained weights.",
        display_name="Checkpoint Path",
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Output ONNX file path (without extension for 'separate' encoder_type).",
        display_name="ONNX File Path",
    )
    encoder_type: str = STR_FIELD(
        value="combined",
        default_value="combined",
        valid_options="combined,separate",
        description="Export mode: 'combined' (single ONNX with both encoders), "
                    "'separate' (two ONNX files: vision and text).",
        display_name="Encoder Type",
    )
    opset_version: int = INT_FIELD(
        value=23,
        default_value=23,
        valid_min=11,
        valid_max=23,
        description="ONNX opset version for export.",
        display_name="ONNX Opset Version",
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="ONNX batch mode: -1 exports a symbolic dynamic batch; "
                    "a positive value exports a fixed batch. The internal "
                    "dynamic tracing sample does not set a runtime maximum.",
        display_name="Batch Size",
    )
    input_height: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=32,
        description="Input image height for vision encoder export.",
        display_name="Input Height",
    )
    input_width: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=32,
        description="Input image width for vision encoder export.",
        display_name="Input Width",
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU device ID to use for export.",
        display_name="GPU ID",
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="If True, export on CPU instead of GPU.",
        display_name="On CPU",
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of channels in the input image.",
        display_name="Input Channel",
        valid_min=1,
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable verbose ONNX export logging.",
        display_name="Verbose",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save exported ONNX models.",
        display_name="Results Directory",
    )


# =============================================================================
# TRT Engine Config
# =============================================================================
@dataclass
class VideoCLIPTrtConfig(TrtConfig):
    """VideoCLIP TensorRT configuration."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        valid_options="fp32,fp16",
        description="TensorRT precision: FP32 or FP16.",
        display_name="Data Type",
    )
    opt_batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description="Preferred batch size for TensorRT tactic optimization.",
        display_name="Optimum batch size",
        popular="yes",
    )
    max_batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Maximum batch size in the TRT optimization profile. "
                    "Matches the default inference batch size of 16.",
        display_name="Maximum batch size",
        popular="yes",
    )


@dataclass
class VideoCLIPGenTrtEngineConfig(GenTrtEngineConfig):
    """VideoCLIP TRT engine generation config."""

    tensorrt: VideoCLIPTrtConfig = DATACLASS_FIELD(VideoCLIPTrtConfig())


# =============================================================================
# Experiment Config
# =============================================================================
@dataclass
class VideoCLIPExperimentConfig(CommonExperimentConfig):
    """VideoCLIP (InternVideo2-CLIP) experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Network name for task invocation (TAO launcher metadata). "
                    "Defaults to 'video_clip'. This is NOT the architecture "
                    "selector -- use model.type for that.",
        display_name="Model Name",
    )
    model: VideoCLIPModelConfig = DATACLASS_FIELD(
        VideoCLIPModelConfig(),
        description="Model config.",
    )
    dataset: VideoCLIPDatasetConfig = DATACLASS_FIELD(
        VideoCLIPDatasetConfig(),
        description="Dataset config.",
    )
    train: VideoCLIPTrainConfig = DATACLASS_FIELD(
        VideoCLIPTrainConfig(),
        description="Training config.",
    )
    evaluate: VideoCLIPEvaluateConfig = DATACLASS_FIELD(
        VideoCLIPEvaluateConfig(),
        description="Evaluation config.",
    )
    inference: VideoCLIPInferenceConfig = DATACLASS_FIELD(
        VideoCLIPInferenceConfig(),
        description="Inference config.",
    )
    peft: VideoCLIPPEFTConfig = DATACLASS_FIELD(
        VideoCLIPPEFTConfig(),
        description="Parameter-efficient fine-tuning config (LoRA). "
                    "Disabled by default.",
    )
    regularization: VideoCLIPRegularizationConfig = DATACLASS_FIELD(
        VideoCLIPRegularizationConfig(),
        description="Geometry-preserving regularization config. "
                    "Disabled by default.",
    )
    export: VideoCLIPExportConfig = DATACLASS_FIELD(
        VideoCLIPExportConfig(),
        description="Export config.",
    )
    gen_trt_engine: VideoCLIPGenTrtEngineConfig = DATACLASS_FIELD(
        VideoCLIPGenTrtEngineConfig(),
        description="TensorRT engine generation config.",
    )

    def __post_init__(self):
        """Set the default network name for VideoCLIP."""
        if self.model_name is None:
            self.model_name = "video_clip"
