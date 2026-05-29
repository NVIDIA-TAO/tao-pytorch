# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP experiment configuration."""

from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING

from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    GenTrtEngineConfig,
    TrainConfig,
    TrtConfig,
)
from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


# =============================================================================
# Model Config
# =============================================================================
@dataclass
class CLIPModelConfig:
    """CLIP model configuration."""

    type: str = STR_FIELD(
        value="siglip2-so400m-patch16-256",
        default_value="siglip2-so400m-patch16-256",
        description="CLIP model type. "
                    "C-RADIO: c-radio_v3-h, c-radio_v3-l, c-radio_v3-b, c-radio_v3-g; "
                    "SigLIP2: siglip2-so400m-patch16-naflex (NaFlex), siglip2-so400m-patch14-224, "
                    "siglip2-so400m-patch14-384, siglip2-so400m-patch16-256, "
                    "siglip2-so400m-patch16-384, siglip2-so400m-patch16-512; "
                    "OpenCLIP: ViT-L-14-SigLIP-CLIPA-224, ViT-L-14-SigLIP-CLIPA-336, "
                    "ViT-H-14-SigLIP-CLIPA-224.",
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
        value=256,
        default_value=256,
        description="Input image resolution for training transforms. "
                    "Common values: 224 (RADIO/OpenCLIP), 384 (SigLIP2-g), "
                    "256 (SigLIP2-so400m). "
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


# =============================================================================
# PEFT Config
# =============================================================================
@dataclass
class CLIPLoRATargetConfig:
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
        arrList=["q_proj", "k_proj", "v_proj", "out_proj"],
        default_value=["q_proj", "k_proj", "v_proj", "out_proj"],
        description="Module name substrings to target for LoRA injection. "
                    "SigLIP2: 'q_proj', 'k_proj', 'v_proj', 'out_proj'. "
                    "RADIO/OpenCLIP: 'qkv', 'proj' (fused attention). "
                    "Matching is by substring on the module name within each block.",
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
class CLIPPEFTConfig:
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
    vision: CLIPLoRATargetConfig = DATACLASS_FIELD(
        CLIPLoRATargetConfig(),
        description="LoRA configuration for the vision encoder.",
    )
    text: CLIPLoRATargetConfig = DATACLASS_FIELD(
        CLIPLoRATargetConfig(),
        description="LoRA configuration for the text encoder.",
    )


# =============================================================================
# Regularization Config
# =============================================================================
@dataclass
class CLIPRegularizationConfig:
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
class CLIPAugmentationConfig:
    """Data augmentation configuration for CLIP training.

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
class CLIPDataPathConfig:
    """Dataset path configuration for custom image-text datasets."""

    image_dir: str = STR_FIELD(
        value=MISSING,
        default_value=MISSING,
        description="Directory containing the images.",
        display_name="Image Directory",
    )
    image_list_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file listing image filenames. If None, all images in image_dir are used.",
        display_name="Image List File",
    )
    caption_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory containing text caption files (.txt). "
                    "If None, captions are expected in image_dir alongside images.",
        display_name="Captions Directory",
    )
    caption_file_suffix: str = STR_FIELD(
        value=".txt",
        default_value=".txt",
        description="File extension for caption files. "
                    "Caption filename = image_basename + caption_file_suffix (e.g., 'image.png' -> 'image.txt').",
        display_name="Caption File Suffix",
    )


@dataclass
class CLIPWDSConfig:
    """WebDataset (sharded) configuration for large-scale CLIP training."""

    root_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Root directory containing WebDataset shards (required when type='wds').",
        display_name="Root Directory",
    )
    shard_list_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to text file listing shard URLs/paths.",
        display_name="Shard List File",
    )
    samples_per_shard: int = INT_FIELD(
        value=10000,
        default_value=10000,
        valid_min=1,
        description="Number of samples per shard (used for progress tracking).",
        display_name="Samples per Shard",
    )


@dataclass
class CLIPDataLoaderConfig:
    """Base dataloader configuration shared by train and validation."""

    datasets: List[CLIPDataPathConfig] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="List of dataset path configurations.",
        display_name="Datasets",
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


@dataclass
class CLIPTrainDataConfig(CLIPDataLoaderConfig):
    """Training data configuration with additional options for dataset type."""

    type: str = STR_FIELD(
        value="custom",
        default_value="custom",
        valid_options="wds,custom",
        description="Dataset type: 'custom' for filesystem-based or 'wds' for WebDataset.",
        display_name="Dataset Type",
    )
    wds: Optional[CLIPWDSConfig] = DATACLASS_FIELD(
        CLIPWDSConfig(),
        description="WebDataset configuration (used when type='wds').",
    )
    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        description="Training batch size per GPU.",
        display_name="Batch Size",
    )


@dataclass
class CLIPValDataConfig(CLIPDataLoaderConfig):
    """Validation data configuration for retrieval evaluation."""

    pass


@dataclass
class CLIPDatasetConfig:
    """Dataset configuration for CLIP training and evaluation."""

    train: CLIPTrainDataConfig = DATACLASS_FIELD(
        CLIPTrainDataConfig(),
        description="Training dataset configuration.",
    )
    val: CLIPValDataConfig = DATACLASS_FIELD(
        CLIPValDataConfig(),
        description="Validation dataset configuration.",
    )
    augmentation: CLIPAugmentationConfig = DATACLASS_FIELD(
        CLIPAugmentationConfig(),
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
class CLIPOptimConfig:
    """Optimizer configuration for CLIP training."""

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
class CLIPTrainConfig(TrainConfig):
    """CLIP training configuration."""

    optim: CLIPOptimConfig = DATACLASS_FIELD(
        CLIPOptimConfig(),
        description="Optimizer configuration with per-tower learning rates.",
    )
    loss_type: str = STR_FIELD(
        value="siglip",
        default_value="siglip",
        valid_options="siglip,clip",
        description="Contrastive loss function: 'siglip' (sigmoid) or 'clip' (softmax).",
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
class CLIPInferenceEvalConfig(CLIPDataLoaderConfig):
    """Configuration for CLIP inference and evaluation.

    Inherits datasets, batch_size, num_workers from CLIPDataLoaderConfig.
    """

    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Path to trained model checkpoint (.ckpt or .pth). "
                    "Not required for TRT-based evaluation.",
        display_name="Checkpoint Path",
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
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="Directory to save inference/evaluation results.",
        display_name="Results Directory",
    )
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
# Export Config
# =============================================================================
@dataclass
class CLIPExportConfig:
    """ONNX export configuration for CLIP models."""

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
        value=17,
        default_value=17,
        valid_min=11,
        description="ONNX opset version for export.",
        display_name="ONNX Opset Version",
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="Export batch size. Use -1 for dynamic batch size.",
        display_name="Batch Size",
    )
    input_height: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=32,
        description="Input image height for vision encoder export.",
        display_name="Input Height",
    )
    input_width: int = INT_FIELD(
        value=256,
        default_value=256,
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
class CLIPTrtConfig(TrtConfig):
    """CLIP TensorRT configuration."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        valid_options="fp32,fp16",
        description="TensorRT precision: FP32 or FP16.",
        display_name="Data Type",
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
class CLIPGenTrtEngineConfig(GenTrtEngineConfig):
    """CLIP TRT engine generation config."""

    tensorrt: CLIPTrtConfig = DATACLASS_FIELD(CLIPTrtConfig())


# =============================================================================
# Experiment Config
# =============================================================================
@dataclass
class CLIPExperimentConfig(CommonExperimentConfig):
    """CLIP experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value="clip",
        default_value="clip",
        description="Name of model for task invocation.",
        display_name="Model Name",
    )
    model: CLIPModelConfig = DATACLASS_FIELD(
        CLIPModelConfig(),
        description="Model config.",
    )
    dataset: CLIPDatasetConfig = DATACLASS_FIELD(
        CLIPDatasetConfig(),
        description="Dataset config.",
    )
    train: CLIPTrainConfig = DATACLASS_FIELD(
        CLIPTrainConfig(),
        description="Training config.",
    )
    evaluate: CLIPInferenceEvalConfig = DATACLASS_FIELD(
        CLIPInferenceEvalConfig(),
        description="Evaluation config.",
    )
    inference: CLIPInferenceEvalConfig = DATACLASS_FIELD(
        CLIPInferenceEvalConfig(),
        description="Inference config.",
    )
    peft: CLIPPEFTConfig = DATACLASS_FIELD(
        CLIPPEFTConfig(),
        description="Parameter-efficient fine-tuning config (LoRA). "
                    "Disabled by default.",
    )
    regularization: CLIPRegularizationConfig = DATACLASS_FIELD(
        CLIPRegularizationConfig(),
        description="Geometry-preserving regularization config. "
                    "Disabled by default.",
    )
    export: CLIPExportConfig = DATACLASS_FIELD(
        CLIPExportConfig(),
        description="Export config.",
    )
    gen_trt_engine: CLIPGenTrtEngineConfig = DATACLASS_FIELD(
        CLIPGenTrtEngineConfig(),
        description="TensorRT engine generation config.",
    )
