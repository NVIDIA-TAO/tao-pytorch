# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Default Config.

The DINOv3 SSL family inherits aggressively from ``nvdinov2``: the dataset, head,
distillation, scheduler, optimizer and train/inference/export schemas are reused by
subclassing the nvdinov2 dataclasses. Only the genuinely new pieces are added here:

* a v3 ``map_params`` table with **patch-16** ViT entries (vit_s, vit_s_plus, vit_b, vit_l,
  vit_h_plus, vit_7b),
* RoPE backbone fields (``rope_theta`` etc.) and patch-16 defaults,
* a ``GramConfig`` for Gram anchoring (wired up in a later step), and
* a disabled ``lora`` stub for forward-compatibility.

DINOv3 is Meta IP implemented inside TAO; the family/endpoint is named ``dinov3``
(not ``nvdinov3``). ``nvdinov2`` stays frozen.
"""

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.nvdinov2.default_config import (
    BackboneConfig,
    NVDINOv2HeadConfig,
    NVDINOv2ModelDistillConfig,
    NVDINOv2TransformConfig,
    NVDINOv2DatasetConfig,
    NVDINOv2TrainExpConfig,
    NVDINOv2InferenceExpConfig,
    NVDINOv2ExportExpConfig,
    GenTrtEngineExpConfig,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    CuDNNConfig,
)

# DINOv3 patch-16 ViT architectures supported by the backbone config.
SUPPORTED_BACKBONES = [
    *["vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"]
]
SUPPORTED_IMAGE_SIZES = (256, 512, 768)


def validate_img_size(backbone_config):
    """Reject image sizes outside the DINOv3 schema enum.

    Args:
        backbone_config: Mapping- or attribute-style DINOv3 backbone configuration.

    Raises:
        ValueError: If ``img_size`` is not one of :data:`SUPPORTED_IMAGE_SIZES`.
    """
    try:
        img_size = backbone_config["img_size"]
    except TypeError:
        img_size = backbone_config.img_size
    if img_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"Invalid value for model.backbone.img_size: {img_size}. "
            f"Allowed values are: {list(SUPPORTED_IMAGE_SIZES)}."
        )


# DINOv3 ViT param map (patch-16). Distinct from the nvdinov2 (patch-14) map.
# FFN note: DINOv3 ViT-S/B/L use a standard MLP; ViT-S+/H+/7B use SwiGLU. The
# ``DinoV2VisionTransformer.__init__`` already accepts ``mlp_layer``, so the v3 build
# just passes the class named here ('mlp' -> timm Mlp, 'swiglu' -> SwiGLUFused).
map_params = {
    'embed_dim': {
        'vit_s': 384,
        'vit_s_plus': 384,
        'vit_b': 768,
        'vit_l': 1024,
        'vit_h_plus': 1280,
        'vit_7b': 4096,
    },
    'depth': {
        'vit_s': 12,
        'vit_s_plus': 12,
        'vit_b': 12,
        'vit_l': 24,
        'vit_h_plus': 32,
        'vit_7b': 40,
    },
    'num_heads': {
        'vit_s': 6,
        'vit_s_plus': 6,
        'vit_b': 12,
        'vit_l': 16,
        'vit_h_plus': 20,
        'vit_7b': 32,
    },
    'init_values': {
        'vit_s': 1e-5,
        'vit_s_plus': 1e-5,
        'vit_b': 1e-5,
        'vit_l': 1e-5,
        'vit_h_plus': 1e-5,
        'vit_7b': 1e-5,
    },
    'drop_path_schedule': {
        'vit_s': 'linear',
        'vit_s_plus': 'linear',
        'vit_b': 'linear',
        'vit_l': 'linear',
        'vit_h_plus': 'linear',
        'vit_7b': 'linear',
    },
    'num_classes': {
        'vit_s': 0,
        'vit_s_plus': 0,
        'vit_b': 0,
        'vit_l': 0,
        'vit_h_plus': 0,
        'vit_7b': 0,
    },
    'mlp_layer': {
        'vit_s': 'mlp',
        'vit_s_plus': 'swiglu',
        'vit_b': 'mlp',
        'vit_l': 'mlp',
        'vit_h_plus': 'swiglu',
        'vit_7b': 'swiglu',
    },
    # SwiGLU inner width = mlp_ratio * embed_dim per gate/value branch. ViT-S+/B/L/H+ use 4.0
    # (timm default); the public DINOv3 ViT-7B checkpoint uses 2.0. Verified against timm
    # 1.0.26 `vit_7b_patch16_dinov3` (eva.py), which sets `mlp_ratio=2` explicitly while H+
    # leaves it at the EVA default of 4.0. With swiglu_align_to=64 this yields fc1_g/fc1_x of
    # (8192, 4096) each, i.e. 2*dim, matching `fuse_timm_swiglu_fc1`'s expected fused shape.
    # Threaded into the backbone build via _resolve_arch.
    'mlp_ratio': {
        'vit_s': 4.0,
        'vit_s_plus': 4.0,
        'vit_b': 4.0,
        'vit_l': 4.0,
        'vit_h_plus': 4.0,
        'vit_7b': 2.0,
    },
}


@dataclass
class DINOv3BackboneConfig(BackboneConfig):
    """DINOv3 backbone config (patch-16 + RoPE).

    Subclasses the nvdinov2 ``BackboneConfig`` and overrides the patch-16 defaults,
    the ViT-B default backbone type, and adds the RoPE frequency base. The absolute
    positional embedding of DINOv2 is replaced by axial RoPE, so there is no
    ``pos_embed`` field; the ``DinoV3VisionTransformer`` does not register one.
    """

    teacher_type: str = STR_FIELD(
        value="vit_b",
        default_value="vit_b",
        display_name="teacher backbone",
        description=(
            "Teacher backbone name. TAO's DINOv3 supports vit_s, vit_s_plus, vit_b, vit_l, vit_h_plus and vit_7b."
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    student_type: str = STR_FIELD(
        value="vit_b",
        default_value="vit_b",
        display_name="student backbone",
        description=(
            "Student backbone name. TAO's DINOv3 supports vit_s, vit_s_plus, vit_b, vit_l, vit_h_plus and vit_7b."
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    num_register_tokens: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        valid_max="inf",
        description="Number of register tokens (DINOv3 ViT-B uses 4)",
        display_name="num register tokens",
        popular="yes"
    )
    patch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        description="Size of patches (DINOv3 uses patch 16)",
        display_name="patch size",
        valid_options="16",
        popular="yes"
    )
    img_size: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Backbone image size. Supported values are 256, 512, and 768.",
        display_name="image size",
        valid_options=",".join(str(size) for size in SUPPORTED_IMAGE_SIZES),
        popular="yes"
    )
    rope_theta: float = FLOAT_FIELD(
        value=100.0,
        default_value=100.0,
        description=(
            "Frequency base for 2D axial RoPE. Must match the timm DINOv3 reference; "
            "verified by the step-4 feature-parity smoke test."
        ),
        display_name="RoPE theta",
        popular="yes"
    )


@dataclass
class GramConfig:
    """DINOv3 Gram-anchoring config.

    Gram anchoring regularizes the student's patch-token Gram matrix toward a frozen
    Gram teacher (initialized from the loaded DINOv3 weights). The loss term itself is
    wired in a later step; these fields configure when/how strongly it applies.
    """

    enable: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to add the Gram-anchoring loss term",
        display_name="enable gram",
        popular="yes"
    )
    w_gram: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Weight of the Gram-anchoring loss term",
        display_name="gram weight",
        popular="yes"
    )
    start_step: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Global step at which the Gram term activates",
        display_name="gram start step",
        popular="yes"
    )
    teacher_source: str = STR_FIELD(
        value="pretrained",
        default_value="pretrained",
        description=(
            "Source of the frozen Gram teacher weights: 'pretrained' anchors to the loaded "
            "DINOv3 weights (Phase 0 default); 'ema' uses an early-EMA snapshot of the current "
            "run's teacher, refreshed every 'refresh_interval' steps (Phase 1 / high-res)."
        ),
        display_name="gram teacher source",
        valid_options="pretrained,ema",
        popular="no"
    )
    refresh_interval: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description=(
            "Steps between refreshing the Gram teacher from the EMA teacher (only when "
            "teacher_source='ema'). 0 = never refresh (frozen snapshot, Phase 0 behavior)."
        ),
        display_name="gram refresh interval",
        popular="no"
    )
    teacher_scale: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description=(
            "Resolution multiple at which the Gram teacher runs relative to the student. "
            "DINOv3 computes Gram on higher-res teacher features; the teacher grid is "
            "average-pooled back to the student grid before the loss. 1.0 = same resolution "
            "(default, memory-safe). The paper uses 2.0, but at 768 the teacher then runs at "
            "1536 (96x96=9216 tokens) -> heavy; raise to 1.5/2.0 only if memory allows."
        ),
        display_name="gram teacher scale",
        popular="no"
    )


@dataclass
class LoRAConfig:
    """Disabled LoRA stub for forward-compatibility (parameter-efficient SSL, Phase 2)."""

    enable: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to apply LoRA adapters (disabled in v1)",
        display_name="enable lora",
        popular="no"
    )
    rank: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="LoRA rank",
        display_name="lora rank",
        popular="no"
    )
    alpha: float = FLOAT_FIELD(
        value=16.0,
        default_value=16.0,
        description="LoRA scaling alpha",
        display_name="lora alpha",
        popular="no"
    )


@dataclass
class DINOv3ModelConfig:
    """DINOv3 model config (reuses nvdinov2 distill/head, adds gram + lora)."""

    centering_method: str = STR_FIELD(
        value="sinkhorn",
        default_value="sinkhorn",
        valid_options="sinkhorn,softmax",
        description=(
            "Teacher-output centering for the DINO/iBOT heads. DINOv3 uses Sinkhorn-Knopp "
            "(SwAV); 'softmax' is the DINOv2 fallback. If training shows instability/collapse, "
            "try 'softmax'."
        ),
        display_name="centering method",
        popular="yes"
    )
    distill: NVDINOv2ModelDistillConfig = DATACLASS_FIELD(
        NVDINOv2ModelDistillConfig(),
        description="Configuration for distillation (reused from nvdinov2)"
    )
    backbone: DINOv3BackboneConfig = DATACLASS_FIELD(
        DINOv3BackboneConfig(),
        description="Configuration for the DINOv3 backbone"
    )
    head: NVDINOv2HeadConfig = DATACLASS_FIELD(
        NVDINOv2HeadConfig(),
        description="Configuration for the DINOv3 head (reused from nvdinov2)"
    )
    gram: GramConfig = DATACLASS_FIELD(
        GramConfig(),
        description="Configuration for DINOv3 Gram anchoring"
    )
    lora: LoRAConfig = DATACLASS_FIELD(
        LoRAConfig(),
        description="Disabled LoRA stub for forward-compatibility"
    )


@dataclass
class DINOv3TransformConfig(NVDINOv2TransformConfig):
    """DINOv3 transform config with a 256 default and patch-16-friendly crop sizes."""

    global_crops_size: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Size of global crops for DINOv3 training.",
        display_name="Global Crops Size",
        popular="yes"
    )
    local_crops_size: int = INT_FIELD(
        value=112,
        default_value=112,
        valid_min=1,
        valid_max="inf",
        description="Size of local crops (multiple of patch 16)",
        display_name="Local Crops Size",
        popular="yes"
    )


@dataclass
class DINOv3DatasetConfig(NVDINOv2DatasetConfig):
    """DINOv3 dataset config (reuses nvdinov2 dataset, v3 transform defaults)."""

    transform: DINOv3TransformConfig = DATACLASS_FIELD(
        DINOv3TransformConfig(),
        description="Configuration parameters for data transformation",
        display_name="transform",
    )


@dataclass
class DINOv3CuDNNConfig(CuDNNConfig):
    """CuDNN defaults compatible with DINOv3 custom attention."""

    benchmark: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable CuDNN benchmarking for DINOv3 training.",
        display_name="CuDNN benchmark",
        popular="no",
    )
    deterministic: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Enable deterministic CuDNN behavior. Keep disabled when using "
            "DINOv3 custom attention, which has no deterministic backward implementation."
        ),
        display_name="CuDNN deterministic",
        popular="no",
    )


@dataclass
class DINOv3TrainExpConfig(NVDINOv2TrainExpConfig):
    """DINOv3 train config.

    Subclasses the nvdinov2 train config, corrects the inherited pretrained-weight
    contract, selects CuDNN defaults compatible with custom attention, and adds a
    ``distributed_strategy`` selector. The nvdinov2 default (Lightning ``'auto'`` ->
    single-device / DDP) is unchanged; FSDP (FULL_SHARD) is opt-in and is what enables
    high-resolution and the larger ViT-L / ViT-H+ backbones, where DDP's full per-GPU
    replication does not fit. The ``DINOV3_STRATEGY`` env var, kept for the de-risking
    smokes, overrides this field when set.
    """

    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        default_type=None,
        description=(
            "Path to DINOv3 pretrained weights matching the configured backbone. "
            "Accepts a timm-format directory or file, or a stripped TAO DINOv3 "
            "backbone checkpoint. DINOv2/NVDINOv2 checkpoints are not supported."
        )
    )
    distributed_strategy: str = STR_FIELD(
        value="auto",
        default_value="auto",
        valid_options="auto,ddp,fsdp",
        description=(
            "Lightning distributed strategy. 'auto' keeps the nvdinov2 behaviour "
            "(single-device or DDP). 'fsdp' shards params/grads/optimizer (FULL_SHARD) "
            "for high-resolution and large-backbone multi-GPU training."
        ),
        display_name="distributed strategy",
        popular="yes"
    )
    cudnn: DINOv3CuDNNConfig = DATACLASS_FIELD(
        DINOv3CuDNNConfig(),
        description="CuDNN settings compatible with DINOv3 custom attention.",
    )


@dataclass
class DINOv3ExportExpConfig(NVDINOv2ExportExpConfig):
    """DINOv3 export config (patch-16 trace shape).

    Overrides the nvdinov2 default ONNX trace shape (518, a patch-14 multiple):
    DINOv3 is patch-16 and defaults to 256, so the default trace matches the backbone
    ``img_size`` and stays divisible by the patch size.
    """

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description=(
            "Path to a stripped DINOv3 teacher checkpoint or a full Lightning training "
            "checkpoint. Export always selects the EMA teacher backbone."
        ),
        display_name="Path to checkpoint file"
    )
    input_width: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Input width",
        display_name="Input width",
        valid_min=128
    )
    input_height: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Input height",
        display_name="Input height",
        valid_min=128
    )


@dataclass
class DINOv3ConvertConfig:
    """DINOv3 backbone-export (``convert``) config.

    Converts an SSL-trained DINOv3 checkpoint into the timm-format layout that the
    ``cv/backbone_v2`` ``dinov3_vitb16`` registry entry (and downstream supervised tasks)
    consume. The EMA ``teacher`` is the recommended feature extractor for continual
    pre-training, so it is the default source.
    """

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Directory for convert results/logs",
        display_name="results dir"
    )
    checkpoint: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "SSL DINOv3 checkpoint to convert: a stripped backbone file "
            "(student_*.pth / teacher_*.pth) or a full Lightning .pth/.ckpt."
        ),
        display_name="checkpoint",
        popular="yes"
    )
    output_path: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Output path for the timm-format backbone (.safetensors or .pth). Defaults to "
            "<results_dir>/dinov3_<arch>_backbone.safetensors."
        ),
        display_name="output path",
        popular="yes"
    )
    source: str = STR_FIELD(
        value="teacher",
        default_value="teacher",
        valid_options="student,teacher,student_ema",
        description="Which SSL sub-model's backbone to export (EMA teacher recommended).",
        display_name="source",
        popular="yes"
    )
    validate: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Validate the converted state dict against a fresh timm DINOv3 model.",
        display_name="validate"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """DINOv3 experiment config."""

    model: DINOv3ModelConfig = DATACLASS_FIELD(
        DINOv3ModelConfig(),
        description="Configurable parameters to construct the model for a DINOv3 experiment.",
    )
    dataset: DINOv3DatasetConfig = DATACLASS_FIELD(
        DINOv3DatasetConfig(),
        description="Configurable parameters to construct the dataset for a DINOv3 experiment.",
    )
    train: DINOv3TrainExpConfig = DATACLASS_FIELD(
        DINOv3TrainExpConfig(),
        description="Configurable parameters to construct the trainer for a DINOv3 experiment.",
    )
    inference: NVDINOv2InferenceExpConfig = DATACLASS_FIELD(
        NVDINOv2InferenceExpConfig(),
        description="Configurable parameters to construct the inference trainer for a DINOv3 experiment.",
    )
    export: DINOv3ExportExpConfig = DATACLASS_FIELD(
        DINOv3ExportExpConfig(),
        description="Configurable parameters to export for a DINOv3 experiment.",
    )
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(
        GenTrtEngineExpConfig(),
        description="Configurable parameters to generate TensorRT engine for a DINOv3 experiment.",
    )
    convert: DINOv3ConvertConfig = DATACLASS_FIELD(
        DINOv3ConvertConfig(),
        description="Configurable parameters to convert an SSL backbone to the backbone_v2 (timm) layout.",
    )

    def __post_init__(self):
        """Set default model name for DINOv3."""
        if self.model_name is None:
            self.model_name = "dinov3"


# Expose a single MLP-layer registry so the pl_model resolves the param-map string
# ('mlp'/'swiglu') without importing torch at config-parse time.
MLP_LAYER_NAMES = ("mlp", "swiglu")
