# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    INT_FIELD,
    DICT_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import CommonExperimentConfig
from nvidia_tao_pytorch.config.grounding_dino.default_config import (
    GDINODatasetConfig,
    GDINOModelConfig,
    GDINOTrainExpConfig,
    GDINOInferenceExpConfig,
    GDINOEvalExpConfig,
    GDINOExportExpConfig,
    GDINOGenTrtEngineExpConfig,
)
from nvidia_tao_pytorch.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


@dataclass
class MaskGDINOEvalExpConfig(GDINOEvalExpConfig):
    """eval config"""

    ioi_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the intersection over instance (ioi) threshold
                    between rela output and segmentation mask to be used when
                    filtering out the final list of mask and box.""",
        display_name="ioi threshold"
    )

    nms_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="""The value of the nms threshold to be used when
                    filtering out the final list of mask and box using nms.""",
        display_name="nms threshold"
    )

    text_threshold: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="""The value of the text threshold to be used when
                    aligning output with expression.""",
        display_name="text threshold"
    )


@dataclass
class MaskGDINOInferenceExpConfig(GDINOInferenceExpConfig):
    """Inference config"""

    ioi_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the intersection over instance (ioi) threshold
                    between rela output and segmentation mask to be used when
                    filtering out the final list of mask and box.""",
        display_name="ioi threshold"
    )

    nms_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="""The value of the nms threshold to be used when
                    filtering out the final list of mask and box using nms.""",
        display_name="nms threshold"
    )

    text_threshold: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="""The value of the text threshold to be used when
                    aligning output with expression.""",
        display_name="text threshold"
    )


@dataclass
class MaskGDINODatasetConfig(GDINODatasetConfig):
    """Dataset config."""

    has_mask: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="has mask",
        description="Flag to load mask annotation from dataset."
    )
    val_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "json_file": "", "data_type": "VG"},
        description=(
            "The data source for validation:\n"
            "* image_dir : The directory that contains the validation images\n"
            "* json_file : The path of the JSON file, which uses validation-annotation COCO format.\n"
            "* data_type : The type of the dataset, OD or VG"
            "Note that category id needs to start from 0 if we want to calculate validation loss.\n"
            "Run Data Services annotation convert to making the categories contiguous."
        ),
        display_name="validation data sources",
    )
    test_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "json_file": "", "data_type": ""},
        description=(
            "The data source for testing:\n"
            "* image_dir : The directory that contains the test images\n"
            "* json_file : The path of the JSON file, which uses test-annotation COCO format.\n"
            "* data_type : The type of the dataset, OD or VG."
        ),
        display_name="test data sources",
    )
    infer_data_sources: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "data_type": ""},
        description=(
            "The data source for inference:\n"
            "* image_dir : Parent directory containing inference images\n"
            "* json_file : Path to JSON file with image_path+caption pairs (VG only)\n"
            "* data_type : Dataset type (VG, OD)\n"
            "* captions  : Class list string (OD only)"
        ),
        display_name="infer data sources",
    )


@dataclass
class MaskGDINOModelConfig(GDINOModelConfig):
    """DINO model config."""

    enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of encoder layers in the transformer (fixed at 6 for Mask Grounding DINO)",
        valid_min=6,
        valid_max=6,
        automl_enabled="FALSE",
        display_name="encoder layers",
    )
    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of decoder layers in the transformer (fixed at 6 for Mask Grounding DINO)",
        valid_min=6,
        valid_max=6,
        automl_enabled="FALSE",
        display_name="decoder layers",
    )
    has_mask: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="has mask",
        description="Flag to enable mask head in grounding dino."
    )
    num_region_queries: int = INT_FIELD(
        value=100,
        default_value=100,
        description="Number of region queries.",
        display_name="num_region_queries",
    )
    loss_types: List[str] = LIST_FIELD(
        arrList=['labels', 'boxes', 'masks'],
        description="Losses to be used during training",
        display_name="loss_types",
    )
    mask_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the mask error in the final loss.",
        display_name="Mask loss coefficient",
    )
    rela_nt_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the no target error in the final loss.",
        display_name="Rela no target loss coefficient",
    )
    rela_minimap_loss_coef: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the minimap error in the final loss.",
        display_name="Rela minimap loss coefficient",
    )
    rela_union_mask_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the mask error in the final loss.",
        display_name="Rela union mask loss coefficient",
    )
    dice_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description=(
            "The relative weight of the dice loss of the segmentation "
            "in the final loss."
        ),
        display_name="GIoU loss coefficient",
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: MaskGDINOModelConfig = DATACLASS_FIELD(
        MaskGDINOModelConfig(),
        description=(
            "Configurable parameters to construct the model for a "
            "Mask Grounding DINO experiment."
        ),
    )
    dataset: MaskGDINODatasetConfig = DATACLASS_FIELD(
        MaskGDINODatasetConfig(),
        description=(
            "Configurable parameters to construct the dataset for a "
            "Mask Grounding DINO experiment."
        ),
    )
    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description=(
            "Configurable parameters to construct the trainer for a "
            "Mask Grounding DINO experiment."
        ),
    )
    evaluate: MaskGDINOEvalExpConfig = DATACLASS_FIELD(
        MaskGDINOEvalExpConfig(),
        description=(
            "Configurable parameters to construct the evaluator for a "
            "Mask Grounding DINO experiment."
        ),
    )
    inference: MaskGDINOInferenceExpConfig = DATACLASS_FIELD(
        MaskGDINOInferenceExpConfig(),
        description=(
            "Configurable parameters to construct the inferencer for a "
            "Mask Grounding DINO experiment."
        ),
    )
    export: GDINOExportExpConfig = DATACLASS_FIELD(
        GDINOExportExpConfig(),
        description=(
            "Configurable parameters to construct the exporter for a "
            "Mask Grounding DINO experiment."
        ),
    )
    gen_trt_engine: GDINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        GDINOGenTrtEngineExpConfig(),
        description=(
            "Configurable parameters to construct the TensorRT engine builder "
            "for a Mask Grounding DINO experiment."
        ),
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Configurable parameters to run model quantization for a Mask Grounding DINO experiment.",
    )

    def __post_init__(self):
        """Set default model name for Mask Grounding DINO."""
        if self.model_name is None:
            self.model_name = "mask_grounding_dino"
