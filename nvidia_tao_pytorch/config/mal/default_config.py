# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    EvaluateConfig,
    CommonExperimentConfig,
    InferenceConfig,
    TrainConfig
)


@dataclass
class MALInferenceExpConfig(InferenceConfig):
    """Inference configuration template."""

    ann_path: str = STR_FIELD(value="")
    img_dir: str = STR_FIELD(value="")
    label_dump_path: str = STR_FIELD(value="instances_val2017_mal.json")
    batch_size: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf")
    load_mask: bool = BOOL_FIELD(value=False)


@dataclass
class MALEvalExpConfig(EvaluateConfig):
    """Evaluation configuration template."""

    batch_size: int = INT_FIELD(value=3, default_value=3, valid_min=1, valid_max="inf")
    use_mixed_model_test: bool = BOOL_FIELD(value=False)
    use_teacher_test: bool = BOOL_FIELD(value=False)
    comp_clustering: bool = BOOL_FIELD(value=False)
    use_flip_test: bool = BOOL_FIELD(value=False)


@dataclass
class MALDatasetConfig:
    """Data configuration template."""

    type: str = STR_FIELD(value='coco', default_value="coco", valid_options="coco", display_name="dataset type")
    train_ann_path: str = STR_FIELD(
        value='',
        display_name="Annotation path of the training set"
    )
    train_img_dir: str = STR_FIELD(
        value='',
        display_name="Image directory of the training set"
    )
    val_ann_path: str = STR_FIELD(
        value='',
        display_name="Annotation path of the validation set"
    )
    val_img_dir: str = STR_FIELD(value='', display_name="Image directory of the validation set")
    min_obj_size: float = FLOAT_FIELD(value=2048, default_value=2048, display_name="minimum object size")
    max_obj_size: float = FLOAT_FIELD(value=1e10, default_value="1.00E+10", display_name="maximum object size")
    num_workers_per_gpu: int = INT_FIELD(value=2, default_value=2)
    load_mask: bool = BOOL_FIELD(value=True, display_name="Whether to load segmentation mask in annotation file")
    crop_size: int = INT_FIELD(value=512, default_value=512, valid_min=256, valid_max="inf")


@dataclass
class MALModelConfig:
    """Model configuration template."""

    arch: str = STR_FIELD(
        value='vit-mae-base/16',
        value_type="ordered",
        default_value="vit-mae-base/16",
        valid_options=(
            "vit-deit-tiny/16,vit-deit-small/16,vit-mae-base/16,"
            "vit-mae-large/16,vit-mae-huge/14"
        )
    )
    frozen_stages: List[int] = LIST_FIELD(arrList=[-1], default_value=[-1], value_type="list_1_backbone")
    mask_head_num_convs: int = INT_FIELD(value=4, default_value=4, valid_min=1, valid_max="inf")
    mask_head_hidden_channel: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf")
    mask_head_out_channel: int = INT_FIELD(value=256, default_value=256, valid_min=1, valid_max="inf")
    teacher_momentum: float = FLOAT_FIELD(value=0.996, default_value=0.996, valid_min=0, valid_max=1)
    not_adjust_scale: bool = BOOL_FIELD(value=False)
    mask_scale_ratio_pre: int = INT_FIELD(value=1)
    mask_scale_ratio: float = FLOAT_FIELD(value=2.0)
    vit_dpr: float = FLOAT_FIELD(value=0)


@dataclass
class MALTrainExpConfig(TrainConfig):
    """Train configuration template."""

    batch_size: int = INT_FIELD(value=3, default_value=1, valid_min=1, valid_max="inf")
    accum_grad_batches: int = INT_FIELD(value=1, default_value=1, valid_min=1, valid_max="inf")
    use_amp: bool = BOOL_FIELD(value=True)
    pretrained_model_path: Optional[str] = STR_FIELD(value=None, default_value="")

    # optim
    optim_type: str = STR_FIELD(value='adamw', valid_options="adamw")
    optim_momentum: float = FLOAT_FIELD(value=0.9, default_value=0.9, valid_min=0, valid_max=1)
    lr: float = FLOAT_FIELD(value=0.000001, default_value=0.000001, valid_min=0, valid_max="inf")
    min_lr: float = FLOAT_FIELD(value=0)
    min_lr_rate: float = FLOAT_FIELD(value=0.2, default_value=0.02, valid_min=0, valid_max=1)
    num_wave: float = FLOAT_FIELD(value=1)
    wd: float = FLOAT_FIELD(value=0.0005)
    optim_eps: float = FLOAT_FIELD(value=1e-8)
    optim_betas: List[float] = LIST_FIELD([0.9, 0.9])
    warmup_epochs: int = INT_FIELD(value=1, default_value=1, valid_min=0, valid_max="inf")

    margin_rate: List[float] = LIST_FIELD([0, 1.2])
    test_margin_rate: List[float] = LIST_FIELD([0.6, 0.6])
    mask_thres: List[float] = LIST_FIELD([0.1])

    # loss
    loss_mil_weight: float = FLOAT_FIELD(value=4)
    loss_crf_weight: float = FLOAT_FIELD(value=0.5)

    # crf
    crf_zeta: float = FLOAT_FIELD(value=0.1)
    crf_kernel_size: int = INT_FIELD(value=3)
    crf_num_iter: int = INT_FIELD(value=100)
    loss_crf_step: int = INT_FIELD(value=4000)
    loss_mil_step: int = INT_FIELD(value=1000)
    crf_size_ratio: int = INT_FIELD(value=1)
    crf_value_high_thres: float = FLOAT_FIELD(value=0.9)
    crf_value_low_thres: float = FLOAT_FIELD(value=0.1)


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration template."""

    dataset: MALDatasetConfig = DATACLASS_FIELD(MALDatasetConfig())
    train: MALTrainExpConfig = DATACLASS_FIELD(MALTrainExpConfig())
    model: MALModelConfig = DATACLASS_FIELD(MALModelConfig())
    inference: MALInferenceExpConfig = DATACLASS_FIELD(MALInferenceExpConfig())
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(MALEvalExpConfig())

    def __post_init__(self):
        """Set default model name for MAL."""
        if self.model_name is None:
            self.model_name = "mal"
