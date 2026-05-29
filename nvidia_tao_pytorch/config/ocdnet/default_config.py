# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig,
    GenTrtEngineConfig,
    TrtConfig,
    CalibrationConfig
)
from nvidia_tao_pytorch.config.common.quantization import ModelQuantizationConfig


@dataclass
class QuantCalibrationDataset:
    """Quantization calibration dataset config."""

    images_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the directory containing calibration images.",
        display_name="calibration images directory"
    )


@dataclass
class OCDNetModelConfig:
    """Model config."""

    backbone: str = STR_FIELD(
        value="deformable_resnet18",
        default_value="deformable_resnet18",
        display_name="backbone",
        description="""The backbone name of the model.
                    It supports deformable_resnet18, deformable_resnet50 and fan_tiny_8_p4_hybrid.""",
        valid_options="deformable_resnet18,deformable_resnet50,fan_tiny_8_p4_hybrid",
        popular="yes",
    )
    pretrained: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to use pretrained model or not.",
        display_name="pretrained"
    )
    in_channels: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=3,
        valid_max=3,
        description="Number of input channels in FPN",
        display_name="in_channels",
    )
    neck: str = STR_FIELD(
        value="FPN",
        default_value="FPN",
        description="Neck name of the model.",
        display_name="neck",
        valid_options="FPN,FANNeck"
    )
    inner_channels: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=256,
        valid_max=256,
        description="Number of inner channels in FPN",
        display_name="inner_channels",
    )
    head: str = STR_FIELD(
        value="DBHead",
        default_value="DBHead",
        description="Head name of the model.",
        display_name="head"
    )
    out_channels: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=2,
        valid_max=2,
        description="Number of out channels",
        display_name="out_channels",
    )
    k: int = INT_FIELD(
        value=50,
        default_value=50,
        valid_min=1,
        valid_max="inf",
        description="Coefficient of Differentiable Binarization",
        display_name="k",
    )
    load_pruned_graph: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to load pruned model or not.",
        display_name="load_pruned_graph"
    )
    pruned_graph_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pruned model path",
        description="[Optional] Path to a pruned model file.",
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained model path",
        description="[Optional] Path to a pretrained model file.",
    )
    enlarge_feature_map_size: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enlarge the output feature map size of the FAN-tiny backbone",
        display_name="enlarge_feature_map_size"
    )
    activation_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to use activation checkpoints to save GPU memory, only for the FAN-tiny backbone",
        display_name="activation_checkpoint"
    )
    quant: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to do quantization",
        display_name="quant"
    )
    fuse_qkv_proj: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to fuse the qkv projection",
        display_name="fuse_qkv_proj"
    )


@dataclass
class Optimargs:
    """Optimargs config."""

    lr: float = FLOAT_FIELD(
        value=0.001,
        default_value=0.001,
        math_cond="> 0.0",
        valid_min=0.0,
        valid_max="inf",
        display_name="learning rate",
        description="The initial learning rate",
        popular="yes",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        math_cond=">= 0.0",
        valid_min=0.0,
        valid_max="inf",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )
    amsgrad: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to use AMSGrad as stochastic optimization method",
        display_name="amsgrad"
    )
    momentum: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="momentum - Adam",
        description="The momentum for the Adam optimizer.",
    )
    eps: float = FLOAT_FIELD(
        value=1e-8,
        default_value=1e-8,
        valid_min=0.0,
        valid_max=1.0,
        display_name="epsilon",
        description="The epsilon coefficient",
    )


@dataclass
class Optimizer:
    """Optimizer config."""

    type: str = STR_FIELD(
        value="Adam",
        default_value="Adam",
        description="Optimizer type.",
        display_name="type"
    )
    args: Optimargs = DATACLASS_FIELD(
        Optimargs(),
        description="Configurable parameters to construct the optimizer.",
    )


@dataclass
class Loss:
    """Loss config."""

    type: str = STR_FIELD(
        value="DBLoss",
        default_value="DBLoss",
        description="Loss function name.",
        display_name="type"
    )
    alpha: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        valid_max="inf",
        description="The alpha coefficient.",
        display_name="alpha",
    )
    beta: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        valid_max="inf",
        description="The beta coefficient",
        display_name="beta",
    )
    ohem_ratio: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="The ohem_ratio coefficient",
        display_name="ohem_ratio",
    )
    eps: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        valid_min=0.0,
        valid_max=1.0,
        display_name="epsilon",
        description="The epsilon coefficient.",
    )


@dataclass
class Postprocessingargs:
    """Postprocessingargs config."""

    thresh: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=1.0,
        display_name="thresh",
        description="The threshold for binarization.",
        popular="yes",
    )
    box_thresh: float = FLOAT_FIELD(
        value=0.55,
        default_value=0.55,
        valid_min=0.0,
        valid_max=1.0,
        display_name="box_thresh",
        description="The threshold for BBOX.",
        popular="yes",
    )
    max_candidates: int = INT_FIELD(
        value=1000,
        default_value=1000,
        valid_min=1,
        valid_max="inf",
        display_name="max_candidates",
        description="The maximum candidate BBOX.",
        popular="yes",
    )
    unclip_ratio: float = FLOAT_FIELD(
        value=1.5,
        default_value=1.5,
        valid_min=0.0,
        valid_max="inf",
        display_name="unclip_ratio",
        description="The unclip ratio using the Vatti clipping algorithm.",
        popular="yes",
    )


@dataclass
class Postprocessing:
    """Postprocessing config."""

    type: str = STR_FIELD(
        value="SegDetectorRepresenter",
        default_value="SegDetectorRepresenter",
        description="The postprocessing name.",
        display_name="type"
    )
    args: Postprocessingargs = DATACLASS_FIELD(
        Postprocessingargs(),
        description="Configurable parameters to construct the postprocessing.",
    )


@dataclass
class Metricargs:
    """Metricargs config."""

    is_output_polygon: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to output polygon or BBOX",
        display_name="is_output_polygon"
    )


@dataclass
class Metric:
    """Metric config."""

    type: str = STR_FIELD(
        value="QuadMetric",
        default_value="QuadMetric",
        description="The configuration for metric computing.",
        display_name="type"
    )
    args: Metricargs = DATACLASS_FIELD(
        Metricargs(),
        description="Configurable parameters to construct the metric computing.",
    )


@dataclass
class LRSchedulerargs:
    """LRSchedulerargs config."""

    warmup_epoch: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="The warmup epoch to the initial learning rate. Should be different from the num_epochs.",
        display_name="warmup_epoch",
    )


@dataclass
class LRScheduler:
    """LRScheduler config."""

    type: str = STR_FIELD(
        value="WarmupPolyLR",
        default_value="WarmupPolyLR",
        description="The learning scheduler.",
        display_name="type"
    )
    args: LRSchedulerargs = DATACLASS_FIELD(
        LRSchedulerargs(),
        description="Configurable parameters to construct the learning scheduler.",
    )


@dataclass
class Trainer:
    """Trainer config."""

    clip_grad_norm: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        display_name="clip gradient norm",
        description="""
        Amount to clip the gradient by L2 Norm.
        A value of 0.0 specifies no clipping.""",
    )


@dataclass
class Trainargs:
    """Train args config."""

    img_mode: str = STR_FIELD(
        value="BGR",
        default_value="BGR",
        description="The image mode.",
        display_name="img_mode",
        valid_options="BGR,RGB,GRAY",
    )
    filter_keys: List[str] = LIST_FIELD(
        arrList=['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags', 'shape'],
        description="List of ignored keys",
        display_name="filter_keys"
    )
    ignore_tags: List[str] = LIST_FIELD(
        arrList=['*', '###'],
        description="List of labels that are not used to train",
        display_name="ignore_tags"
    )
    pre_processes: Optional[List[Dict[Any, Any]]] = LIST_FIELD(
        arrList=[
            {
                "args": {
                    "keep_ratio": True,
                    "max_tries": 50,
                    "size": [640, 640]
                },
                "type": "EastRandomCropData"
            },
            {
                "args": {
                    "shrink_ratio": 0.4,
                    "thresh_max": 0.7,
                    "thresh_min": 0.3
                },
                "type": "MakeBorderMap"
            },
            {
                "args": {
                    "shrink_ratio": 0.4,
                    "min_text_size": 8
                },
                "type": "MakeShrinkMap"
            }
        ],
        description="The pre-processing configuration.",
        display_name="pre_processes"
    )


@dataclass
class Dataloader:
    """Train args config."""

    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=1,
        valid_max="inf",
        automl_enabled="FALSE",
        description="The batch size during training.",
        display_name="batch_size",
        popular="yes",
    )
    shuffle: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to shuffle the data or not.",
        display_name="shuffle"
    )
    pin_memory: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable pinned memory or not",
        display_name="pin_memory"
    )
    num_workers: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="The threads used to load data.",
        display_name="num_workers",
        popular="yes",
    )
    collate_fn: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="The collate function."
    )


@dataclass
class TrainDataset:
    """Train Dataset config."""

    data_name: str = STR_FIELD(
        value="ICDAR2015Dataset",
        default_value="ICDAR2015Dataset",
        description="The dataset type",
        display_name="data_name",
    )
    data_path: List[str] = LIST_FIELD(
        arrList=[],
        description="The list of training dataset paths",
        display_name="data_path"
    )
    args: Trainargs = DATACLASS_FIELD(
        Trainargs(),
        description="Configurable parameters to construct the training dataset.",
    )
    loader: Dataloader = DATACLASS_FIELD(
        Dataloader(),
        description="Configurable parameters to construct the training dataloader.",
    )


@dataclass
class Validateargs:
    """Validate args config."""

    img_mode: str = STR_FIELD(
        value="BGR",
        default_value="BGR",
        description="The image mode.",
        display_name="img_mode",
        valid_options="BGR,RGB,GRAY",
    )
    filter_keys: List[str] = LIST_FIELD(
        arrList=[''],
        description="List of ignored keys",
        display_name="filter_keys"
    )
    ignore_tags: List[str] = LIST_FIELD(
        arrList=['*', '###'],
        description="List of labels that are not used to evaluate",
        display_name="ignore_tags"
    )
    pre_processes: Optional[List[Dict[Any, Any]]] = LIST_FIELD(
        arrList=[{"args": {"resize_text_polys": True, "short_size": [1280, 736]}, "type": "Resize2D"}],
        description="The pre-processing configuration.",
        display_name="pre_processes"
    )


@dataclass
class Validateloader:
    """Validate args config."""

    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="The batch size during evaluation",
        display_name="batch_size",
        popular="yes",
    )
    shuffle: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to shuffle the data or not",
        display_name="shuffle"
    )
    pin_memory: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable pinned memory or not",
        display_name="pin_memory"
    )
    num_workers: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="The threads used to load data.",
        display_name="num_workers",
        popular="yes",
    )
    collate_fn: Optional[str] = STR_FIELD(
        value="ICDARCollateFN",
        default_value="ICDARCollateFN",
        description="The collate function."
    )


@dataclass
class ValidateDataset:
    """Validate Dataset config."""

    data_name: str = STR_FIELD(
        value="ICDAR2015Dataset",
        default_value="ICDAR2015Dataset",
        description="The dataset type",
        display_name="data_name",
    )
    data_path: List[str] = LIST_FIELD(
        arrList=[],
        description="The list of training dataset paths",
        display_name="data_path"
    )
    args: Validateargs = DATACLASS_FIELD(
        Validateargs(),
        description="Configurable parameters to construct the validation dataset.",
    )
    loader: Validateloader = DATACLASS_FIELD(
        Validateloader(),
        description="Configurable parameters to construct the validation dataloader.",
    )


@dataclass
class OCDNetDataConfig:
    """Dataset config."""

    train_dataset: TrainDataset = DATACLASS_FIELD(
        TrainDataset(),
        display_name="train_dataset",
        description="Hyper parameters to configure the training dataset."
    )
    validate_dataset: ValidateDataset = DATACLASS_FIELD(
        ValidateDataset(),
        display_name="validate_dataset",
        description="Hyper parameters to configure the validation dataset."
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for quantization calibration dataset.",
    )


@dataclass
class OCDNetTrainExpConfig(TrainConfig):
    """Train experiment config."""

    post_processing: Postprocessing = DATACLASS_FIELD(
        Postprocessing(),
        display_name="post_processing",
        description="Hyper parameters to configure the post_processing."
    )
    metric: Metric = DATACLASS_FIELD(
        Metric(),
        display_name="metric",
        description="Hyper parameters to configure the metric."
    )
    trainer: Trainer = DATACLASS_FIELD(
        Trainer(),
        display_name="trainer",
        description="Hyper parameters to configure the trainer."
    )
    loss: Loss = DATACLASS_FIELD(
        Loss(),
        display_name="loss",
        description="Hyper parameters to configure the loss."
    )
    optimizer: Optimizer = DATACLASS_FIELD(
        Optimizer(),
        display_name="optimizer",
        description="Hyper parameters to configure the optimizer."
    )
    lr_scheduler: LRScheduler = DATACLASS_FIELD(
        LRScheduler(),
        display_name="lr_scheduler",
        description="Hyper parameters to configure the learning rate scheduler."
    )
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        description="The training precision",
        display_name="precision",
    )
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        default_value="ddp",
        description="The strategy for distributed training",
        display_name="distributed_strategy",
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to run only one batch for debugging purposes",
        display_name="is_dry_run"
    )
    use_distributed_sampler: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use distributed sampler for multi-GPU training",
        display_name="use_distributed_sampler"
    )
    model_ema: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable model EMA",
        display_name="model_ema"
    )
    model_ema_decay: float = FLOAT_FIELD(
        value=0.9999,
        default_value=0.9999,
        valid_min=0.0,
        valid_max=1.0,
        display_name="model_ema_decay",
        description="The decay of model EMA",
    )


@dataclass
class OCDNetInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    input_folder: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="input_folder",
        description="The input folder for test images",
    )
    width: int = INT_FIELD(
        value=1280,
        default_value=1280,
        valid_min=1,
        display_name="width",
        description="The width for inference.",
        popular="yes",
    )
    height: int = INT_FIELD(
        value=736,
        default_value=736,
        valid_min=1,
        display_name="height",
        description="The height for inference.",
        popular="yes",
    )
    img_mode: str = STR_FIELD(
        value=MISSING,
        default_value="BGR",
        description="The image mode.",
        display_name="img_mode",
        valid_options="BGR,RGB,GRAY",
    )
    polygon: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to show the polygon",
        display_name="polygon"
    )
    show: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to show the pred image",
        display_name="show"
    )
    post_processing: Postprocessing = DATACLASS_FIELD(
        Postprocessing(),
        description="Configurable parameters to construct the Postprocessing.",
    )


@dataclass
class OCDNetEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="The batch size during evaluation.",
        display_name="batch_size",
        popular="yes",
    )
    post_processing: Postprocessing = DATACLASS_FIELD(
        Postprocessing(),
        description="Configurable parameters to construct the Postprocessing.",
    )
    metric: Metric = DATACLASS_FIELD(
        Metric(),
        display_name="metric",
        description="Hyper parameters to configure the metric."
    )


@dataclass
class OCDNetPruneExpConfig:
    """Prune experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="results_dir",
        description="[Optional] Path to a results dir for pruning.",
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="checkpoint",
        description="The path to PyTorch model to prune.",
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="The gpu id.",
        display_name="gpu_id",
    )
    ch_sparsity: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        display_name="ch_sparsity",
        description="The pruning threshold.",
    )
    round_to: int = INT_FIELD(
        value=32,
        default_value=32,
        valid_min=1,
        valid_max="inf",
        description="The round channels to the nearest multiple of round_to",
        display_name="round_to",
    )
    p: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="The norm degree to estimate the importance of channels.",
        display_name="p",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to print prune information.",
        display_name="verbose"
    )


@dataclass
class OCDNetExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="results_dir",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to the checkpoint file to run export.",
        display_name="checkpoint"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="onnx file",
        description="""
        Path to the onnx model file.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="The gpu id.",
        display_name="gpu_id",
    )
    width: int = INT_FIELD(
        value=1280,
        default_value=1280,
        valid_min=1,
        display_name="width",
        description="The width for exporting.",
        popular="yes",
    )
    height: int = INT_FIELD(
        value=736,
        default_value=736,
        valid_min=1,
        display_name="height",
        description="The height for exporting.",
        popular="yes",
    )
    opset_version: int = INT_FIELD(
        value=11,
        default_value=11,
        description="""Operator set version of the ONNX model used to generate
                    the TensorRT engine.""",
        display_name="opset_version",
        valid_min=1,
        popular="yes",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="""Flag to enable verbose exporting logging."""
    )


@dataclass
class OCDNetTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16", "INT8"])
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(
        CalibrationConfig(),
        description="""The configuration elements to define the
                    TensorRT calibrator for int8 PTQ.""",
    )


@dataclass
class OCDNetGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    # @seanf: OCDNet is unique in that it uses width and height upon engine construction
    width: int = INT_FIELD(
        value=1280,
        default_value=1280,
        valid_min=1,
        display_name="width",
        description="The width of input image tensor.",
        popular="yes",
    )
    height: int = INT_FIELD(
        value=736,
        default_value=736,
        valid_min=1,
        display_name="height",
        description="The height for input image tensor.",
        popular="yes",
    )
    img_mode: str = STR_FIELD(
        value="BGR",
        default_value="BGR",
        description="The image mode.",
        display_name="img_mode",
        valid_options="BGR,RGB,GRAY",
    )
    tensorrt: OCDNetTrtConfig = DATACLASS_FIELD(
        OCDNetTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: OCDNetModelConfig = DATACLASS_FIELD(
        OCDNetModelConfig(),
        description="Configurable parameters to construct the model for an OCDNet experiment.",
    )
    dataset: OCDNetDataConfig = DATACLASS_FIELD(
        OCDNetDataConfig(),
        description="Configurable parameters to construct the dataset for an OCDNet experiment.",
    )
    train: OCDNetTrainExpConfig = DATACLASS_FIELD(
        OCDNetTrainExpConfig(),
        description="Configurable parameters to construct the trainer for an OCDNet experiment.",
    )
    evaluate: OCDNetEvalExpConfig = DATACLASS_FIELD(
        OCDNetEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for an OCDNet experiment.",
    )
    inference: OCDNetInferenceExpConfig = DATACLASS_FIELD(
        OCDNetInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for an OCDNet experiment.",
    )
    export: OCDNetExportExpConfig = DATACLASS_FIELD(
        OCDNetExportExpConfig(),
        description="Configurable parameters to construct the exporter for an OCDNet experiment.",
    )
    gen_trt_engine: OCDNetGenTrtEngineExpConfig = DATACLASS_FIELD(
        OCDNetGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for an OCDNet experiment.",
    )
    prune: OCDNetPruneExpConfig = DATACLASS_FIELD(
        OCDNetPruneExpConfig(),
        description="Configurable parameters to construct the pruner for an OCDNet experiment.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters for model quantization.",
    )

    def __post_init__(self):
        """Set default model name for OCDNet."""
        if self.model_name is None:
            self.model_name = "ocdnet"
