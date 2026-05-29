# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""
from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig
)


@dataclass
class ReIDModelConfig:
    """Re-Identification model configuration for training, testing & validation."""

    backbone: str = STR_FIELD(
        value="resnet_50",
        description="Backbone type.",
        display_name="Backbone Type",
        popular="yes"
    )
    last_stride: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Stride size of the last layer of the backbone.",
        display_name="Last Stride"
    )
    pretrain_choice: str = STR_FIELD(
        value="imagenet",
        description="Source of pretraining.",
        display_name="Pretrain Choice"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        description="Path to the pretrained model file.",
        display_name="Pretrained Model Path"
    )
    input_channels: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=3,
        valid_max=3,
        description="Number of input channels.",
        display_name="Input Channels"
    )
    input_width: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=128,
        valid_max=128,
        description="Width of the input image.",
        display_name="Input Width"
    )
    input_height: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=256,
        valid_max=256,
        description="Height of the input image.",
        display_name="Input Height"
    )
    neck: str = STR_FIELD(
        value="bnneck",
        description="Type of neck used in the model architecture.",
        display_name="Neck Type"
    )
    feat_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=32,
        valid_max=768,
        description="Dimension of the feature vector.",
        display_name="Feature Dimension"
    )
    neck_feat: str = STR_FIELD(
        value="after",
        description="Position of the feature extraction in the neck.",
        display_name="Neck Feature Position"
    )
    metric_loss_type: str = STR_FIELD(
        value="triplet",
        description="Type of metric loss used.",
        display_name="Metric Loss Type"
    )
    with_center_loss: bool = BOOL_FIELD(
        value=False,
        description="Whether center loss is used.",
        display_name="Center Loss"
    )
    with_flip_feature: bool = BOOL_FIELD(
        value=False,
        description="Whether flip feature is enabled.",
        display_name="Flip Feature"
    )
    label_smooth: bool = BOOL_FIELD(
        value=True,
        description="Whether label smoothing is applied.",
        display_name="Label Smooth"
    )
    pretrain_hw_ratio: float = FLOAT_FIELD(
        value=2,
        default_value=2,
        valid_min=2,
        valid_max=2,
        description="Height-width ratio of the pretraining model.",
        display_name="Pretrain HW Ratio"
    )
    id_loss_type: str = STR_FIELD(
        value="softmax",
        description="Type of ID loss used.",
        display_name="ID Loss Type"
    )
    id_loss_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0,
        valid_max=1.0,
        description="Weight of the ID loss.",
        display_name="ID Loss Weight"
    )
    triplet_loss_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0,
        valid_max=1.0,
        description="Weight of the triplet loss.",
        display_name="Triplet Loss Weight"
    )
    no_margin: bool = BOOL_FIELD(
        value=False,
        description="Whether margin is used in loss computation.",
        display_name="No Margin"
    )
    cos_layer: bool = BOOL_FIELD(
        value=False,
        description="Whether cosine layer is used for the output.",
        display_name="Cosine Layer"
    )
    dropout_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Dropout rate applied in the model.",
        display_name="Dropout Rate"
    )
    reduce_feat_dim: bool = BOOL_FIELD(
        value=False,
        description="Whether feature dimension reduction is applied.",
        display_name="Reduce Feature Dimension"
    )
    drop_path: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        description="Drop path probability.",
        display_name="Drop Path"
    )
    drop_out: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Dropout probability.",
        display_name="Drop Out"
    )
    att_drop_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Attention dropout rate.",
        display_name="Attention Drop Rate"
    )
    stride_size: List[int] = LIST_FIELD(
        arrList=[16, 16],
        description="Size of stride in the convolution layers.",
        display_name="Stride Size"
    )
    gem_pooling: bool = BOOL_FIELD(
        value=False,
        description="Whether generalized mean pooling is used.",
        display_name="GEM Pooling"
    )
    stem_conv: bool = BOOL_FIELD(
        value=False,
        description="Whether a convolutional stem is used at the model input.",
        display_name="Stem Convolution"
    )
    jpm: bool = BOOL_FIELD(
        value=False,
        description="Whether Joint Part and Global feature learning module is enabled.",
        display_name="JPM"
    )
    shift_num: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=5,
        valid_max=5,
        description="Number of positions to shift in shift layer.",
        display_name="Shift Number"
    )
    shuffle_group: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=2,
        valid_max=2,
        description="Number of groups for channel shuffling.",
        display_name="Shuffle Group"
    )
    devide_length: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=4,
        valid_max=4,
        description="Length for division in the re-arrangement process.",
        display_name="Divide Length"
    )
    re_arrange: bool = BOOL_FIELD(
        value=True,
        description="Whether to re-arrange elements in some pattern.",
        display_name="Re-arrange"
    )
    sie_coe: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        valid_min=0.0,
        valid_max="inf",
        description="Coefficient for scaling in SIE module.",
        display_name="SIE Coefficient"
    )
    sie_camera: bool = BOOL_FIELD(
        value=False,
        description="Whether camera-based Spatial Information Enhancement is used.",
        display_name="SIE Camera"
    )
    sie_view: bool = BOOL_FIELD(
        value=False,
        description="Whether view-based Spatial Information Enhancement is used.",
        display_name="SIE View"
    )
    semantic_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Weight for the semantic component in loss calculation.",
        display_name="Semantic Weight"
    )


@dataclass
class OptimConfig:
    """Optimizer configuration for the LR scheduler."""

    name: str = STR_FIELD(
        value="Adam",
        description="Name of the optimizer.",
        display_name="Optimizer Name"
    )
    lr_monitor: str = STR_FIELD(
        value="val_loss",
        description="Metric to monitor for learning rate adjustments.",
        display_name="LR Monitor Metric"
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[40, 70],
        description="Epochs at which the learning rate will decrease.",
        display_name="LR Decay Steps",
        value_type="list_2",
        automl_enabled="TRUE"
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        description="Factor by which the learning rate will decrease.",
        display_name="LR Decay Factor"
    )
    bias_lr_factor: float = FLOAT_FIELD(
        value=1,
        default_value=1,
        valid_min=0,
        valid_max=1,
        description="Learning rate factor for bias parameters.",
        display_name="Bias LR Factor"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.0005,
        default_value=0.0005,
        valid_min=0,
        valid_max=1,
        description="Weight decay for regularization.",
        display_name="Weight Decay"
    )
    weight_decay_bias: float = FLOAT_FIELD(
        value=0.0005,
        default_value=0.0005,
        valid_min=0,
        valid_max=1,
        description="Weight decay for bias regularization.",
        display_name="Weight Decay for Bias"
    )
    warmup_factor: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        valid_min=0.0,
        valid_max=1.0,
        description="Initial learning rate as a factor of the base learning rate during warm-up.",
        display_name="Warmup Factor"
    )
    warmup_iters: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=0,
        valid_max="inf",
        description="Number of iterations for warm-up.",
        display_name="Warmup Iterations"
    )
    warmup_epochs: int = INT_FIELD(
        value=20,
        default_value=20,
        valid_min=0,
        valid_max="inf",
        description="Number of epochs for warm-up.",
        display_name="Warmup Epochs"
    )
    warmup_method: str = STR_FIELD(
        value='linear',
        description="Method used for warm-up (e.g., 'linear', 'exp').",
        display_name="Warmup Method"
    )
    base_lr: float = FLOAT_FIELD(
        value=0.00035,
        default_value=0.00035,
        valid_min=0,
        valid_max=1,
        description="Base learning rate.",
        display_name="Base Learning Rate"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        description="Momentum factor for optimization.",
        display_name="Momentum"
    )
    center_loss_weight: float = FLOAT_FIELD(
        value=0.0005,
        default_value=0.0005,
        valid_min=0,
        valid_max=1,
        description="Weight of the center loss in the loss function.",
        display_name="Center Loss Weight"
    )
    center_lr: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1,
        description="Learning rate for center loss parameters.",
        display_name="Center Learning Rate"
    )
    triplet_loss_margin: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=1,
        description="Margin for triplet loss.",
        display_name="Triplet Loss Margin"
    )
    large_fc_lr: bool = BOOL_FIELD(
        value=False,
        description="Use a larger learning rate for the fully connected layer.",
        display_name="Large FC Learning Rate"
    )
    cosine_margin: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Margin for cosine similarity in losses.",
        display_name="Cosine Margin"
    )
    cosine_scale: float = FLOAT_FIELD(
        value=30,
        default_value=30,
        valid_min=1,
        valid_max="inf",
        description="Scaling factor for cosine similarity.",
        display_name="Cosine Scale"
    )
    trp_l2: bool = BOOL_FIELD(
        value=False,
        description="Apply L2 normalization in triplet loss calculation.",
        display_name="Triplet L2 Normalization"
    )


@dataclass
class ReIDDatasetConfig:
    """Re-Identification Dataset configuration template."""

    train_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the training dataset.",
        display_name="Training Dataset Directory"
    )
    test_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the testing dataset.",
        display_name="Testing Dataset Directory"
    )
    query_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the query dataset.",
        display_name="Query Dataset Directory"
    )
    num_classes: int = INT_FIELD(
        value=751,
        default_value=751,
        valid_min=1,
        valid_max="inf",
        description="Number of classes.",
        display_name="Number of Classes",
        popular="yes"
    )
    batch_size: int = INT_FIELD(
        value=64,
        default_value=64,
        valid_min=1,
        valid_max="inf",
        description="Batch size.",
        display_name="Batch Size",
        popular="yes"
    )
    val_batch_size: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=1,
        valid_max="inf",
        description="Validation Batch size.",
        display_name="Validation Batch Size",
        popular="yes"
    )
    num_workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of workers.",
        display_name="Workers",
        popular="yes"
    )
    pixel_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="Mean values for normalization.",
        display_name="Pixel Mean"
    )
    pixel_std: List[float] = LIST_FIELD(
        arrList=[0.226, 0.226, 0.226],
        description="Standard deviation values for normalization.",
        display_name="Pixel Standard Deviation"
    )
    padding: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=0,
        valid_max=10,
        description="Padding size.",
        display_name="Padding"
    )
    prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability for certain augmentations.",
        display_name="Probability"
    )
    re_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability for re-augmentation.",
        display_name="Re-augmentation Probability"
    )
    sampler: str = STR_FIELD(
        value="softmax_triplet",
        description="Type of sampler used for selecting instances.",
        display_name="Sampler Type"
    )
    num_instances: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=4,
        valid_max="inf",
        description="Number of instances per class in a batch.",
        display_name="Number of Instances",
        popular="yes"
    )


@dataclass
class ReIDReRankingConfig:
    """Re-Ranking configuration template for evaluation."""

    re_ranking: bool = BOOL_FIELD(
        value=False,
        description="Enable or disable re-ranking.",
        display_name="Re-Ranking"
    )
    k1: int = INT_FIELD(
        value=20,
        default_value=20,
        valid_min=20,
        valid_max=20,
        description="The number of top-k candidates in the first round of re-ranking.",
        display_name="K1",
        popular="yes"
    )
    k2: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=6,
        valid_max=6,
        description="The number of top-k candidates in the second round of re-ranking.",
        display_name="K2",
        popular="yes"
    )
    lambda_value: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=0.3,
        description="The lambda value for balancing the original and Jaccard distance in re-ranking.",
        display_name="Lambda Value"
    )
    max_rank: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=10,
        valid_max=10,
        description="The maximum rank considered in re-ranking.",
        display_name="Max Rank",
        popular="yes"
    )
    num_query: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=10,
        valid_max=10,
        description="The number of query images used in re-ranking.",
        display_name="Number of Queries",
        popular="yes"
    )


@dataclass
class ReIDTrainExpConfig(TrainConfig):
    """Train experiment configuration template."""

    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description="Training optimization config.",
        display_name="Optimization config",
    )

    grad_clip: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Maximum norm of the gradients for clipping.",
        display_name="Gradient Clipping"
    )


@dataclass
class ReIDInferenceExpConfig(InferenceConfig):
    """Inference experiment configuration template."""

    output_file: Optional[str] = STR_FIELD(
        value=None,
        description="File path for output json results.",
        display_name="Output JSON File Path"
    )
    test_dataset: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the testing dataset.",
        display_name="Test Dataset Directory"
    )
    query_dataset: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the query dataset.",
        display_name="Query Dataset Directory"
    )


@dataclass
class ReIDEvalExpConfig(EvaluateConfig):
    """Evaluation experiment configuration template."""

    output_sampled_matches_plot: Optional[str] = STR_FIELD(
        value=None,
        description="File path for the output plot of sampled matches.",
        display_name="Output Sampled Matches Plot"
    )
    output_cmc_curve_plot: Optional[str] = STR_FIELD(
        value=None,
        description="File path for the output plot of the CMC curve.",
        display_name="Output CMC Curve Plot"
    )
    test_dataset: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the testing dataset.",
        display_name="Test Dataset Directory"
    )
    query_dataset: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for the query dataset.",
        display_name="Query Dataset Directory"
    )


@dataclass
class ReIDExportExpConfig:
    """Export experiment configuraiton template."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        description="Directory for storing results.",
        display_name="Results Directory"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        description="Path to the checkpoint file.",
        display_name="Checkpoint File"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        description="Path to the ONNX model file.",
        display_name="ONNX File"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="GPU ID for computation.",
        display_name="GPU ID"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: ReIDModelConfig = DATACLASS_FIELD(
        ReIDModelConfig(),
        description=(
            "Configurable parameters to construct the model for a "
            "Re-Identification experiment."
        ),
    )
    dataset: ReIDDatasetConfig = DATACLASS_FIELD(
        ReIDDatasetConfig(),
        description=(
            "Configurable parameters to construct the dataset for a "
            "Re-Identification experiment."
        ),
    )
    train: ReIDTrainExpConfig = DATACLASS_FIELD(
        ReIDTrainExpConfig(),
        description=(
            "Configurable parameters to construct the trainer for a "
            "Re-Identification experiment."
        ),
    )
    evaluate: ReIDEvalExpConfig = DATACLASS_FIELD(
        ReIDEvalExpConfig(),
        description=(
            "Configurable parameters to construct the evaluator for a "
            "Re-Identification experiment."
        ),
    )
    inference: ReIDInferenceExpConfig = DATACLASS_FIELD(
        ReIDInferenceExpConfig(),
        description=(
            "Configurable parameters to construct the inferencer for a "
            "Re-Identification experiment."
        ),
    )
    export: ReIDExportExpConfig = DATACLASS_FIELD(
        ReIDExportExpConfig(),
        description=(
            "Configurable parameters to construct the exporter for a "
            "Re-Identification experiment."
        ),
    )
    re_ranking: ReIDReRankingConfig = DATACLASS_FIELD(
        ReIDReRankingConfig(),
        description=(
            "Configurable parameters to construct the re-ranking parameters for a "
            "Re-Identification experiment."
        ),
    )

    def __post_init__(self):
        """Set default model name for Re-Identification."""
        if self.model_name is None:
            self.model_name = "re_identification"
