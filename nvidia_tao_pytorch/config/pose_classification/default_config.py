# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""
from typing import Optional, List, Dict
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    TrainConfig,
    EvaluateConfig,
    InferenceConfig,
    CommonExperimentConfig
)


@dataclass
class PCModelConfig:
    """Pose classification model config."""

    model_type: str = STR_FIELD(
        value="ST-GCN",
        default_value="ST-GCN",
        description=(
            "The type of model, which can only be ST-GCN for now. "
            "Newer architectures will be supported in the future."
        ),
        display_name="model type",
        valid_options="ST-GCN"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to the pre-trained model.",
        display_name="pretrained model path"
    )
    input_channels: int = INT_FIELD(
        value=3,
        default_value=3,
        description="The number of input channels (dimension of body poses).",
        display_name="input channels",
        valid_min=1,
        valid_max="inf"
    )
    dropout: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability to drop hidden units.",
        display_name="dropout",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max=1.0
    )
    graph_layout: str = STR_FIELD(
        value="nvidia",
        default_value="nvidia",
        description=(
            "The layout of the graph for modeling skeletons. "
            "It can be nvidia, openpose, human3.6m, ntu-rgb+d, ntu_edge, or coco."
        ),
        display_name="graph layout",
        valid_options="nvidia,openpose,human3.6m,ntu-rgb+d,ntu_edge,coco"
    )
    graph_strategy: str = STR_FIELD(
        value="spatial",
        default_value="spatial",
        description="The strategy of the graph for modeling skeletons. It can be uniform, distance, or spatial.",
        display_name="graph strategy",
        valid_options="uniform,distance,spatial"
    )
    edge_importance_weighting: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Specifies whether to enable edge importance weighting.",
        display_name="edge importance weighting"
    )


@dataclass
class OptimConfig:
    """Optimizer config."""

    optimizer_type: str = STR_FIELD(
        value="torch.optim.SGD",
        default_value="torch.optim.SGD",
        description="The type of the optimizer.",
        display_name="optimizer type",
        valid_options="torch.optim.SGD,torch.optim.Adam,torch.optim.Adamax"
    )
    lr: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="The initial learning rate for the training.",
        display_name="learning rate",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max="inf"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        description="The momentum for the SGD optimizer.",
        display_name="momentum",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max="inf"
    )
    nesterov: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Specifies whether to enable Nesterov momentum.",
        display_name="nesterov"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.0001,
        default_value=0.0001,
        description="The weight decay coefficient.",
        display_name="weight decay",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max="inf"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        default_value="MultiStep",
        description=(
            "The learning scheduler. Two schedulers are provided:\n"
            "* MultiStep : Decrease the lr by lr_decay at setting steps.\n"
            "* AutoReduce : Decrease the lr by lr_decay while lr_monitor doesn't decline "
            "more than 0.1 percent of the previous value."
        ),
        display_name="learning rate scheduler",
        valid_options="AutoReduce,MultiStep"
    )
    lr_monitor: str = STR_FIELD(
        value="val_loss",
        default_value="val_loss",
        description="The monitor value for the AutoReduce scheduler.",
        display_name="learning rate monitor",
        valid_options="val_loss,train_loss"
    )
    patience: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of epochs with no improvement, after which learning rate will be reduced.",
        display_name="patience",
        automl_enabled="TRUE",
        valid_min=1,
        valid_max="inf"
    )
    min_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        description="The minimum learning rate in the training.",
        display_name="minimum learning rate",
        valid_min=0.0,
        valid_max="inf"
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[10, 60],
        description="The steps to decrease the learning rate for the MultiStep scheduler.",
        display_name="learning rate steps",
        value_type="list_2",
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="The decreasing factor for the learning rate scheduler.",
        display_name="learning rate decay",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max="inf"
    )


@dataclass
class SkeletonDatasetConfig:
    """Skeleton dataset config."""

    data_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to the data file.",
        display_name="data path"
    )
    label_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to the label file.",
        display_name="label path"
    )


@dataclass
class PCDatasetConfig:
    """Dataset config."""

    train_dataset: SkeletonDatasetConfig = DATACLASS_FIELD(
        SkeletonDatasetConfig(),
        description=(
            "The data path to the data in a NumPy array and label path to the labels "
            "in a pickle file for training."
        ),
        display_name="train dataset."
    )
    val_dataset: SkeletonDatasetConfig = DATACLASS_FIELD(
        SkeletonDatasetConfig(),
        description=(
            "The data path to the data in a NumPy array and label path to the labels "
            "in a pickle file for validation."
        ),
        display_name="validation dataset."
    )
    num_classes: int = INT_FIELD(
        value=6,
        default_value=6,
        description="The number of action classes.",
        display_name="number of classes",
        valid_min=1,
        valid_max="inf"
    )
    label_map: Optional[Dict[str, int]] = DICT_FIELD(
        hashMap=None,
        description="A dict that maps the class names to indices.",
        display_name="label map"
    )
    random_choose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Specifies whether to randomly choose a portion of the input sequence.",
        display_name="random choose",
    )
    random_move: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Specifies whether to randomly move the input sequence.",
        display_name="random move",
    )
    window_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="The length of the output sequence. A value of -1 specifies the original length.",
        display_name="window size",
        automl_enabled="TRUE",
        valid_min=-1,
        valid_max="inf"
    )
    batch_size: int = INT_FIELD(
        value=16,
        default_value=16,
        description="The batch size for training and validation.",
        display_name="batch size",
        valid_min=1,
        valid_max="inf"
    )
    num_workers: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of parallel workers processing data.",
        display_name="number of workers",
        valid_min=1,
        valid_max="inf"
    )


@dataclass
class PCTrainExpConfig(TrainConfig):
    """Train experiment config."""

    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description=(
            "The configuration for the SGD optimizer, including the learning rate, "
            "learning scheduler, weight decay, etc."
        ),
        display_name="optimization configuration"
    )
    grad_clip: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="The amount to clip the gradient by the L2 norm. A value of 0.0 specifies no clipping.",
        display_name="gradient clip",
        automl_enabled="TRUE",
        valid_min=0.0,
        valid_max="inf"
    )


@dataclass
class PCInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    output_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to the output text file.",
        display_name="output file"
    )
    test_dataset: SkeletonDatasetConfig = DATACLASS_FIELD(
        SkeletonDatasetConfig(),
        description=(
            "The data path to the data in a NumPy array and label path to the labels "
            "in a pickle file for testing."
        ),
        display_name="train dataset."
    )


@dataclass
class PCEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    test_dataset: SkeletonDatasetConfig = DATACLASS_FIELD(
        SkeletonDatasetConfig(),
        description=(
            "The data path to the data in a NumPy array and label path to the labels "
            "in a pickle file for testing."
        ),
        display_name="train dataset."
    )


@dataclass
class PCExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to a folder where the experiment outputs should be written.",
        display_name="results directory"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The .tlt model.",
        display_name="checkpoint"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description=(
            "The path to save the exported model to. "
            "The default path is in the same directory as the .tlt model."
        ),
        display_name="ONNX file",
        required="yes"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="""The GPU index used to run the evaluation. You can specify the GPU index used to run evaluation
                    when the machine has multiple GPUs installed. Note that evaluation can only run on a single GPU.""",
        display_name="GPU ID",
        valid_min=0,
        valid_max="inf"
    )


@dataclass
class PCDatasetConvertExpConfig:
    """Dataset conversion experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The path to a folder where the experiment outputs should be written.",
        display_name="results directory"
    )
    data: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The output JSON data from the deepstream-bodypose-3d app.",
        display_name="data path"
    )
    pose_type: str = STR_FIELD(
        value="3dbp",
        default_value="3dbp",
        description="The pose type can be chosen from 3dbp, 25dbp, 2dbp.",
        display_name="data path",
        valid_options="3dbp,25dbp,2dbp"
    )
    num_joints: int = INT_FIELD(
        value=34,
        default_value=34,
        description="The number of joint points in the graph layout.",
        display_name="number of joints",
        valid_min=1,
        valid_max="inf"
    )
    input_width: int = INT_FIELD(
        value=1920,
        default_value=1920,
        description="The width of input images in pixels for normalization.",
        display_name="input width",
        valid_min=1,
        valid_max="inf"
    )
    input_height: int = INT_FIELD(
        value=1080,
        default_value=1080,
        description="The height of input images in pixels for normalization.",
        display_name="input height",
        valid_min=1,
        valid_max="inf"
    )
    focal_length: float = FLOAT_FIELD(
        value=1200.0,
        default_value=1200.0,
        description="The focal length of the camera for normalization.",
        display_name="focal length",
        valid_min=0.0,
        valid_max="inf"
    )
    sequence_length_max: int = INT_FIELD(
        value=300,
        default_value=300,
        description="The maximum sequence length for defining array shape.",
        display_name="maximum sequence length",
        valid_min=1,
        valid_max="inf"
    )
    sequence_length_min: int = INT_FIELD(
        value=10,
        default_value=10,
        description="The minimum sequence length for filtering short sequences.",
        display_name="minimum sequence length",
        valid_min=1,
        valid_max="inf"
    )
    sequence_length: int = INT_FIELD(
        value=100,
        default_value=100,
        description="The general sequence length for sampling.",
        display_name="sequence length",
        valid_min=1,
        valid_max="inf"
    )
    sequence_overlap: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The overlap between sequences for sampling.",
        display_name="sequence overlap",
        valid_min=0.0,
        valid_max=1.0
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: PCModelConfig = DATACLASS_FIELD(
        PCModelConfig(),
        description="The configuration for modeling.",
        display_name="model configuration"
    )
    dataset: PCDatasetConfig = DATACLASS_FIELD(
        PCDatasetConfig(),
        description="The configuration for dataset.",
        display_name="dataset configuration"
    )
    train: PCTrainExpConfig = DATACLASS_FIELD(
        PCTrainExpConfig(),
        description="The configuration for training.",
        display_name="training configuration"
    )
    inference: PCInferenceExpConfig = DATACLASS_FIELD(
        PCInferenceExpConfig(),
        description="The configuration for inference.",
        display_name="inference configuration"
    )
    evaluate: PCEvalExpConfig = DATACLASS_FIELD(
        PCEvalExpConfig(),
        description="The configuration for evaluation.",
        display_name="evaluation configuration"
    )
    export: PCExportExpConfig = DATACLASS_FIELD(
        PCExportExpConfig(),
        description="The configuration for exporting.",
        display_name="exporting configuration"
    )
    dataset_convert: PCDatasetConvertExpConfig = DATACLASS_FIELD(
        PCDatasetConvertExpConfig(),
        description="The configuration for dataset conversion.",
        display_name="dataset conversion configuration"
    )

    def __post_init__(self):
        """Set default model name for Pose Classification."""
        if self.model_name is None:
            self.model_name = "pose_classification"
