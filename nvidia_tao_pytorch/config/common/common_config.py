# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common config fields across all models"""

from dataclasses import dataclass

from omegaconf import MISSING
from typing import Optional, List

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.config.common.mlops import WandBConfig


@dataclass
class CuDNNConfig:
    """Common CuDNN config"""

    benchmark: bool = BOOL_FIELD(value=False)
    deterministic: bool = BOOL_FIELD(value=True)


@dataclass
class TrainConfig:
    """Common train experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the train job.""",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the training on. The length of this list
        must be equal to the number of gpus in train.num_gpus.""",
        popular="yes")
    num_nodes: int = INT_FIELD(
        value=1,
        display_name="Number of nodes",
        description="Number of nodes to run the training on. If > 1, then multi-node is enabled.",
        valid_min=1,
        popular="yes",
    )
    seed: int = INT_FIELD(
        value=1234,
        default_value=1234,
        valid_min=-1,
        valid_max="inf",
        description="The seed for the initializer in PyTorch. If < 0, disable fixed seed.",
        display_name="Seed for randomization",
    )
    cudnn: CuDNNConfig = DATACLASS_FIELD(CuDNNConfig())

    num_epochs: int = INT_FIELD(
        value=10,
        valid_min=1,
        valid_max="inf",
        description="Number of epochs to run the training.",
        display_name="Number of epochs",
        popular="yes",
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Checkpoint interval",
        description="The interval (in epochs) at which a checkpoint will be saved. Helps resume training.",
        popular="yes",
    )
    checkpoint_interval_unit: str = STR_FIELD(
        value="epoch",
        default_value="epoch",
        valid_options="epoch,step",
        display_name="Checkpoint interval unit",
        description="The unit of the checkpoint interval.",
    )
    validation_interval: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Validation interval",
        description="""
        The interval (in epochs) at which a evaluation
        will be triggered on the validation dataset.""",
        popular="yes",
    )

    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        description="Path to the checkpoint to resume training from.",
        display_name="Resume checkpoint path"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)


@dataclass
class EvaluateConfig:
    """Common eval experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the evaluation job.""",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the evaluation on. The length of this list
        must be equal to the number of gpus in evaluate.num_gpus.""",
        popular="yes",
    )
    num_nodes: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of nodes",
        description="Number of nodes to run the evaluation on. If > 1, then multi-node is enabled.",
        popular="yes",
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        description="Path to the checkpoint used for evaluation.",
        display_name="Checkpoint path",
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for evaluation.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT Engine"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor. This is important if batch_size > 1 for large dataset.""",
        display_name="batch size"
    )


@dataclass
class InferenceConfig:
    """Common inference experiment config."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the inference job.""",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the inference on. The length of this list
        must be equal to the number of gpus in inference.num_gpus.""",
        popular="yes",
    )
    num_nodes: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of nodes",
        description="Number of nodes to run the inference on. If > 1, then multi-node is enabled.",
        popular="yes",
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        description="Path to the checkpoint used for inference.",
        display_name="Checkpoint path"
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to the TensorRT engine to be used for inference.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT Engine"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """)
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor. This is important if batch_size > 1 for large dataset.""",
        display_name="batch size"
    )


@dataclass
class ExportConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="""The index of the GPU to build the TensorRT engine.""",
        display_name="GPU ID"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to the checkpoint file to run export.",
        display_name="checkpoint"
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        display_name="onnx file",
        description="""
        Path to the onnx model file.
        """
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="on cpu",
        description="""Flag to export CPU compatible model."""
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of channels in the input Tensor.",
        display_name="input channel",
        valid_min=1,
        valid_options="1,3",
    )
    input_width: int = INT_FIELD(
        value=960,
        default_value=960,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: int = INT_FIELD(
        value=544,
        default_value=544,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=32,
    )
    opset_version: int = INT_FIELD(
        value=17,
        default_value=17,
        description="""Operator set version of the ONNX model used to generate
                    the TensorRT engine.""",
        display_name="opset version",
        valid_min=1,
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor for the engine.
                    A value of :code:`-1` implies dynamic tensor shapes.""",
        display_name="batch size"
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="""Flag to enable verbose TensorRT logging."""
    )
    format: str = STR_FIELD(
        value="onnx",
        display_name="export format",
        description="""File format to export to.""",
        valid_options="onnx,xdl",
    )

# TAO Deploy configs


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = LIST_FIELD(
        arrList=MISSING,
        display_name="Calibration image directories",
        description="""List of image directories to be used for calibration
                    when running Post Training Quantization using TensorRT.""",
    )
    cal_cache_file: str = STR_FIELD(
        value=MISSING,
        display_name="Calibration cache file",
        description="""The path to save the calibration cache file containing
                    scales that were generated during Post Training Quantization.""",
    )
    cal_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The batch size of the input TensorRT to run calibration on.""",
        display_name="Calibration batch size",
        popular="yes",
    )
    cal_batches: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The number of input tensor batches to run calibration on.
                    It is recommended to use atleast 10% of the training images.""",
        display_name="Number of calibration batches",
        popular="yes",
    )


@dataclass
class TrtConfig:
    """Trt config."""

    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=0,
        description="""The size (in MB) of the workspace TensorRT has
                    to run it's optimization tactics and generate the
                    TensorRT engine.""",
        display_name="Max workspace size",
    )
    min_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The minimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Min batch size",
        popular="yes",
    )
    opt_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The optimum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Optimum batch size",
        popular="yes",
    )
    max_batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="""The maximum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Maximum batch size",
        popular="yes",
    )
    layers_precision: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        description="The list to specify layer precision.",
        display_name="layers_precision"
    )


@dataclass
class GenTrtEngineConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        description="""The index of the GPU to build the TensorRT engine.""",
        display_name="GPU ID",
        popular="yes",
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        display_name="ONNX file",
        description="""
        Path to the ONNX model file.
        """
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=MISSING,
        description="""Path to the TensorRT engine generated should be stored.
                    This only works with :code:`tao-deploy`.""",
        display_name="TensorRT engine"
    )
    timing_cache: Optional[str] = STR_FIELD(
        value=None,
        description="""Path to a TensorRT timing cache that speeds up engine generation.
                    This will be created/read/updated.""",
        display_name="TensorRT timing cache"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="""The batch size of the input Tensor for the engine.
                    A value of :code:`-1` implies dynamic tensor shapes.""",
        display_name="Batch size",
        popular="yes",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Verbose",
        description="""Flag to enable verbose TensorRT logging."""
    )


@dataclass
class CommonExperimentConfig:
    """Common experiment config."""

    model_name: Optional[str] = STR_FIELD(
        value=None,
        display_name="Model name",
        description="Name of model if invoking task via :code:`model_agnostic`"
    )
    encryption_key: Optional[str] = STR_FIELD(
        value=None,
        display_name="Encryption key",
        description="Key for encrypting model checkpoints"
    )
    results_dir: Optional[str] = STR_FIELD(
        value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(
            project="TAO Toolkit",
            name="TAO Toolkit training experiment",
            tags=["training", "tao-toolkit"]
        )
    )
