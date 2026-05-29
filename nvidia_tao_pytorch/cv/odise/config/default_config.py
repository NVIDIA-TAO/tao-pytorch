# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class InferConfig:
    """Inference configuration template."""

    image_dir: str = MISSING
    checkpoint: str = MISSING
    precision: str = 'fp32'
    label_set: List[str] = field(default_factory=lambda: ["COCO", "ADE", "LVIS"])
    vocab: Optional[str] = None
    caption: Optional[str] = None
    overlap_threshold: float = 0
    object_mask_threshold: float = 0.0
    results_dir: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation configuration template."""

    checkpoint: str = MISSING
    results_dir: Optional[str] = None
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])


@dataclass
class Dataset:
    name: str = "coco_2017_train_panoptic"
    root_dir: str = "/home/scratch.p3/yuw/datasets"
    panoptic_json: str = "coco/annotations/panoptic_train2017.json"
    instance_json: str = "coco/annotations/instance_train2017.json"
    instance_root: str = "/home/scratch.p3/yuw/datasets/coco/train2017"
    panoptic_root: str = "/home/scratch.p3/yuw/datasets/coco/panoptic_train2017"
    semantic_root: str =  "coco/panoptic_semseg_train2017"
    prompt_eng_file: str = ""
    category_json: str = ""


@dataclass
class DataConfig:
    """Data configuration template."""

    input_height: int = 1024
    input_width: int = 1024
    min_scale: float = 0.1
    max_scale: float = 2.0
    total_batch_size: int = 64
    num_workers: int = 4
    train: Dataset = Dataset()
    val: Dataset = Dataset()
    test: Dataset = Dataset()


@dataclass
class ModelConfig:
    """Model configuration template."""

    type: str = "category"
    name: str = "convnext_large_d_320"
    pretrained_weights: str = "laion2b_s29b_b131k_ft_soup"
    alpha: float = 0.4
    beta: float = 0.8
    num_queries: int = 250
    num_classes: int = 133
    pixel_mean: List[float] = field(default_factory=lambda: [122.7709383, 116.7460125, 104.09373615])
    pixel_std: List[float] = field(default_factory=lambda: [68.5005327, 66.6321579, 70.32316305])
    test_topk_per_image: int = 100
    object_mask_threshold: float = 0.0
    overlap_threshold: float = 0.8
    mask_dim: int = 256
    hidden_dim: int = 256
    conv_dim: int = 256
    mask_proj_dim: int = 768
    text_proj_dim: int = -1


@dataclass
class TrainConfig:
    """Train configuration template."""

    use_amp: bool = True
    max_iter: int = 92188
    grad_clip: float = 0.01
    checkpoint_interval: int = 4500
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    checkpoint: Optional[str] = None
    results_dir: Optional[str] = None
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    resume: bool = False
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: Optional[str] = "auto"
    reference_world_size: int = 0
    dataset: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    inference: InferConfig = InferConfig()
    evaluate: EvalConfig = EvalConfig()
    results_dir: str = MISSING
