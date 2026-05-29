# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core config for TAO distillation."""

from typing import List
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class DistillationBindingConfig:
    """Distillation binding configuration."""

    student_module_name: str = MISSING
    teacher_module_name: str = MISSING
    criterion: str = MISSING
    weight: float = 1.0


@dataclass
class DistillationConfig:
    """Distillation configuration."""

    teacher: dataclass = MISSING
    pretrained_teacher_model_path: str = MISSING
    bindings: List[DistillationBindingConfig] = field(default_factory=list)
