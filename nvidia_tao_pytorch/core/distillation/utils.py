# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for distilling TAO Toolkit models."""

from typing import Dict, Callable, Optional
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.distillation.losses import Criterion


class Binding(nn.Module):
    """Binding class to bind and capture student, teacher layers together."""

    def __init__(self, student: str,
                 teacher: str,
                 criterion: Criterion,
                 visualizer: Optional[Callable] = None,
                 loss_coef: float = 1.0,
                 ):
        """Initializes the Binding.

        Args:
            student (str): Student module name.
            teacher (str): Teacher module name.
            criterion (Criterion): Loss function to use for distillation.
            visualizer (Optional[Callable], optional): Visualizer function to visualize feature maps. Defaults to None.
            loss_coef (float, optional): Loss coefficient. Defaults to 1.0.
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.visualizer = visualizer
        self.loss_coef = loss_coef


class CaptureModule:
    """CaptureModule class to capture output of a module."""

    def __init__(self, module: nn.Module):
        """Initializes the CaptureModule.

        Args:
            module (nn.Module): Module to capture output.
        """
        self.module = module
        self.output = None
        self.handle = None

    def hook(self, module, inputs, output):
        """Hook function to capture output.

        Args:
            module (nn.Module): Module.
            inputs (Any): Inputs to the module.
            output (Any): Output of the module.

        Returns:
            Any: Output of the module.
        """
        self.output = output
        return output

    def attach_hook(self):
        """Attaches hook to the module."""
        self.remove_hook()
        self.module.register_forward_hook(self.hook)

    def remove_hook(self):
        """Removes hook from the module."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self, *args, **kwargs):
        """Enters the context manager."""
        self.attach_hook()

    def __exit__(self, *args, **kwargs):
        """Exits the context manager."""
        self.remove_hook()


def visualize_feature_maps(path: str,
                           image: torch.Tensor,
                           student_maps: Dict[str, torch.Tensor],
                           teacher_maps: Dict[str, torch.Tensor],
                           batch_index: int = 0,
                           channel_idx: int = 0
                           ):
    """Visualize feature maps.

    Args:
        path (str): Path to save the visualization.
        image (torch.Tensor): Input image.
        student_maps (Dict[str, torch.Tensor]): Student feature maps.
        teacher_maps (Dict[str, torch.Tensor]): Teacher feature maps.
        batch_index (int, optional): Batch index. Defaults to 0.
        channel_idx (int, optional): Channel index. Defaults to 0.

    Returns:
        None
    """
    keys = sorted(student_maps.keys())

    student_maps = [student_maps[k] for k in keys]
    teacher_maps = [teacher_maps[k] for k in keys]

    image = image[batch_index, 0].detach().cpu().numpy()

    num_cols = len(student_maps) + 1

    plt.figure(figsize=(20, 20))
    plt.subplot(2, num_cols, 1)
    plt.imshow(image)
    plt.subplot(2, num_cols, 1 + num_cols)
    plt.imshow(image)

    for i in range(num_cols - 1):
        plt.subplot(2, num_cols, 1 + i + 1)
        plt.imshow(teacher_maps[i][batch_index, channel_idx].detach().cpu().numpy())
        plt.subplot(2, num_cols, 1 + num_cols + i + 1)
        plt.imshow(student_maps[i][batch_index, channel_idx].detach().cpu().numpy())

    plt.savefig(path)
    plt.close()
