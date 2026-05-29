# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVDINOv2 Loss"""

from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class KoLeoLoss(nn.Module):
    """
    Spreading vectors for similarity search from Sablayrolles et al.
    https://arxiv.org/abs/1806.03198
    """

    def __init__(self):
        """Initializes the KoLeoLoss instance."""
        super().__init__()

        self.pairwise_distance = nn.PairwiseDistance(p=2, eps=1e-8)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x, eps=1e-8):
        """Computes the KoLeo loss for the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: The computed loss value.
        """
        x = F.normalize(x, p=2, dim=-1, eps=eps)

        # find min distance assignments
        dot_product = torch.mm(x, x.t())
        dot_product.fill_diagonal_(-1)
        index = torch.argmax(dot_product, dim=1)

        distances = self.pairwise_distance(x, x[index])  # (batch_size, )
        loss = -torch.log(distances + eps).mean()

        return loss


class DinoV2Loss(nn.Module):
    """DINOv2 Loss """

    def __init__(
        self,
        num_prototypes: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        centering_method: Literal["sinkhorn", "softmax"] = "softmax",
        n_iters: int = 3,
    ):
        """Initializes the DinoV2Loss instance.

        Args:
            num_prototypes (int): The number of prototypes.
            student_temp (float): Temperature for the student model outputs. Defaults to 0.1.
            center_momentum (float): Momentum for updating the center. Defaults to 0.9.
            centering_method (Literal["sinkhorn", "softmax"]): Method for centering the teacher outputs. Defaults to "softmax".
            n_iters (int): Number of iterations for Sinkhorn-Knopp centering. Defaults to 3.
        """
        super().__init__()

        self.num_prototypes = num_prototypes
        self.centering_method = centering_method
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        if centering_method == "softmax":
            self.register_buffer("center", torch.zeros(1, num_prototypes))
            self.updated = True
            self.reduce_handle = None
            self.len_teacher_output = None
            self.async_batch_center = None
        elif centering_method == "sinkhorn":
            self.n_iters = n_iters
        else:
            raise NotImplementedError("Only sinkhorn and softmax are supported")

    @staticmethod
    def _reduce(x: torch.Tensor):
        """Reduces the tensor across all processes in the distributed group.

        Args:
            x (torch.Tensor): The input tensor to be reduced.

        Returns:
            torch.Tensor: The reduced tensor.
        """
        if dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

        return x

    @staticmethod
    def _world_size():
        """Gets the number of processes in the current distributed group.

        Returns:
            int: The number of processes in the group.
        """
        return dist.get_world_size() if dist.is_initialized() else 1

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        """Applies softmax centering to the teacher output.

        Args:
            teacher_output (torch.Tensor): The teacher output.
            teacher_temp (float): The temperature value for softmax scaling.

        Returns:
            torch.Tensor: The softmax-centered teacher output.
        """
        self.center = self.center.to(teacher_output.device)  # Move center to the same device as the teacher output

        if self.updated is False:
            # Wait for previous all_reduce to finish
            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            # Get the mean of the teacher output, which is the center
            t = self.async_batch_center / self._world_size()
            self.center = self.center * self.center_momentum + t * (
                1 - self.center_momentum
            )
            self.updated = True

        x = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

        self.updated = False
        self.len_teacher_output = teacher_output.shape[0]
        self.async_batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # All reduce to get the sum of all teacher outputs
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

        return x

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output, teacher_temp, num_masked_patches=None
    ):
        """Applies Sinkhorn-Knopp centering to the teacher output.
        This function is not tested yet

        Args:
            teacher_output (torch.Tensor): The teacher output.
            teacher_temp (float): The temperature value for scaling.
            num_masked_patches (torch.Tensor, optional): The number of masked patches. Defaults to None.

        Returns:
            torch.Tensor: The Sinkhorn-centered teacher output.
        """
        Q = torch.exp(teacher_output / teacher_temp).T  # (prototypes, batch_size)
        B = Q.shape[1] * self._world_size()

        if num_masked_patches is not None:
            assert isinstance(num_masked_patches, torch.Tensor)
            B = self._reduce(num_masked_patches)

        K = Q.shape[0]  # total samples and prototypes

        # make the matrix sums to 1
        sum_Q = self._reduce(torch.sum(Q))
        Q /= sum_Q

        for _ in range(self.n_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_feat = self._reduce(torch.sum(Q, dim=1, keepdim=True))
            Q /= sum_feat
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        return (Q * B).T

    @torch.no_grad()
    def centering(self, teacher_output, teacher_temp, num_masked_patches=None):
        """Centers the teacher output using the specified centering method.

        Args:
            teacher_output (torch.Tensor): The teacher output.
            teacher_temp (float): The temperature value for scaling.
            num_masked_patches (torch.Tensor, optional): The number of masked patches. Defaults to None.

        Returns:
            torch.Tensor: The centered teacher output.
        """
        assert (
            teacher_output.shape[1] == self.num_prototypes
        ), f"The last dim of teacher_output should be {self.num_prototypes}"

        if self.centering_method == "softmax":
            return self.softmax_center_teacher(teacher_output, teacher_temp)
        elif self.centering_method == "sinkhorn":
            return self.sinkhorn_knopp_teacher(
                teacher_output, teacher_temp, num_masked_patches
            )
        else:
            raise NotImplementedError("Only sinkhorn and softmax are supported")

    def forward(self, student_output_list, teacher_assignments_list):
        """Computes the DINOv2 loss based on student and teacher outputs.

        Args:
            student_output_list (list): List of student outputs.
            teacher_assignments_list (list): List of teacher assignments.

        Returns:
            torch.Tensor: The total computed loss.
        """
        total_loss = 0.0
        for student_output in student_output_list:
            log_softmax = F.log_softmax(student_output / self.student_temp, dim=-1)

            for t in teacher_assignments_list:
                loss = torch.sum(-t * log_softmax, dim=-1)
                total_loss += loss.mean()

        return total_loss

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """ Computes the DINOv2 loss specifically for masked tokens.

        Args:
            student_patch_tokens_masked (torch.Tensor): Masked student outputs.
            teacher_patch_tokens_masked (torch.Tensor): Masked teacher outputs.
            student_output (torch.Tensor): Unmasked student outputs.

        Returns:
            torch.Tensor: The total computed loss for masked tokens.
        """
        loss = torch.sum(
            teacher_patch_tokens_masked *
            F.log_softmax(student_patch_tokens_masked / self.student_temp, dim=-1),
            dim=-1,
        )

        if masks_weight is None:
            mask_weight = 1 / student_masks_flat.sum(dim=-1).clamp(min=1)
            masks_weight = mask_weight.unsqueeze(-1).expand_as(student_masks_flat)[
                student_masks_flat
            ]

        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]

        loss = loss * masks_weight

        return -loss.sum() / student_masks_flat.shape[0]


class CLIPLoss(nn.Module):
    """CLIP Loss"""

    def __init__(self):  # pylint: disable=useless-parent-delegation
        """Initializes the CLIPLoss instance."""
        super().__init__()

    def forward(self, teacher_features, student_features):
        """Computes the CLIP Loss between teacher and student features.

        Args:
            teacher_features (torch.Tensor): The features from the teacher model.
            student_features (torch.Tensor): The features from the teacher model.

        Returns:
            torch.Tensor: The computed CLIP loss value.
        """
        teacher_features = teacher_features.reshape(-1, teacher_features.shape[-1])
        student_features = student_features.reshape(-1, student_features.shape[-1])

        teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        student_features = F.normalize(student_features, p=2, dim=-1)

        logits_per_teacher = teacher_features @ student_features.T
        logits_per_student = student_features @ teacher_features.T
        labels = torch.arange(teacher_features.shape[0], device=teacher_features.device, dtype=torch.long)

        loss = (F.cross_entropy(logits_per_teacher, labels) + F.cross_entropy(logits_per_student, labels)) / 2

        return loss
