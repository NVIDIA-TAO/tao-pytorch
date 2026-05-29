# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contrastive loss function"""

import torch


class ContrastiveLoss(torch.nn.Module):
    """Contrastive Loss for comparing image embeddings.

    Args:
        margin (float): The margin used for contrastive loss.
    """

    def __init__(self, margin=2.0):
        """Initialize"""
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        """
        Compute the contrastive loss.

        Args:
            euclidean_distance (torch.Tensor): Euclidean distance between the two output tensors from the model
            label (torch.Tensor): Label indicating if the images are similar or dissimilar.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
