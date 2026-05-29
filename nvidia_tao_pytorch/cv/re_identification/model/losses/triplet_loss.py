# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triplet Loss for traning."""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalize a Tensor to unit length along the specified dimension.

    Args:
        x (torch.Tensor): The data to normalize.
        axis (int, optional): The axis along which to normalize. Defaults to -1.

    Returns:
        torch.Tensor: The normalized data.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """Compute the euclidean distance between two tensors.

    Args:
        x (torch.Tensor): The first input tensor.
        y (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The euclidean distance between x and y.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels):
    """Perform hard example mining for Triplet loss.

    For each anchor, find the hardest positive and negative samples.

    Args:
        dist_mat (torch.Tensor): The distance matrix.
        labels (torch.Tensor): The labels tensor.

    Returns:
        torch.Tensor: The hardest positive samples distances for each anchor.
        torch.Tensor: The hardest negative samples distances for each anchor.
    """
    assert len(dist_mat.size()) == 2, "The distance matrix generated should have a length of 2."
    assert dist_mat.size(0) == dist_mat.size(1), "The distance matrix generated should be a square matrix."
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, _ = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, _ = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an


class TripletLoss(object):
    """Triplet Loss for training deep embedding models."""

    def __init__(self, margin=None):
        """Initialize TripletLoss module.

        Args:
            margin (float, optional): Margin for the triplet loss. Defaults to None.
        """
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        """Compute the Triplet Loss.

        Args:
            global_feat (torch.Tensor): The feature embeddings.
            labels (torch.Tensor): The corresponding labels.
            normalize_feature (bool, optional): Whether to normalize the features or not. Defaults to False.

        Returns:
            list: The triplet loss value.
            torch.Tensor: The hardest positive samples distances for each anchor.
            torch.Tensor: The hardest negative samples distances for each anchor.
        """
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        """Initialize the CrossEntropyLabelSmooth class.

        Args:
            num_classes (int): Number of classes.
            epsilon (float, optional): Smoothing factor. Defaults to 0.1.
            use_gpu (bool, optional): Whether to use gpu for computation. Defaults to True.
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """Compute the loss based on inputs and targets.

        Args:
            inputs (torch.Tensor): Prediction matrix (before softmax) with shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels with shape (num_classes).

        Returns:
            list: Loss values.
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
