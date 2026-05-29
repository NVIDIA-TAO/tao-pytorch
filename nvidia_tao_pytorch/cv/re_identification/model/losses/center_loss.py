# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Center Loss for traning."""
from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss class for deep learning models.

    This class implements Center Loss, a discriminative feature learning approach,
    which is beneficial for tasks like face recognition. It computes the loss between
    the deep features and their corresponding class centers.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Attributes:
        num_classes (Tensor): The number of classes in the dataset.
        feat_dim (int): The dimension of the feature vector.
        use_gpu (bool): If True, CUDA will be used for computation.
        centers (nn.Parameter): Parameterized center vectors for each class.

    Methods:
        forward(x, labels): Computes the loss between feature vectors and their corresponding class centers.
    """

    def __init__(self, num_classes, feat_dim=2048, use_gpu=True):
        """Initializes the CenterLoss module.

        Args:
            num_classes (Tensor): The number of classes in the dataset.
            feat_dim (int, optional): The dimension of the feature vector. Default is 2048.
            use_gpu (bool, optional): If True, CUDA will be used for computation. Default is True.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """Computes the loss by passing the feature vectors and labels.

        This method calculates the distance between the deep features and their
        corresponding class centers. The loss is the mean of these distances.

        Args:
            x (Tensor): The deep feature vectors of shape (batch_size, feat_dim).
            labels (Tensor): The corresponding labels of the deep features of shape (batch_size,).

        Returns:
            loss (Tensor): A scalar tensor representing the mean loss.
        """
        assert x.size(0) == labels.size(0), "Features.size(0) is not equal to Labels.size(0)."

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
