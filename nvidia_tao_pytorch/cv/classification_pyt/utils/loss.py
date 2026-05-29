# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss Functions for Classification"""

import torch.nn as nn
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import NOCLASS_IDX


class Cross_Entropy(nn.Module):
    """
    Cross Entropy Loss with label smoothing

    Args:
        weight (Tensor): A manual rescaling weight given to each class.
        label_smoothing (float): The label smoothing value.
        soft (bool): If True, allow soft label from a teacher model.
    """

    def __init__(self, weight=None, label_smoothing=0.1, soft=False):
        super(Cross_Entropy, self).__init__()
        self.soft = soft
        if soft:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction="mean",
                ignore_index=NOCLASS_IDX,
            )

    def forward(self, pred, target):
        """
        Forward pass for Cross_Entropy
        """
        return self.loss(pred, target)
