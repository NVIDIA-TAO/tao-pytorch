# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this code are based on the VGGT project by Facebook Research (Meta):
# https://github.com/facebookresearch/vggt

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py

"""Drop path module for the VGGT model."""

from torch import nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop path

    Args:
        x (Tensor): Input features of shape (B, N, C).
        drop_prob (float): Dropout probability. Default is 0.0.
        training (bool): Whether to enable training. Default is False.

    Returns:
        Tensor: Output features of shape (B, N, C).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        drop_prob (float): Dropout probability. Default is None.

    Returns:
        Tensor: Output features of shape (B, N, C).
    """

    def __init__(self, drop_prob=None):
        """Drop path constructor"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward pass"""
        return drop_path(x, self.drop_prob, self.training)
