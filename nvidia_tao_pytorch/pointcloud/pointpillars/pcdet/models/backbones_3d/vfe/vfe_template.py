# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VFE Template module."""
import torch.nn as nn


class VFETemplate(nn.Module):
    """VFETemplate class."""

    def __init__(self, model_cfg, **kwargs):
        """Initialize."""
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        """Get output feature dimension."""
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError
