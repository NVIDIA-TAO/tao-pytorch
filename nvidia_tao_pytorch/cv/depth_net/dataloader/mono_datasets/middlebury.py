# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Class for Middlebury data."""

from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_gt_middlebury
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets.base_relative_mono import BaseRelativeMonoDataset


class Middlebury(BaseRelativeMonoDataset):
    """Dataset class for Middlebury, providing ground truth in disparity format."""

    def read_gt_depth(self, disp_path):
        """Read Middlebury ground truth disparity and mask data.

        Args:
            disp_path (str): path to the disparity map.

        Returns:
            depth (np.ndarray): depth map.
        """
        return read_gt_middlebury(disp_path, normalize_depth=self.normalize_depth)
