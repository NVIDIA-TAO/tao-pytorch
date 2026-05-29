# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Class for Crestereo data."""

from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_gt_crestereo
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets.base_relative_mono import BaseRelativeMonoDataset


class Crestereo(BaseRelativeMonoDataset):
    """Dataset class for Crestereo, providing ground truth in disparity format."""

    def read_gt_depth(self, disp_path):
        """Read Crestereo ground truth disparity and mask data.

        Args:
            disp_path (str): path to the disparity map.

        Returns:
            depth (np.ndarray): depth map.
        """
        return read_gt_crestereo(disp_path, normalize_depth=self.normalize_depth)
