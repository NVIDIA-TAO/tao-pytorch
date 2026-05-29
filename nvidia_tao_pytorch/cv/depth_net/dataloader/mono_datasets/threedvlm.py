# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Class for 3DVLM data."""

import numpy as np
import torch
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_image, read_gt_3dvlm
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets.base_relative_mono import BaseRelativeMonoDataset


class ThreeDVLM(BaseRelativeMonoDataset):
    """Dataset class for 3DVLM, providing ground truth in disparity format."""

    def __getitem__(self, index):
        """Get item from the ThreeDVLM dataset.

        Args:
            index (int): index to retrieve.

        Returns:
            sample (dict): sample from the ThreeDVLM dataset.
        """
        split_list = self.filelist[index].split(' ')
        if len(split_list) == 1:
            left_img_path = split_list[0]
            depth_path = None
        elif len(split_list) == 2:
            left_img_path = split_list[0]
            depth_path = split_list[1]
        else:
            left_img_path = split_list[0]
            depth_path = split_list[2]

        left_image = read_image(left_img_path)
        image_size = left_image.shape[:2]

        if depth_path is not None:
            depth = np.array(read_gt_3dvlm(depth_path, normalize_depth=self.normalize_depth, return_disparity=True))
            depth_dict = {'disparity': depth}
        else:
            depth_dict = {}

        if self.transform is not None:
            sample = self.transform({'image': left_image, **depth_dict})
        else:
            sample = {'image': left_image, **depth_dict}

        sample['image'] = torch.from_numpy(sample['image'])
        sample['image_size'] = torch.tensor([image_size[0], image_size[1]])

        if "disparity" in sample:
            sample['disparity'] = torch.from_numpy(sample['disparity'])  # (1, H, W)
            valid_mask = sample['disparity'] < 1000  # (1, H, W)
            sample['disparity'][valid_mask == 0] = 0
            sample['valid_mask'] = valid_mask.squeeze(0)  # (B, H, W)
        else:
            valid_mask = torch.ones(image_size[0], image_size[1]).bool()
            sample['valid_mask'] = valid_mask  # (B, H, W)

        sample['image_path'] = left_img_path
        return sample

    def __len__(self):
        """Returns length of the ThreeDVLM dataset."""
        return len(self.filelist)
