# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Class for NYUDV2 data."""

import numpy as np
import torch
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_gt_nyudv2, read_image
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.misc import apply_3d_mask
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_datasets.base_metric_mono import BaseMetricMonoDataset


class NYUDV2(BaseMetricMonoDataset):
    """Dataset class for NYUDV2, providing ground truth in Metric Depth format."""

    def __getitem__(self, index):
        """Get item from the NYUDV2 dataset.

        Args:
            index (int): index to retrieve.

        Returns:
            sample (dict): sample from the NYUDV2 dataset.
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
            depth = np.array(read_gt_nyudv2(depth_path, normalize_depth=self.normalize_depth, return_disparity=False))
            depth_dict = {'depth': depth}
        else:
            depth_dict = {}

        if self.transform is not None:
            sample = self.transform({'image': left_image, **depth_dict})
        else:
            sample = {'image': left_image, **depth_dict}

        sample['image'] = torch.from_numpy(sample['image'])
        sample['image_size'] = torch.tensor([image_size[0], image_size[1]])

        if 'depth' in sample:
            depth = torch.from_numpy(sample['depth'])  # (1, H, W)
            valid_mask = torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            eval_mask[45:471, 41:601] = 1
            eval_mask = eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
            sample['valid_mask'] = valid_mask.squeeze(0)  # (B, H, W)
            sample['depth'] = apply_3d_mask(depth, valid_mask)
        else:
            valid_mask = torch.ones(image_size[0], image_size[1]).bool()
            sample['valid_mask'] = valid_mask  # (B, H, W)

        sample['image_path'] = left_img_path
        return sample

    def __len__(self):
        """Returns length of the NYUDV2 dataset."""
        return len(self.filelist)
