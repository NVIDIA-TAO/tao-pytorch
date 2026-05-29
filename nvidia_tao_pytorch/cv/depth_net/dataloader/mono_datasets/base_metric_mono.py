# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base Mono Dataset Class."""

import numpy as np
import torch
from torch.utils.data import Dataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_depth, read_image
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.misc import apply_3d_mask


class BaseMetricMonoDataset(Dataset):
    """Base Metric Mono Dataset Class."""

    def __init__(self, data_file, transform, min_depth=1e-3, max_depth=10.0, normalize_depth=False):
        """Initialize the Base Monocular Depth Dataset Class.

        Args:
            data_file (str): path to the data file.
            transform (callable): augmentations to apply.
            min_depth (float): minimum depth value.
            max_depth (float): maximum depth value.
            normalize_depth (bool): whether to normalize the depth.
        """
        self.transform = transform
        self.normalize_depth = normalize_depth
        self.min_depth = min_depth
        self.max_depth = max_depth

        with open(data_file, 'r') as f:
            self.filelist = f.read().splitlines()

    def read_gt_depth(self, disp_path):
        """Read Middlebury ground truth disparity and mask data.

        Args:
            disp_path (str): path to the disparity map.

        Returns:
            depth (np.ndarray): depth map.
        """
        return read_depth(disp_path, self.normalize_depth)

    def __getitem__(self, index):
        """Get item from the dataset.

        Args:
            index (int): index to retrieve.

        Returns:
            sample (dict): sample from the dataset.
        """
        split_list = self.filelist[index].split(' ')
        if len(split_list) == 1:
            left_img_path = split_list[0]
            depth_path = None
        elif len(split_list) == 2:
            left_img_path = split_list[0]
            depth_path = split_list[1]
        else:
            # ignore the right image path if list length is 3
            left_img_path = split_list[0]
            depth_path = split_list[2]

        left_image = read_image(left_img_path)
        image_size = left_image.shape[:2]

        if depth_path is not None:
            depth = np.array(read_depth(depth_path, normalize_depth=self.normalize_depth))
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
            sample['depth'] = apply_3d_mask(depth, valid_mask)
            sample['valid_mask'] = valid_mask.squeeze(0)  # (B, H, W)
        else:
            valid_mask = torch.ones(image_size[0], image_size[1]).bool()
            sample['valid_mask'] = valid_mask  # (B, H, W)
        sample['image_path'] = left_img_path
        return sample

    def __len__(self):
        """Returns length of the BaseMetricMonoDataset."""
        return len(self.filelist)
