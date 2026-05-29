# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base Mono Dataset Class."""

import numpy as np
import torch
from torch.utils.data import Dataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_depth, read_image


class BaseRelativeMonoDataset(Dataset):
    """Base Relative Mono Dataset Class."""

    def __init__(self, data_file, transform, normalize_depth=False):
        """Initialize the Base Monocular Depth Dataset Class.

        Args:
            data_file (str): path to the data file.
            transform (callable): augmentations to apply.
            normalize_depth (bool): whether to normalize the depth.
        """
        self.transform = transform
        self.normalize_depth = normalize_depth

        with open(data_file, 'r') as f:
            self.filelist = f.read().splitlines()

    def read_gt_depth(self, disp_path):
        """Read ground truth disparity and mask data.

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
            depth = np.array(self.read_gt_depth(depth_path))
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
            # No GT provided, set valid mask to all 1s
            valid_mask = torch.ones(image_size[0], image_size[1]).bool()
            sample['valid_mask'] = valid_mask  # (B, H, W)  # (B, H, W)

        sample['image_path'] = left_img_path
        return sample

    def __len__(self):
        """Returns length of the BaseRelativeMonoDataset."""
        return len(self.filelist)
