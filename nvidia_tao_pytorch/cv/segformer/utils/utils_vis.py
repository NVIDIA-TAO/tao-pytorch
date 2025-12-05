# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""SegFormer utils."""
import numpy as np
import torch
from torchvision import utils


# LandSCD mapping for external use."""
colour_mappings_landSCD = {
    '0': (255, 255, 255),
    '1': (255, 165, 0),
    '2': (230, 30, 100),
    '3': (70, 140, 0),
    '4': (218, 112, 214),
    '5': (0, 170, 240),
    '6': (127, 235, 170),
    '7': (230, 80, 0),
    '8': (205, 220, 57),
    '9': (218, 165, 32)
}


def get_color_mapping(dataset_name, num_classes=None, color_mapping_custom=None):
    """
    Get the color mapping for semantic segmentation visualization for num_classes>2.
    For binary segmentation, black and white color coding is used.

    Args:
        dataset_name (str): The name of the dataset ('LandSCD' or 'custom').
        num_classes (int, optional): The number of classes in the dataset (default is None).
        color_mapping_custom (dict, optional): Custom color mapping provided as a dictionary with class indices as keys and RGB tuples as values (default is None).

    Returns:
        dict: A dictionary containing the color mapping for each class, where class indices are the keys, and RGB tuples are the values.
    """
    output_color_mapping = None
    if color_mapping_custom is not None:
        assert num_classes == len(color_mapping_custom.keys()), \
            f"""Number of color mappings ({len(color_mapping_custom.keys())}) provided must match number of classes ({num_classes})"""
        output_color_mapping = color_mapping_custom
    elif dataset_name == 'LandSCD':
        output_color_mapping = colour_mappings_landSCD
    else:  # dataset_name=='custom' or color_mapping_custom is None:
        # If color mapping not provided, randomly generate color mapping
        color_mapping_custom = {}
        for i in range(num_classes):  # TODO: add index to output viz image to indicate each color naming - check segformer
            color_mapping_custom[str(i)] = tuple(np.random.randint(1, 254, size=(3)))  # Reserve (0,0,0) for mismatch pred
        color_mapping_custom[str('0')] = (255, 255, 255)  # Enforce no-change class to be white
        output_color_mapping = color_mapping_custom
    return output_color_mapping


def make_numpy_grid(tensor_data, pad_value=0, padding=0, already_img=False, gt=None, color_map=None, label_transform=None):
    """
    Convert a batch of PyTorch tensors into a numpy grid for visualization.

    Args:
        tensor_data (torch.Tensor): The input batch of PyTorch tensors.
        pad_value (int, optional): The padding value for the grid (default is 0).
        padding (int, optional): The padding between grid cells (default is 0).
        already_img (bool, optional): Whether the input tensor_data is already an image (default is False), no need to color map.
        gt (torch.Tensor or np.ndarray, optional): The ground truth segmentation (default is None).
        color_map (dict, optional): Custom color mapping provided as a dictionary with class indices as keys and RGB tuples as values (default is None).
            For binary segmentation, black and white color coding is used.
        label_transform (str, optional): The label transform method (default is None). if norm, the color map will be normalized to [0, 1]
    Returns:
        np.ndarray: The numpy grid for visualization.
        np.ndarray (optional): The numpy grid showing mismatches between ground truth and predicted segmentation (only for multi-class segmentation).
    """
    # TODO: make sure only 1 route even if multi/binary - binary should also use multi-class viz route
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if not already_img:
        # multi-class visualisation
        vis_multi = vis[:, :, 0]

        # Code for visualising FN/FP (gt!=pred) #TODO: check if this is needed if onloy gt is jpg/png or always
        if isinstance(gt, torch.Tensor):
            gt = gt.detach()
            gt = utils.make_grid(gt, pad_value=pad_value, padding=padding)
            gt = np.array(gt.cpu()).transpose((1, 2, 0))[:, :, 0]
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    if not already_img:
        color_coded = np.ones(np.shape(vis))
        # Can take custom color map/randomly generate color map for custom datasets
        assert color_map is not None, 'Provide a color map for output visualization'
        for key, value in color_map.items():
            color_coded[vis_multi == int(key)] = value

        if label_transform != 'norm':
            color_coded = color_coded / 255

        color_coded = color_coded.astype(float)

        # Code for visualising FN/FP (gt!=pred)
        if isinstance(gt, np.ndarray):
            color_coded_mismatch = np.copy(color_coded)
            color_coded_mismatch[vis_multi != gt] = (0, 0, 0)
            color_coded_mismatch = color_coded_mismatch.astype(float)
            return color_coded, color_coded_mismatch
        return color_coded
    return vis


def de_norm(tensor_data, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Perform de-normalization on a tensor by reversing the normalization process.

    Args:
        tensor_data (torch.Tensor): The normalized tensor data to be de-normalized.
        mean (list): The mean values used for normalization.
        std (list): The standard deviation values used for normalization.

    Returns:
        torch.Tensor: The de-normalized tensor data.
    """
    # TODO: @tichou, not sure if there is better way to do this
    if len(tensor_data.shape) == 4:
        for data in tensor_data:
            for t, m, s in zip(data, mean, std):
                t.mul_(s).add_(m)
        return tensor_data
    else:
        for t, m, s in zip(tensor_data, mean, std):
            t.mul_(s).add_(m)
        return tensor_data
