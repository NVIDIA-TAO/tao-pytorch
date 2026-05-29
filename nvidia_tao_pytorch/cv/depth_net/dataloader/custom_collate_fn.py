# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom collate function for DepthNet."""

import torch


def custom_collate_fn(batch):
    """Custom collate function for depth models to ensure uniform image resolution within a batch after padding.

    Args:
        batch (dict): dictionary of a single batch. Contains image and meta information tensors.

    Returns:
        batch (dict): dictionary of a single batch with uniform image resolution after padding along with meta information.
    """
    collated_batch = {}
    for item in batch:
        for key in item:
            if key not in collated_batch:
                collated_batch[key] = []
            collated_batch[key].append(item[key])
    collated_batch['image'], collated_batch['resized_size'] = tensor_from_tensor_list(collated_batch['image'])

    return collated_batch


def _max_by_axis(the_list):
    """Get maximum image shape for padding.

    Args:
        the_list (list): list of image shapes.

    Returns:
        maxes (list): list of maximum image shapes.
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def tensor_from_tensor_list(tensor_list):
    """Convert list of tensors with different size to fixed resolution.

    The final size is determined by largest height and width.
    In theory, the batch could become [3, 1333, 1333] on dataset with different aspect ratio, e.g. COCO
    A fourth channel dimension is the mask region in which 0 represents the actual image and 1 means the padded region.
    This is to give size information to the transformer archicture. If transform-padding is applied,
    then only the pre-padded regions gets mask value of 1.

    Args:
        tensor_list (List[Tensor]): list of image tensors
        targets (List[dict]): list of labels that contain the size information

    Returns:
        tensors (torch.Tensor): list of image tensors in shape of (B, 4, H, W)
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        temp_tensors = torch.zeros((b, c, h, w), dtype=dtype, device=device)
        resized_size = []
        for img, pad_img in zip(tensor_list, temp_tensors):
            # Get original image size before transform-padding
            # If no transform-padding has been applied,
            # then height == img.shape[1] and width == img.shape[2]
            actual_height, actual_width = img.shape[1], img.shape[2]
            pad_img[:img.shape[0], :actual_height, :actual_width].copy_(img[:, :actual_height, :actual_width])
            resized_size.append(torch.tensor([actual_height, actual_width]))
    else:
        raise ValueError('Channel size other than 3 is not supported')

    return temp_tensors, resized_size
