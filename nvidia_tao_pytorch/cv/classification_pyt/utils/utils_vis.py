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
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification utils."""
import os
import numpy as np
import torch
import cv2
from PIL import Image


def save_with_text_overlay(image, text, output_path, img_size=None, font_scale=0.4, font_thickness=1):
    """
    Adds a text overlay to the top-left corner of an image.

    Parameters:
    - image: A PIL Image or numpy array representing the image.
    - text: The text to overlay on the image.
    - output_path: Path to save the resulting image.
    - font_scale: Font size for the text.
    - font_thickness: Thickness of the font.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, str):
        image = Image.open(image)
        # convert PIL image to cv2 image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            # make image from 0~1 to 0~255
            image = (image * 255).to(torch.uint8)
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
        # Ensure the image is in BGR format for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            # convert RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError("Unsupported image format!")

        # Resize the image if necessary, using bicubic interpolation
        if img_size is not None:
            image = cv2.resize(image, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)

    # Set up text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30)  # Top-left corner
    color = (255, 255, 255)  # White text
    background_color = (0, 0, 0)  # Black background for text
    margin = 5  # Padding around text

    # Convert text to a string that align with mm's prediction
    text = "Prediction: " + ", ".join([str(x) for x in text])

    # Calculate text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    # Create background rectangle for text
    top_left = (org[0] - margin, org[1] - text_height - margin)
    bottom_right = (org[0] + text_width + margin, org[1] + margin)
    cv2.rectangle(image, top_left, bottom_right, background_color, -1)

    # Add text on top of the rectangle
    cv2.putText(image, text, org, font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    # Save the resulting image
    cv2.imwrite(output_path, image)


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


def sync_tensor(tensor: torch.Tensor | float, reduce_method="mean") -> torch.Tensor | list[torch.Tensor]:
    """
    Syncs a tensor across all GPUs.

    Args:
        tensor (torch.Tensor | float): The tensor to sync.
        reduce_method (str): The reduction method to use. Options are "mean", "cat", and "root".

    Returns:
        torch.Tensor | list[torch.Tensor]: The synced tensor.
    """
    if not torch.distributed.is_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(int(os.environ["WORLD_SIZE"]))]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce_method == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce_method == "root":
        return tensor_list[0]
    else:
        return tensor_list
