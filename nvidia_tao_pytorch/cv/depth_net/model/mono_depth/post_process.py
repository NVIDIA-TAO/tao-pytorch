# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Post processing for inference. """

import torch
import torch.nn.functional as F
from torch import nn


class PostProcess(nn.Module):
    """Post-processing module for depth estimation inference.

    This module converts the model's raw output predictions to the original image
    size by performing unpadding and interpolation operations. It handles the
    conversion from padded model outputs back to the original image dimensions
    and formats the results for evaluation and visualization.

    The module processes batched inputs and outputs, handling cases where
    images may have been resized and padded during preprocessing for model input.

    Attributes:
        None: This module has no learnable parameters.
    """

    def __init__(self) -> None:  # pylint: disable=useless-parent-delegation
        """Initialize the PostProcess module.

        This constructor initializes the post-processing module. Since this
        module only performs deterministic operations (unpadding and interpolation),
        no parameters are required.
        """
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs, outputs, image_size, valid_mask, resized_size=None, image_names=None, gt_depth=None):
        """Perform post-processing on model outputs.

        This method converts raw model predictions back to the original image
        dimensions by unpadding and interpolating the outputs. It handles both
        cases where images were resized and padded during preprocessing.

        Args:
            inputs (torch.Tensor): Input image tensor of shape (B, C, H, W) where
                B is batch size, C is number of channels, H and W are padded dimensions.
            outputs (torch.Tensor): Raw model outputs of shape (B, H'', W'') where
                H'' and W'' are the padded output dimensions from the model.
            image_size (torch.Tensor): Original image sizes of shape (B, 2) containing
                [height, width] for each image in the batch.
            valid_mask (List[torch.Tensor]): Valid pixel masks for each image in the batch.
                Each mask has shape (H', W') where H' and W' are the resized dimensions.
            resized_size (torch.Tensor, optional): Resized image dimensions before padding
                of shape (B, 2) containing [height, width]. If None, assumes no resizing
                was performed. Defaults to None.
            image_names (List[str], optional): List of image file names for each image
                in the batch. Used for result identification. Defaults to None.
            gt_depth (List[torch.Tensor], optional): Ground truth depth maps for each
                image in the batch. Each depth map has shape (1, H', W'). If None,
                ground truth information will not be included in results. Defaults to None.

        Returns:
            List[dict]: List of dictionaries containing post-processed results for each
                image in the batch. Each dictionary contains:
                - depth_pred (torch.Tensor): Interpolated depth prediction of original size
                - image (torch.Tensor): Input image interpolated to original size
                - image_names (str): Image file name
                - image_size (torch.Tensor): Original image dimensions [height, width]
                - disp_gt (torch.Tensor): Ground truth depth/disparity (if provided)
                - valid_mask (torch.Tensor): Valid pixel mask
        """
        pred_results = []

        for i in range(len(valid_mask)):
            # unpad outputs and inputs
            if resized_size is not None:
                depth_pred = outputs[i, 0:resized_size[i][0], 0:resized_size[i][1]]
                input_image = inputs[i, :, 0:resized_size[i][0], 0:resized_size[i][1]]
            else:
                depth_pred = outputs[i]
                input_image = inputs[i]

            # interpolate depth_pred to original size
            depth_pred = F.interpolate(depth_pred[None, None, ...], (image_size[i][0], image_size[i][1]), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            input_image = F.interpolate(input_image[None, ...], (image_size[i][0], image_size[i][1]), mode='bilinear', align_corners=True).squeeze(0)
            if gt_depth is not None:
                gt = gt_depth[i].squeeze(0)
            else:
                gt = None

            pred_dict = {"depth_pred": depth_pred, "image": input_image, "image_names": image_names[i], "image_size": image_size[i], "disp_gt": gt, "valid_mask": valid_mask[i]}
            pred_results.append(pred_dict)

        return pred_results
