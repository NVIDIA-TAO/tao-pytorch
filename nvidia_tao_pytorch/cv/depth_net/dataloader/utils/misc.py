# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet utils Misc module."""

import matplotlib
import numpy as np
import cv2
import os
import torch


def apply_3d_mask(tensor, mask, value=0):
    """
    Applies a 3D boolean mask to a 3D tensor, keeping the original shape.

    Args:
        tensor: A 3D PyTorch tensor or numpy array.
        mask: A 3D boolean mask tensor of the same shape as the input tensor.
        value: The value to fill in where the mask is False (default: 0).

    Returns:
        A new 3D tensor with the mask applied, preserving the original shape.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.where(mask, tensor, torch.full_like(tensor, value))
    elif isinstance(tensor, np.ndarray):
        return np.where(mask, tensor, np.full_like(tensor, value))
    else:
        raise ValueError("Input tensor must be a PyTorch tensor or numpy array.")


def check_if_any_element_in_list_is_in_string(string, my_list):
    """
    Checks if any element from a list is present within a string.

    Args:
        string: The string to search within.
        my_list: The list of elements to check for.

    Returns:
        True if any element from the list is found in the string, False otherwise.
    """
    for element in my_list:
        if element in string:
            return element
    return None


def sanity_check_data_model(gt_depth, model_type):
    """sanity check on data type and model type.

    Args:
        gt_depth (str): Ground truth depth.
        model_type (str): Model type.
    """
    if ("relative" in gt_depth.lower() and "relative" in model_type.lower()):
        return True
    elif ("metric" in gt_depth.lower() and "metric" in model_type.lower()):
        return True
    elif ("disparity" in gt_depth.lower() and "stereo" in model_type.lower()):
        return True
    else:
        return False


def save_inference_batch(predictions, output_dir, aug_config,  normalize_depth=False):
    """Save batched inference in format.

    Args:
        predictions (list): Predictions.
        output_dir (str): Output directory.
        aug_config (dict): Augmentation configuration.
        normalize_depth (bool): Normalize depth map.

    Returns:
        None
    """
    for pred in predictions:
        image_name = pred['image_names']
        pred_depth = pred['depth_pred']
        img1 = pred['image']
        disp_gt = pred['disp_gt']
        if "valid_mask" in pred:
            valid_mask = pred['valid_mask']
        else:
            valid_mask = None

        img1 = img1.permute(1, 2, 0).data.cpu().numpy()
        # img1 = img1.data.cpu().numpy()
        # denormalize image for visualization
        img1 = img1 * aug_config["input_std"] + aug_config["input_mean"]
        img1 = img1 * 255

        path_list = image_name.split(os.sep)
        basename, extension = os.path.splitext(path_list[-1])

        folder_name = path_list[:-1]
        output_annotate_root = os.path.join(output_dir, *folder_name)

        os.makedirs(output_annotate_root, exist_ok=True)
        output_image_name = os.path.join(output_annotate_root, basename + extension)
        disp_vis = vis_disparity(pred_depth.data.cpu().numpy(), normalize_depth=normalize_depth, valid_mask=valid_mask.data.cpu().numpy())
        if disp_gt is not None:
            gt_vis = vis_disparity(disp_gt.data.cpu().numpy(), normalize_depth=normalize_depth, valid_mask=valid_mask.data.cpu().numpy())
            concat_img = np.concatenate([img1, gt_vis, disp_vis], axis=1).astype(np.uint8)
        else:
            concat_img = np.concatenate([img1, disp_vis], axis=1).astype(np.uint8)
        scale = 0.5
        concat_img = cv2.resize(concat_img, fx=scale, fy=scale, dsize=None)
        concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_name, concat_img)


def vis_mono(image1, pred_disp, disp_gt, aug_config, n_sample=5, normalize_depth=False, image_path=None, valid_mask=None):
    """Visualize monocular prediction with ground truth disparity.

    Args:
        image1 (np.ndarray): Image.
        pred_disp (np.ndarray): Predicted disparity.
        disp_gt (np.ndarray): Ground truth disparity.
        aug_config (dict): Augmentation configuration.
        n_sample (int): Number of samples to visualize.
        normalize_depth (bool): Normalize depth map.
        image_path (list): Image path.
        valid_mask (np.ndarray): Valid mask.

    Returns:
        canvas (list): Visualized depth map.
    """
    ids = np.random.choice(len(image1), size=min(n_sample, len(image1)), replace=False)
    canvas = []
    canvas_caption = []
    for index in ids:
        if image_path is not None:
            canvas_caption.append(image_path[index])
        img1 = image1[index].permute(1, 2, 0).data.cpu().numpy()
        # denormalize image for visualization
        img1 = img1 * aug_config["input_std"] + aug_config["input_mean"]
        img1 = img1 * 255
        if valid_mask is not None:
            mask = valid_mask[index].cpu().numpy()
        else:
            mask = None
        gt_vis = vis_disparity(disp_gt[index].cpu().numpy(), normalize_depth=normalize_depth, valid_mask=mask)
        disp_vis = vis_disparity(pred_disp[index].cpu().numpy(), normalize_depth=normalize_depth, valid_mask=mask)

        row = np.concatenate([img1, gt_vis, disp_vis], axis=1).astype(np.uint8)
        scale = 1 / row.shape[0] * 300
        row = cv2.resize(row, fx=scale, fy=scale, dsize=None)
        canvas.append(row)

    return canvas, canvas_caption


def vis_disparity(depth, normalize_depth=False, valid_mask=None):
    """Visualize disparity with min,max normalization.

    Args:
        depth (np.ndarray): Depth map.
        normalize_depth (bool): Normalize depth map.
        valid_mask (np.ndarray): Valid mask.

    Returns:
        vis (np.ndarray): Visualized depth map.
    """
    depth = depth.copy()

    if valid_mask is not None:
        depth = apply_3d_mask(depth, valid_mask)

    if normalize_depth:
        depth = depth * 255.0
    else:
        if depth.max() == depth.min():
            depth = depth = (depth / depth.max()) * 255.0
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    cmap = matplotlib.colormaps.get_cmap('Spectral')
    vis = depth.astype(np.uint8)
    vis = (cmap(vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return vis


def parse_checkpoint(model_dict, model_type):
    """
    Parse public DepthAnythingV2 checkpoints.
    Public checkpoints are available from https://github.com/DepthAnything/Depth-Anything-V2

    Args:
        model_dict (dict): Model dictionary.
        model_type (str): Model type.

    Returns:
        final (dict): Parsed model dictionary.
    """
    is_metric = "metric" in model_type.lower()

    final = {}
    for k, v in model_dict.items():
        k = f"model.{k}"
        if is_metric and "depth_head" in k:
            k = k.replace("depth_head", "metric_depth_head")
        final[k] = v
    return final
