# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Generates TRT compatible StyleGAN-XL onnx model. """

import torch
import onnx


def affine_grid(theta, size, align_corners=False):
    """Generate an affine grid for sampling in transformations.

    Args:
        theta (torch.Tensor): The affine transformation matrix of shape (N, 2, 3).
        size (Tuple[int, int, int, int]): The target output size as (N, C, H, W).
        align_corners (bool, optional): If True, aligns corners for interpolation. Default is False.

    Returns:
        torch.Tensor: Generated affine grid of shape (N, H, W, 2).
    """
    N, C, H, W = size
    device = theta.device  # Determine the device of the input tensor
    dtype = theta.dtype
    grid = create_grid(N, C, H, W, device, dtype)  # Pass the device to create_grid
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid


def create_grid(N, C, H, W, device, dtype):
    """Create a base grid for affine transformations.

    Args:
        N (int): Batch size.
        C (int): Number of channels.
        H (int): Grid height.
        W (int): Grid width.
        device (torch.device): Device on which to create the grid.
        dtype (torch.dtype): Data type of the grid.

    Returns:
        torch.Tensor: A base grid of shape (N, H, W, 3) with specified device and dtype.
    """
    grid = torch.empty((N, H, W, 3), dtype=dtype, device=device)  # Initialize on the specified device
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, device, dtype))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, device, dtype).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid


def linspace_from_neg_one(num_steps, device, dtype):
    """Generate a linspace ranging from -1 to 1.

    Args:
        num_steps (int): Number of steps in the linspace.
        device (torch.device): Device on which to create the linspace.
        dtype (torch.dtype): Data type of the linspace.

    Returns:
        torch.Tensor: Generated linspace on the specified device and dtype.
    """
    r = torch.linspace(-1, 1, num_steps, dtype=dtype, device=device)  # Generate linspace on the specified device
    r = r * (num_steps - 1) / num_steps
    return r


def patch_affine_grid_generator():
    """Patch PyTorch's affine_grid function with a custom implementation."""
    torch.nn.functional.affine_grid = affine_grid


class ONNXExporter(object):
    """Onnx Exporter"""

    @classmethod
    def setUpClass(cls):
        """SetUpclass to set the manual seed for reproduceability"""
        torch.manual_seed(123)

    def export_model(self, model, batch_size, onnx_file, args, do_constant_folding=False, opset_version=17,
                     output_names=None, input_names=None, verbose=False):
        """ Export_model.

        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).

        Args:
            model (nn.Module): torch model to export.
            batch_size (int): batch size of the ONNX model. -1 means dynamic batch size.
            onnx_file (str): output path of the onnx file.
            args (Tuple[torch.Tensor]): Tuple of input tensors.
            do_constant_folding (bool): flag to indicate whether to fold constants in the ONNX model.
            opset_version (int): opset_version of the ONNX file.
            output_names (str): output names of the ONNX file.
            input_names (str): input names of the ONNX file.
            verbose (bool): verbosity level.
        """
        if batch_size is None or batch_size == -1:
            dynamic_axes = {
                "z": {0: "batch_size"},
                "labels": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        else:
            dynamic_axes = None

        torch.onnx.export(model, args, onnx_file,
                          input_names=input_names, output_names=output_names, export_params=True,
                          training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=do_constant_folding,
                          verbose=verbose, dynamic_axes=dynamic_axes,
                          dynamo=False)

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
