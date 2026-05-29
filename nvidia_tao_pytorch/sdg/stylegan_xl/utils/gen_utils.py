# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generate images."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib


def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None):
    """Create a grid with the fed images.

    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)

    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, _ = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


def w_to_img(G, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const', to_np: bool = True) -> np.ndarray:
    """
    Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
    returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].

    Args:
        G: The generator model.
        dlatents (Union[List[torch.Tensor], torch.Tensor]): The dlatent tensor(s).
        noise_mode (str, optional): The noise mode to use. Default is 'const'.
        to_np (bool, optional): Whether to convert the output to a numpy array. Default is True.

    Returns:
        np.ndarray: The generated image(s).

    Raises:
        AssertionError: If dlatents is not a torch.Tensor.
    """
    assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
    if len(dlatents.shape) == 2:
        dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]

    synth_image = G.synthesis(dlatents, noise_mode=noise_mode)
    synth_image = (synth_image + 1) * 255 / 2  # [-1.0, 1.0] -> [0.0, 255.0]
    if to_np:
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
    return synth_image


def get_w_from_seed(G, batch_sz, device, truncation_psi=1.0, seed=None, centroids_path=None, class_idx=None, deterministic=False):
    """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional).

    Args:
        G: The generator model.
        batch_sz (int): The batch size.
        device: The device to run the computation on.
        truncation_psi (float, optional): The truncation psi value. Default is 1.0.
        seed (int, optional): The random seed. Default is None.
        centroids_path (str, optional): Path to the centroids file. Default is None.
        class_idx (int, optional): The class index. Default is None.
        deterministic (bool, optional): Whether to use a deterministic random seed. Default is False.

    Returns:
        torch.Tensor: The dlatent tensor.

    Raises:
        AssertionError: If both seed and deterministic are set to True.
    """
    if (seed is not None and deterministic is True):
        raise AssertionError("Can only choose to set specific `seed` for generate same w or to set 'deterministic' flag for using global random seed")
    if G.c_dim != 0:
        # sample random labels if no class idx is given
        if class_idx is None:
            if (deterministic is False):
                class_indices = np.random.RandomState(seed).randint(low=0, high=G.c_dim, size=(batch_sz))
            else:
                class_indices = np.random.randint(low=0, high=G.c_dim, size=(batch_sz))
            class_indices = torch.from_numpy(class_indices).to(device)
            w_avg = G.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
            class_indices = torch.full((batch_sz,), class_idx).to(device)

        labels = F.one_hot(class_indices, G.c_dim)

    else:
        w_avg = G.mapping.w_avg.unsqueeze(0)
        labels = None
        if class_idx is not None:
            logging.warning('Warning: --class is ignored when running an unconditional network')

    if (deterministic is False):
        z = np.random.RandomState(seed).randn(batch_sz, G.z_dim)
    else:
        z = np.random.randn(batch_sz, G.z_dim)

    z = torch.from_numpy(z).to(device)
    w = G.mapping(z, labels)

    # multimodal truncation
    if centroids_path:

        with dnnlib.util.open_url(centroids_path, verbose=False) as f:
            w_centroids = np.load(f)
        w_centroids = torch.from_numpy(w_centroids).to(device)
        w_centroids = w_centroids[None].repeat(batch_sz, 1, 1)

        # measure distances
        dist = torch.norm(w_centroids - w[:, :1], dim=2, p=2)
        w_avg = w_centroids[0].index_select(0, dist.argmin(1))

    w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    w = w_avg + (w - w_avg) * truncation_psi

    return w


def make_transform(translate: Tuple[float, float], angle: float):
    """Create a transformation matrix for translation and rotation.

    Args:
        translate (Tuple[float, float]): Translation values for x and y axes.
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: The transformation matrix.
    """
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]

    return m
