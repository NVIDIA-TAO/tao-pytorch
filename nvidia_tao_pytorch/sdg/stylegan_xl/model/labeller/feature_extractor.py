# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BigDatasetGAN Feature Extractor. """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from nvidia_tao_pytorch.sdg.stylegan_xl.utils import gen_utils


def save_tensors(module: nn.Module, features, name: str):
    """Process and save activations in the module.

    Args:
        module (nn.Module): The module to save activations in.
        features: The activations to be saved.
        name (str): The name under which to save the activations.
    """
    if type(features) in [list, tuple]:
        features = [f if f is not None else None
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features)


def save_out_hook(self, inp, out):
    """Hook to save output activations.

    Args:
        self (nn.Module): The module to save activations in.
        inp: The input to the module.
        out: The output from the module.

    Returns:
        The output from the module.
    """
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    """Hook to save input activations.

    Args:
        self (nn.Module): The module to save activations in.
        inp: The input to the module.
        out: The output from the module.

    Returns:
        The output from the module.
    """
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    """Parent feature extractor class."""

    def __init__(self, generator: torch.nn.Module,
                 generator_checkpoint: dict,
                 input_activations: bool, **kwargs):
        """Initializes the FeatureExtractor.

        Args:
            generator (torch.nn.Module): The generator model.
            generator_checkpoint (dict): The checkpoint for the generator model.
            input_activations (bool): If True, features are input activations of the corresponding blocks.
                                      If False, features are output activations of the corresponding blocks.
        """
        super().__init__()
        self._load_pretrained_model(generator, generator_checkpoint, **kwargs)
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, generator: torch.nn.Module, generator_checkpoint: dict, **kwargs):
        """Loads the pretrained model.

        Args:
            generator (torch.nn.Module): The generator model.
            generator_checkpoint (dict): The checkpoint for the generator model.
        """
        pass


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


class FeatureExtractorSGXL(FeatureExtractor):
    """Wrapper to extract features from pretrained StyleganXL."""

    def __init__(self, blocks: List[int], **kwargs):
        """Initializes the FeatureExtractorSGXL.

        Args:
            blocks (List[int]): List of the Stylegan blocks.
        """
        super().__init__(**kwargs)

        # Save decoder activations
        self.block_channels = []
        for idx, block in enumerate(self.model.synthesis.children()):
            if idx in blocks:
                self.block_channels.append(block.out_channels)
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, generator: torch.nn.Module, generator_checkpoint: dict, **kwargs):
        """Loads the pretrained model and sets it to evaluation mode.

        Args:
            generator (torch.nn.Module): The generator model.
            generator_checkpoint (dict): The checkpoint for the generator model.
        """
        G = generator
        G.load_state_dict(generator_checkpoint)
        G = G.eval().requires_grad_(False)

        self.model = G

    @torch.no_grad()
    def forward(self, seeds, class_idx, device,
                centroids_path=None, translate=[0.0, 0.0], rotate=0.0, truncation_psi=1.0):
        """Forward function to generate images and extract features.

        Args:
            seeds (List[int]): List of seeds for generating images.
            class_idx (int): Class index for conditional generation.
            device (torch.device): Device to run the model on.
            centroids_path (str, optional): Path to centroids for truncation. Default is None.
            translate (List[float], optional): Translation values for x and y axes. Default is [0.0, 0.0].
            rotate (float, optional): Rotation angle in degrees. Default is 0.0.
            truncation_psi (float, optional): Truncation psi value. Default is 1.0.

        Returns:
            Tuple[Tensor, List[Tensor]]: Generated images and extracted features.
        """
        feats_dict = {}
        img_list = []
        for seed in seeds:
            if hasattr(self.model.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                self.model.synthesis.input.transform.copy_(torch.from_numpy(m))

            w = gen_utils.get_w_from_seed(self.model, 1, device, truncation_psi, seed=seed,
                                          centroids_path=centroids_path, class_idx=class_idx)

            img = gen_utils.w_to_img(self.model, w, to_np=False)
            img_list.append(img)

            # Extract activations
            for i, block in enumerate(self.feature_blocks):
                if (i in feats_dict.keys()) is False:
                    feats_dict[i] = [block.activations.float()]
                else:
                    feats_dict[i].append(block.activations.float())

                block.activations = None

        # merge features
        feats_list = []
        for i in range(len(feats_dict.keys())):
            feats_list.append(
                torch.cat(feats_dict[i], dim=0)
            )

        # resize features
        feats_list[0] = F.interpolate(feats_list[0], size=(16, 16), mode='bilinear', align_corners=False)
        feats_list[1] = F.interpolate(feats_list[1], size=(32, 32), mode='bilinear', align_corners=False)
        feats_list[2] = F.interpolate(feats_list[2], size=(64, 64), mode='bilinear', align_corners=False)
        feats_list[3] = F.interpolate(feats_list[3], size=(128, 128), mode='bilinear', align_corners=False)

        img_list = torch.cat(img_list, dim=0) / 255.0

        return (img_list, feats_list)

    @torch.no_grad()
    def forward_with_numpy_images(self, seeds, class_idx, device,
                                  centroids_path=None, translate=[0.0, 0.0], rotate=0.0, truncation_psi=1.0):
        """Forward function to generate images and extract features, with images in numpy format.

        Args:
            seeds (List[int]): List of seeds for generating images.
            class_idx (int): Class index for conditional generation.
            device (torch.device): Device to run the model on.
            centroids_path (str, optional): Path to centroids for truncation. Default is None.
            translate (List[float], optional): Translation values for x and y axes. Default is [0.0, 0.0].
            rotate (float, optional): Rotation angle in degrees. Default is 0.0.
            truncation_psi (float, optional): Truncation psi value. Default is 1.0.

        Returns:
            Tuple[List[np.ndarray], List[Tensor]]: Generated images in numpy format and extracted features.
        """
        feats_dict = {}
        img_list = []
        for seed in seeds:
            if hasattr(self.model.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                self.model.synthesis.input.transform.copy_(torch.from_numpy(m))

            w = gen_utils.get_w_from_seed(self.model, 1, device, truncation_psi, seed=seed,
                                          centroids_path=centroids_path, class_idx=class_idx)

            img = gen_utils.w_to_img(self.model, w, to_np=True)
            img_list.append(img)

            # Extract activations
            for i, block in enumerate(self.feature_blocks):
                if (i in feats_dict.keys()) is False:
                    feats_dict[i] = [block.activations.float()]
                else:
                    feats_dict[i].append(block.activations.float())

                block.activations = None

        # merge features
        feats_list = []
        for i in range(len(feats_dict.keys())):
            feats_list.append(
                torch.cat(feats_dict[i], dim=0)
            )

        # resize features
        feats_list[0] = F.interpolate(feats_list[0], size=(16, 16), mode='bilinear', align_corners=False)
        feats_list[1] = F.interpolate(feats_list[1], size=(32, 32), mode='bilinear', align_corners=False)
        feats_list[2] = F.interpolate(feats_list[2], size=(64, 64), mode='bilinear', align_corners=False)
        feats_list[3] = F.interpolate(feats_list[3], size=(128, 128), mode='bilinear', align_corners=False)

        return (img_list, feats_list)
