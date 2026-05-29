# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigDatasetGAN model builder"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.sdg.stylegan_xl.model.stylegan import build_generator, retrieve_generator_checkpoint_from_stylegan_pl_model
from nvidia_tao_pytorch.sdg.stylegan_xl.model.labeller.feature_extractor import FeatureExtractorSGXL
from nvidia_tao_pytorch.sdg.stylegan_xl.model.labeller.feature_labeller import FeatureLabellerSGXL


class BigDatasetGAN(nn.Module):
    """
    BigDatasetGAN model using a feature extractor and feature labeller.

    Args:
        blocks (list): List of feature blocks of feature extractor as the inputs for feature labeller (default is [0, 1, 2, 3]).
        n_class (int): Number of classes of the conditional stylegan generator backbone used for feature extractor (default is 0).
        class_idx (int): The desired images of classes for BigDatasetGAN to generate (default is 0).
    """

    def __init__(self, stylegan_generator, stylegan_generator_checkpoint, blocks=[0, 1, 2, 3], n_class=0, class_idx=0):
        """Initialize BigDatasetGAN class"""
        super(BigDatasetGAN, self).__init__()
        self.class_idx = class_idx
        self.n_class = n_class
        self.feature_extractor = FeatureExtractorSGXL(generator=stylegan_generator,
                                                      generator_checkpoint=stylegan_generator_checkpoint,
                                                      blocks=blocks, input_activations=False)
        self.feature_labeller = FeatureLabellerSGXL(n_class=n_class, in_channels=self.feature_extractor.block_channels)

    def onnx_forward(self, z, labels):
        """Generate images and segmentation logits using ONNX with the truncation trick.

        Args:
            z (torch.Tensor): Latent codes for generating images.
            labels (torch.Tensor): Labels for conditional generation. If unspecified, random labels are used.

        Returns:
            tuple: A tuple containing:
                - images (torch.Tensor): Generated images from the latent codes.
                - seg_logits (torch.Tensor): Segmentation logits for the generated images.
        """
        truncation_psi = 1.0
        if self.feature_extractor.model.c_dim != 0:
            # sample random labels if no class idx is given
            class_indices = torch.argmax(labels, dim=-1)
            w_avg = self.feature_extractor.model.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = self.feature_extractor.model.mapping.w_avg.unsqueeze(0)

        w = self.feature_extractor.model.mapping(z, labels)

        w_avg = w_avg.unsqueeze(1).repeat(1, self.feature_extractor.model.mapping.num_ws, 1)
        w = w_avg + (w - w_avg) * truncation_psi

        images = self.feature_extractor.model.synthesis(w)

        # Extract activations
        feats_dict = {}
        for i, block in enumerate(self.feature_extractor.feature_blocks):
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

        # assert 1==2, (img.shape, len(feats_list), feats_list[0].shape, feats_list[1].shape, feats_list[2].shape, feats_list[3].shape)
        seg_logits = self.feature_labeller(feats_list)

        return images, seg_logits

    def forward(self, seeds, class_idx, device,
                centroids_path=None, translate=[0.0, 0.0], rotate=0.0, truncation_psi=1.0):
        """Generate torch images and segmentation logits from input seeds.

        Args:
            seeds (torch.Tensor): A tensor of input seeds used for image generation.
            class_idx (int): The index representing the class for which to generate images.
            device (torch.device): The device (CPU or GPU) to perform computations on.
            centroids_path (str, optional): Path to the centroids used in the feature extraction process.
                Default is None, indicating no centroids are used.
            translate (list, optional): A list of two floats representing the translation offsets for the generated images.
                Default is [0.0, 0.0].
            rotate (float, optional): The rotation angle in degrees applied to the generated images. Default is 0.0.
            truncation_psi (float, optional): A truncation parameter that controls the variability of the generated images.
                Default is 1.0.

        Returns:
            tuple: A tuple containing:
                - img_list (list): A list of generated torch images.
                - seg_logits (torch.Tensor): The segmentation logits corresponding to the generated images.
        """
        with torch.no_grad():
            img_list, feats_list = self.feature_extractor(seeds, class_idx=class_idx, device=device,
                                                          centroids_path=centroids_path, translate=translate,
                                                          rotate=rotate, truncation_psi=truncation_psi)
        seg_logits = self.feature_labeller(feats_list)
        return img_list, seg_logits

    def forward_with_numpy_images(self, seeds, class_idx, device,
                                  centroids_path=None, translate=[0.0, 0.0], rotate=0.0, truncation_psi=1.0):
        """Generate numpy images and torch segmentation logits from input seeds.

        Args:
            seeds (numpy.ndarray): An array of input seeds used for image generation.
            class_idx (int): The index representing the class for which to generate images.
            device (torch.device): The device (CPU or GPU) to perform computations on.
            centroids_path (str, optional): Path to the centroids used in the feature extraction process.
                Default is None, indicating no centroids are used.
            translate (list, optional): A list of two floats representing the translation offsets for the generated images.
                Default is [0.0, 0.0].
            rotate (float, optional): The rotation angle in degrees applied to the generated images. Default is 0.0.
            truncation_psi (float, optional): A truncation parameter that controls the variability of the generated images.
                Default is 1.0.

        Returns:
            tuple: A tuple containing:
                - img_list (list): A list of generated numpy images.
                - seg_logits (torch.Tensor): The segmentation logits corresponding to the generated images.
        """
        with torch.no_grad():
            img_list, feats_list = self.feature_extractor.forward_with_numpy_images(seeds, class_idx=class_idx, device=device,
                                                                                    centroids_path=centroids_path, translate=translate,
                                                                                    rotate=rotate, truncation_psi=truncation_psi)
        seg_logits = self.feature_labeller(feats_list)
        return img_list, seg_logits


def build_model(experiment_config, export=False):
    """ Build BigDatasetGAN model (= feature_extractor + feature labeller) according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export # TODO

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    blocks = model_config['bigdatasetgan']['feature_extractor']['blocks']
    class_idx = dataset_config['bigdatasetgan']['class_idx']
    n_class = dataset_config['bigdatasetgan']['seg_classes']

    stylegan_generator = build_generator(experiment_config, return_stem=False)
    stylegan_generator_ema_checkpoint = retrieve_generator_checkpoint_from_stylegan_pl_model(
        model_config['bigdatasetgan']['feature_extractor']['stylegan_checkpoint_path'])

    model = BigDatasetGAN(stylegan_generator=stylegan_generator,
                          stylegan_generator_checkpoint=stylegan_generator_ema_checkpoint,
                          blocks=blocks, class_idx=class_idx, n_class=n_class)

    return model
