# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" StyleGAN-XL Discriminator model. """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.diffaug import diff_augment
from nvidia_tao_pytorch.sdg.stylegan_xl.model.generator.networks_stylegan2 import FullyConnectedLayer
from nvidia_tao_pytorch.sdg.stylegan_xl.model.generator.networks_styleganxl import load_pretrained_embedding_for_embed_layers
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.blocks import conv2d, DownBlock, DownBlockPatch
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.projector import F_RandomProj
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.constants import VITS


class SingleDisc(nn.Module):
    """Single Scale Discriminator."""

    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False, embed_path=None):
        """Initializes the SingleDisc.

        Args:
            nc (int, optional): Number of input channels.
            ndf (int, optional): Number of discriminator feature channels.
            start_sz (int, optional): Initial size of the input image.
            end_sz (int, optional): Final size after downscaling.
            head (bool, optional): Whether to include a head layer at the start.
            patch (bool, optional): Whether to use patch-based downsampling blocks.
        """
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Discriminator output.
        """
        return self.main(x)


class SingleDiscCond(nn.Module):
    """Single Scale Conditional Discriminator."""

    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8,
                 head=None, patch=False, c_dim=1000, cmap_dim=64,
                 embed_path=None, rand_embedding=False):
        """Initializes the SingleDiscCond.

        Args:
            nc (int, optional): Number of input channels.
            ndf (int, optional): Number of discriminator feature channels.
            start_sz (int, optional): Initial size of the input image.
            end_sz (int, optional): Final size after downscaling.
            head (bool, optional): Whether to include a head layer at the start.
            patch (bool, optional): Whether to use patch-based downsampling blocks.
            c_dim (int, optional): Dimension of the condition vector.
            cmap_dim (int, optional): Dimension of the condition embedding.
            rand_embedding (bool, optional): Whether to initialize the embeddings randomly.
        """
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)

        # Pretrained Embeddings
        self.embed = torch.nn.Embedding(num_embeddings=1000, embedding_dim=320)  # This embed layer will be loaded with a pretrained embed if needed. Find "load_pretrained_embedding".

        # Construct layers.
        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.cmap_dim, activation='lrelu')

    def forward(self, x, c):
        """Forward pass through the conditional discriminator.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Discriminator output.
        """
        h = self.main(x)
        out = self.cls(h)

        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    """Multi-Scale Discriminator for."""

    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        patch=False,
        embed_path=None,
        **kwargs,
    ):
        """Initializes the MultiScaleD.

        Args:
            channels (list): List of channel sizes for each scale.
            resolutions (list): List of resolutions for each scale.
            num_discs (int, optional): Number of discriminators to use. Default is 4.
            proj_type (int, optional): Type of projection to use. Default is 2.
            cond (int, optional): Whether to use conditional discrimination. Default is 0.
            patch (bool, optional): Whether to use patch-based downsampling blocks. Default is False.
        """
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5], f"Invalid num_discs value: {num_discs}. Expected one of [1, 2, 3, 4, 5]."

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            # pylint: disable=trailing-comma-tuple
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch, embed_path=embed_path)],

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, rec=False):
        """Forward pass through the multi-scale discriminators.

        Args:
            features (list): List of feature maps from different scales.
            c (torch.Tensor): Condition tensor.
            rec (bool, optional): Whether to return reconstruction output. Default is False.

        Returns:
            torch.Tensor: Concatenated logits from all discriminators.
        """
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    """Projected Discriminator with Multiple Backbones."""

    def __init__(
        self,
        backbones,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        """Initializes the ProjectedDiscriminator.

        Args:
            backbones (list): List of backbone models to use.
            diffaug (bool, optional): Whether to apply differentiable augmentation. Default is True.
            interp224 (bool, optional): Whether to interpolate input to 224x224. Default is True.
            backbone_kwargs (dict, optional): Additional keyword arguments for the backbone.
        """
        super().__init__()
        self.backbones = backbones
        self.diffaug = diffaug
        self.interp224 = interp224

        # get backbones and multi-scale discs
        feature_networks, discriminators = [], []

        for _, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )

            feature_networks.append([bb_name, feat])
            discriminators.append([bb_name, disc])

        self.feature_networks = nn.ModuleDict(feature_networks)
        self.discriminators = nn.ModuleDict(discriminators)

    def train(self, mode=True):
        """Set the module in training mode.

        Args:
            mode (bool, optional): Whether to set the module in training mode. Default is True.
        """
        super().train(mode)
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

    def forward(self, x, c):
        """Forward pass through the projected discriminator.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Condition tensor.

        Returns:
            torch.Tensor: Concatenated logits from all discriminators.
        """
        logits = []

        for bb_name, feat in self.feature_networks.items():

            # apply augmentation (x in [-1, 1])
            x_aug = diff_augment(x, policy='color,translation,cutout') if self.diffaug else x

            # transform to [0,1]
            x_aug = x_aug.add(1).div(2)

            # apply F-specific normalization
            x_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(x_aug)

            # upsample if smaller, downsample if larger + VIT
            if self.interp224 or bb_name in VITS:
                x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)

            # forward pass
            features = feat(x_n)
            logits += self.discriminators[bb_name](features, c)

        return logits

    def load_pretrained_embedding(self, embedding_checkpoint):
        """Find embedding layers and load the pretrained embedding checkpoint for each layers"""
        # Example: Find all ".embed" layers in the discriminator
        load_pretrained_embedding_for_embed_layers(self, embedding_checkpoint)
