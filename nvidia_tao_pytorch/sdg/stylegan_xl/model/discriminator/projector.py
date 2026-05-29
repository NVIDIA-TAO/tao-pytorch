# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Projector of Discriminator. """

import torch.nn as nn

from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.vit import forward_vit
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.pretrained_builder import _make_pretrained
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.constants import NORMALIZED_INCEPTION, NORMALIZED_IMAGENET, NORMALIZED_CLIP, VITS
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.blocks import FeatureFusionBlock


def get_backbone_normstats(backbone):
    """Retrieve normalization statistics for a given backbone.

    Args:
        backbone (str): The name of the backbone model.

    Returns:
        dict: A dictionary containing 'mean' and 'std' for normalization.

    Raises:
        NotImplementedError: If the backbone is not recognized.
    """
    if backbone in NORMALIZED_INCEPTION:
        return {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        }

    elif backbone in NORMALIZED_IMAGENET:
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

    elif backbone in NORMALIZED_CLIP:
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        }

    else:
        raise NotImplementedError(f"Backbone stats unavailable for {backbone}. Please choose backbone among:{[*NORMALIZED_CLIP, *NORMALIZED_IMAGENET, *NORMALIZED_INCEPTION]}")


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    """Create cross-channel mixing layers for the scratch network.

    Args:
        scratch (nn.Module): The scratch network to which layers will be added.
        in_channels (list): Number of input channels for each layer.
        cout (int): Number of output channels for each layer.
        expand (bool, optional): Whether to expand the number of output channels. Default is False.

    Returns:
        nn.Module: The scratch network with added cross-channel mixing layers.
    """
    # shapes
    out_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    """Create cross-scale mixing layers for the scratch network.

    Args:
        scratch (nn.Module): The scratch network to which layers will be added.
        in_channels (list): Number of input channels for each layer.
        cout (int): Number of output channels for each layer.
        expand (bool): Whether to expand the number of output channels.

    Returns:
        nn.Module: The scratch network with added cross-scale mixing layers.
    """
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4

    return scratch


def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    """Build a feature network and optional cross-channel and cross-scale mixing layers.

    Args:
        im_res (int): Image resolution.
        backbone (str): The name of the backbone model.
        cout (int): Number of output channels for the mixing layers.
        proj_type (int): Type of projection (0: no projection, 1: cross-channel mixing, 2: cross-scale mixing).
        expand (bool, optional): Whether to expand the number of output channels. Default is False.

    Returns:
        tuple: A tuple containing:
            - Pretrained feature network (nn.Module).
            - Scratch network with optional mixing layers (nn.Module), or None if proj_type is 0.
    """
    assert proj_type in [0, 1, 2], "Invalid projection type"

    # Build pretrained feature network
    pretrained = _make_pretrained(backbone)

    # Following Projected GAN
    im_res = 256
    pretrained.RESOLUTIONS = [im_res // 4, im_res // 8, im_res // 16, im_res // 32]

    if proj_type == 0:
        return pretrained, None

    # Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)

    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1:
        return pretrained, scratch

    # build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res * 2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class F_Identity(nn.Module):
    """Identity function, returning input as-is."""

    def forward(self, x):
        """Forward pass through the identity function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor, same as input.
        """
        return x


class F_RandomProj(nn.Module):
    """Feature Network with Random Projection for Image Generation."""

    def __init__(
        self,
        backbone="tf_efficientnet_lite3",
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        """Initializes the F_RandomProj.

        Args:
            backbone (str, optional): Name of the backbone model. Default is "tf_efficientnet_lite3".
            im_res (int, optional): Image resolution. Default is 256.
            cout (int, optional): Number of output channels. Default is 64.
            expand (bool, optional): Whether to expand the number of output channels. Default is True.
            proj_type (int, optional): Type of projection (0: no projection, 1: cross-channel mixing, 2: cross-scale mixing). Default is 2.
            **kwargs: Additional keyword arguments for the feature network and mixing layers.
        """
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        self.normstats = get_backbone_normstats(backbone)

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone, cout=self.cout,
                                                        proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        """Forward pass through the feature network with optional projection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary of feature maps at different layers.
        """
        # predict feature maps
        if self.backbone in VITS:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        else:
            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0:
            return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1:
            return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out
