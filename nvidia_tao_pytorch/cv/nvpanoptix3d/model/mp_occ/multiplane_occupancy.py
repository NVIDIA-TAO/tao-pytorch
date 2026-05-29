# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-plane occupancy head."""

import torch
from torch import nn
from torch.nn import functional as F


class UpProjection(nn.Module):
    """
    Up-projection block for feature map upsampling with skip connections.

    This module performs bilinear upsampling followed by two parallel convolutional
    branches that are summed together, creating a residual-style connection for
    better gradient flow during training.
    """

    def __init__(self, num_input_features, num_output_features):
        """Initialize the up-projection block.

        Args:
            num_input_features: Number of input feature channels
            num_output_features: Number of output feature channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        """Forward pass with upsampling and dual-branch processing.

        Args:
            x: Input feature map
                - Shape: (N, C_in, H, W)
            size: Target spatial size for upsampling as (height, width)

        Returns:
            out: Upsampled and processed feature map
                - Shape: (N, C_out, size[0], size[1])
        """
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out


class Decoder(nn.Module):
    """
    Decoder module for progressive feature map upsampling.

    This decoder takes multi-scale features from a backbone network and progressively
    upsamples them through multiple up-projection stages, combining information from
    different resolution levels.
    """

    def __init__(self, block_channel):
        """Initialize the decoder module.

        Args:
            block_channel: List of channel dimensions for each resolution level
                - Expected format: [C1, C2, C3, C4] where C1 > C2 > C3 > C4
                - Example: [2048, 1024, 512, 256]
        """
        super().__init__()
        self.conv = nn.Conv2d(
            block_channel[0], block_channel[1], kernel_size=1, stride=1, bias=False
        )
        self.bn = nn.BatchNorm2d(block_channel[1])

        self.up1 = UpProjection(
            num_input_features=block_channel[1], num_output_features=block_channel[2]
        )

        self.up2 = UpProjection(
            num_input_features=block_channel[2], num_output_features=block_channel[3]
        )

        add_feat_channel = block_channel[3]
        self.up3 = UpProjection(
            num_input_features=add_feat_channel, num_output_features=add_feat_channel // 2
        )

        add_feat_channel = add_feat_channel // 2
        self.up4 = UpProjection(
            num_input_features=add_feat_channel, num_output_features=add_feat_channel // 2
        )

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        """Forward pass through the decoder with progressive upsampling.

        Args:
            x_block1: Features from lowest resolution block
                - Shape: (N, block_channel[0], H1, W1)
            x_block2: Features from second resolution block
                - Shape: (N, C2, H2, W2) where H2 > H1, W2 > W1
            x_block3: Features from third resolution block
                - Shape: (N, C3, H3, W3) where H3 > H2, W3 > W2
            x_block4: Features from highest resolution block
                - Shape: (N, block_channel[3], H4, W4) where H4 > H3, W4 > W3

        Returns:
            x_d4: Decoded feature map at 2x the resolution of x_block1
                - Shape: (N, block_channel[3]//4, H1*2, W1*2)
        """
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        return x_d4


class MultiFeatureFusion(nn.Module):
    """
    Multi-feature fusion module for combining multi-scale features.

    This module takes features from multiple resolution levels, upsamples them to
    a common resolution, and concatenates them to create a rich multi-scale feature
    representation.
    """

    def __init__(self, block_channel, num_features=64, num_output_features=16):
        """Initialize the multi-feature fusion module.

        Args:
            block_channel: List of channel dimensions for each resolution level
                - Expected format: [C1, C2, C3, C4]
            num_features: Total number of output features after fusion (default: 64)
                - Must equal sum of all up-projection outputs (4 * 16 = 64)
        """
        super().__init__()
        assert num_features == 4 * 16, "num_features must be 4 * 16."
        assert len(block_channel) == 4, "block_channel must be a list of 4 elements."
        self.up1 = UpProjection(num_input_features=block_channel[3], num_output_features=num_output_features)
        self.up2 = UpProjection(num_input_features=block_channel[2], num_output_features=num_output_features)
        self.up3 = UpProjection(num_input_features=block_channel[1], num_output_features=num_output_features)
        self.up4 = UpProjection(num_input_features=block_channel[0], num_output_features=num_output_features)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        """Forward pass to fuse multi-scale features.

        Args:
            x_block1: Features from first resolution block
                - Shape: (N, block_channel[3], H1, W1)
            x_block2: Features from second resolution block
                - Shape: (N, block_channel[2], H2, W2)
            x_block3: Features from third resolution block
                - Shape: (N, block_channel[1], H3, W3)
            x_block4: Features from fourth resolution block
                - Shape: (N, block_channel[0], H4, W4)
            size: Target spatial size for all features as (height, width)

        Returns:
            x: Fused multi-scale features
                - Shape: (N, num_features, size[0], size[1])
        """
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)
        return x


class PredictionHead(nn.Module):
    """
    Occupancy prediction head for multi-plane occupancy estimation.

    This module takes fused features and produces per-plane occupancy predictions
    through a series of convolutions. The output is resized to a fixed target
    resolution (120x160) before making predictions.
    """

    def __init__(self, channel, num_class=1, target_size=(120, 160)):
        """Initialize the occupancy head module.

        Args:
            channel: Number of input feature channels
            num_class: Number of output classes for occupancy prediction (default: 1)
                - Typically set to 100 for multi-plane (depth bin) predictions
        """
        super().__init__()

        self.target_size = target_size
        self.resize = UpProjection(num_input_features=channel, num_output_features=channel)

        self.conv0 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(channel)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, num_class, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        """Forward pass to generate occupancy predictions.

        Args:
            x: Input feature map
                - Shape: (N, channel, H, W)

        Returns:
            x2: Occupancy prediction logits
                - Shape: (N, num_class, 120, 160)
        """
        x0 = self.resize(x, self.target_size)  # resize to 120*160
        x0 = self.conv0(x0)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        return x2


class MultiPlaneOccupancyHead(nn.Module):
    """
    Multi-plane occupancy head for 3D scene understanding.

    This is the main module that combines decoder, multi-feature fusion, and prediction
    components to generate multi-plane occupancy predictions from multi-scale backbone
    features. It predicts occupancy at multiple depth planes (100 bins) for 3D reconstruction.
    """

    def __init__(
        self,
        block_channel: list[int] | None = None,
        feature_key: list[str] | None = None,
        feature_channels: int = 64,
        num_bins: int = 100
    ):
        """Initialize the multi-plane occupancy head.

        Initializes decoder, multi-feature fusion, and prediction head with fixed
        architecture designed for ResNet-style backbones with feature maps at
        res2, res3, res4, and res5 levels.
        Args:
            block_channel: Backbone channel dimensions ordered from res5 to res2.
                Defaults to [2048, 1024, 512, 256] for ResNet-style backbones.
            feature_key: Feature map keys in the input dict, ordered from res2 to res5.
                Defaults to ['res2', 'res3', 'res4', 'res5'].
            feature_channels: Number of fused features produced by the MFF block.
                Defaults to 64 (4 * 16).
            num_bins: Number of depth bins for multi-plane occupancy prediction.
                Defaults to 100.
        """
        super().__init__()
        block_channel = block_channel or [2048, 1024, 512, 256]
        self.feature_key = feature_key or ['res2', 'res3', 'res4', 'res5']
        assert len(block_channel) == len(self.feature_key), "block_channel and feature_key must have the same length."

        self.decoder = Decoder(block_channel)
        self.multi_feature_fuser = MultiFeatureFusion(block_channel, feature_channels)
        head_channels = block_channel[-1] // 4 + feature_channels
        self.num_bins = num_bins
        self.prediction = PredictionHead(head_channels, self.num_bins)

    def forward(self, x):
        """Forward pass to generate multi-plane occupancy predictions.

        Args:
            x: Dictionary of multi-scale features from backbone
                - Keys: ['res2', 'res3', 'res4', 'res5']
                - x['res2']: Shape (N, 256, H/4, W/4)
                - x['res3']: Shape (N, 512, H/8, W/8)
                - x['res4']: Shape (N, 1024, H/16, W/16)
                - x['res5']: Shape (N, 2048, H/32, W/32)

        Returns:
            occ_pred: Multi-plane occupancy prediction logits
                - Shape: (N, 100, 120, 160)
                - 100 depth bins, each predicting occupancy at that depth plane
        """
        x_block1, x_block2, x_block3, x_block4 = x[self.feature_key[0]], x[self.feature_key[1]], \
            x[self.feature_key[2]], x[self.feature_key[3]]
        x_decoder = self.decoder(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.multi_feature_fuser(
            x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)]
        )

        x_feat = torch.cat((x_decoder, x_mff), 1)
        occ_pred = self.prediction(x_feat)
        return occ_pred
