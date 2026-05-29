# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monocular Depth Network Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.depth_anything_v2.dinov2 import DINOV2
from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.depth_anything_v2.blocks import _make_fusion_block, _make_scratch
from nvidia_tao_pytorch.cv.depth_net.utils.misc import parse_lighting_checkpoint_to_backbone, parse_public_checkpoint_to_backbone


class MetricDPTHead(nn.Module):
    """ DPT Head for metric depth prediction."""

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        """
        Initialize MetricDPTHead.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of features for fusion blocks and output layers.
            use_bn (bool): Whether to use batch normalization in fusion blocks.
            out_channels (list of int): Output channels for each stage.
            use_clstoken (bool): Whether to use class token in transformer outputs.
        """
        super(MetricDPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, out_features, patch_h, patch_w):
        """
        Forward pass for metric depth prediction head.

        Args:
            out_features (list): List of feature tensors from the backbone.
            patch_h (int): Patch height after transformer embedding.
            patch_w (int): Patch width after transformer embedding.

        Returns:
            torch.Tensor: Predicted metric depth map.
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class DPTHead(nn.Module):
    """ DPT Head for relative depth prediction. """

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        """
        Initialize DPTHead.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of features for fusion blocks and output layers.
            use_bn (bool): Whether to use batch normalization in fusion blocks.
            out_channels (list of int): Output channels for each stage.
            use_clstoken (bool): Whether to use class token in transformer outputs.
        """
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w, normalize_output=False):
        """
        Forward pass for relative depth prediction head.

        Args:
            out_features (list): List of feature tensors from the backbone.
            patch_h (int): Patch height after transformer embedding.
            patch_w (int): Patch width after transformer embedding.

        Returns:
            torch.Tensor: Predicted relative depth map.
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        if normalize_output:
            depth = self.scratch.output_conv2(out)
            depth = F.relu(depth)
            disp = 1 / depth
            disp[depth == 0] = 0
            disp = disp / disp.max()
            return out, path_1, path_2, path_3, path_4, disp
        else:
            out = self.scratch.output_conv2(out)

        return out


class RelativeDepthAnythingV2(nn.Module):
    """ Relative depth prediction model using DPT Head. """

    def __init__(
        self,
        model_config, max_depth, export=False
    ):
        """
        Initialize RelativeDepthAnythingV2.

        Args:
            model_config (dict): Model configuration dictionary.
            max_depth (float): Maximum depth value for normalization.
            export (bool, optional): Whether the model is being used for export.
                Defaults to False.
        """
        super(RelativeDepthAnythingV2, self).__init__()
        encoder = model_config['encoder']
        use_bn = model_config['mono_backbone']['use_bn']
        use_clstoken = model_config['mono_backbone']['use_clstoken']
        self.encoder = encoder
        self.max_depth = max_depth

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'intermediate_layer_idx': [2, 5, 8, 11]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'intermediate_layer_idx': [2, 5, 8, 11]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'intermediate_layer_idx': [4, 11, 17, 23]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'intermediate_layer_idx': [9, 19, 29, 39]}
        }

        features = self.model_configs[self.encoder]['features']
        out_channels = self.model_configs[self.encoder]['out_channels']

        self.pretrained = DINOV2(model_name=encoder, export=export)

        if model_config['mono_backbone']['pretrained_path']:
            model_dict = torch.load(model_config['mono_backbone']['pretrained_path'], map_location='cpu')
            if "pytorch-lightning_version" in model_dict:
                # parse pytorch lightning checkpoint for relative/metric depth
                parsed_model_dict = parse_lighting_checkpoint_to_backbone(model_dict['state_dict'])
                self.pretrained.load_state_dict(parsed_model_dict, strict=True)
            else:
                # parse public checkpoint for relative depth
                if model_config['model_type'] == 'RelativeDepthAnything':
                    self.pretrained.load_state_dict(model_dict, strict=True)
                elif model_config['model_type'] == 'MetricDepthAnything':
                    # parse public checkpoint for metric depth
                    parsed_model_dict = parse_public_checkpoint_to_backbone(model_dict)
                    self.pretrained.load_state_dict(parsed_model_dict, strict=True)
                else:
                    raise NotImplementedError(f"Model type {model_config['model_type']} not implemented")

        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        """
        Forward pass for relative depth prediction.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted relative depth map of shape (B, H, W).
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x.contiguous(), self.model_configs[self.encoder]['intermediate_layer_idx'], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)


class MetricDepthAnythingV2(RelativeDepthAnythingV2):
    """
    Metric depth prediction model extending RelativeDepthAnythingV2.

    Args:
        model_config (dict): Model configuration dictionary.
        max_depth (float): Maximum depth value for normalization.
    """

    def __init__(
        self,
        model_config, max_depth, export=False
    ):
        """
        Initialize MetricDepthAnythingV2.

        Args:
            model_config (dict): Model configuration dictionary.
            max_depth (float): Maximum depth value for normalization.
            export (bool, optional): Whether the model is being used for export.
                Defaults to False.
        """
        super().__init__(model_config, max_depth, export=export)
        features = self.model_configs[self.encoder]['features']
        out_channels = self.model_configs[self.encoder]['out_channels']
        use_bn = model_config['mono_backbone']['use_bn']
        use_clstoken = model_config['mono_backbone']['use_clstoken']

        # remove depth_head attribute from parent class and create metric_depth_head attribute.
        delattr(self, 'depth_head')
        self.metric_depth_head = MetricDPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        """
        Forward pass for metric depth prediction.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted metric depth map of shape (B, H, W).
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x.contiguous(), self.model_configs[self.encoder]['intermediate_layer_idx'], return_class_token=True)
        depth = self.metric_depth_head(features, patch_h, patch_w)
        if self.max_depth is not None:
            depth = depth * self.max_depth
        return depth.squeeze(1)
