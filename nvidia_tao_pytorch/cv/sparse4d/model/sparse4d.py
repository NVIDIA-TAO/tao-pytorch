# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4D model implementation."""

import inspect
import torch
import torch.nn as nn
from typing import Dict

from nvidia_tao_pytorch.cv.sparse4d.model.grid_mask import GridMask
from nvidia_tao_pytorch.cv.sparse4d.model.backbone.registry import SPARSE4D_BACKBONE_REGISTRY

from nvidia_tao_pytorch.cv.sparse4d.model.neck import build_neck
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_head import build_head
from nvidia_tao_pytorch.cv.sparse4d.model.blocks import DenseDepthNet
from nvidia_tao_pytorch.cv.sparse4d.model.ops.deformable_aggregation import feature_maps_format


class Sparse4D(nn.Module):
    """Sparse4D model for 3D object detection from multi-view images."""

    def __init__(
        self,
        config: Dict,
    ):
        """Initialize Sparse4D model.

        Args:
            img_backbone (nn.Module or Dict): Backbone network or configuration
            head (nn.Module or Dict): Detection head or configuration
            img_neck (nn.Module or Dict): Neck network or configuration
            depth_branch (nn.Module or Dict): Depth branch or configuration
            use_grid_mask (bool): Whether to use grid mask augmentation
            use_deformable_func (bool): Whether to use deformable aggregation
            pretrained (str): Path to pretrained weights
        """
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.backbone_config = self.model_config["backbone"]
        self.neck_config = self.model_config["neck"]
        self.head_config = self.model_config["head"]
        self.depth_branch_config = self.model_config["depth_branch"]
        self.backbone_type = self.backbone_config["type"]
        self.img_backbone = SPARSE4D_BACKBONE_REGISTRY.get(self.backbone_type)(out_indices=[0, 1, 2, 3], freeze_norm=True)
        del self.img_backbone.global_pool
        del self.img_backbone.fc
        self.img_backbone.set_grad_checkpointing(True)
        self.img_neck = build_neck(self.neck_config)
        self.head = build_head(self.config)
        self.use_grid_mask = self.model_config["use_grid_mask"]
        self.use_deformable_func = self.model_config["use_deformable_func"]
        depth_branch = self.model_config["depth_branch"]
        use_grid_mask = self.model_config["use_grid_mask"]
        if depth_branch is not None:
            if isinstance(depth_branch, nn.Module):
                self.depth_branch = depth_branch
            else:
                self.depth_branch = DenseDepthNet(embed_dims=depth_branch['embed_dims'], num_depth_layers=depth_branch['num_depth_layers'])
        else:
            self.depth_branch = None

        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def init_weights(self):
        """Initialize model weights."""
        # Initialize backbone
        if hasattr(self.img_backbone, 'init_weights'):
            self.img_backbone.init_weights()

        # Initialize neck if present
        if self.img_neck is not None and hasattr(self.img_neck, 'init_weights'):
            self.img_neck.init_weights()

        # Initialize head
        if hasattr(self.head, 'init_weights'):
            self.head.init_weights()

        # Initialize depth branch if present
        if self.depth_branch is not None and hasattr(self.depth_branch, 'init_weights'):
            self.depth_branch.init_weights()

    def extract_feat(self, img, return_depth=False, metas=None):
        """Extract features from input images.

        Args:
            img: Input images tensor [B, N, C, H, W] or [B*N, C, H, W]
            return_depth: Whether to return depth predictions
            metas: Additional metadata information

        Returns:
            feature_maps: Extracted feature maps
            depths: Depth predictions (if return_depth=True)
        """
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        # img = img.to(torch.float16)
        # Extract features using backbone
        signature_params = inspect.signature(self.img_backbone.forward).parameters
        if "metas" in signature_params:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone.forward_feature_pyramid(img)

        # Apply neck if present
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))

        # Reshape feature maps to [batch_size, num_cams, channels, height, width]
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        # Generate depth predictions if requested
        if return_depth and self.depth_branch is not None:
            focal = None
            if metas is not None and "focal" in metas:
                focal = metas["focal"]
            depths = self.depth_branch(feature_maps, focal)
        else:
            depths = None

        # Format feature maps for deformable aggregation if used
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        if return_depth:
            return feature_maps, depths
        return feature_maps

    def forward(self, img, metas=None, **data):
        """Forward pass.

        Args:
            img: Input images
            metas: Image metadata
            data: Additional input data

        Returns:
            Model outputs
        """
        # Handle case where data is passed as a dict for compatibility with dataloaders
        if metas is None and isinstance(data, dict):
            # If all data is in a single dict, extract components
            if 'img_metas' in data:
                metas = data
            elif len(data) > 0:
                metas = data

        if self.training:
            return self.forward_train(img, metas)
        else:
            return self.forward_test(img, metas)

    def forward_train(self, img, metas):
        """Forward pass during training.

        Args:
            img: Input images
            metas: Image metadata and other inputs

        Returns:
            Loss dictionary
        """
        # Extract features and depth predictions
        feature_maps, depths = self.extract_feat(img, True, metas)
        # Get outputs from head
        model_outs = self.head(feature_maps, metas)

        return (model_outs, depths)

    def forward_test(self, img, metas):
        """Forward pass during testing.

        Args:
            img: Input images
            metas: Image metadata and other inputs

        Returns:
            Detection results
        """
        if isinstance(img, list):
            return self.aug_test(img, metas)
        else:
            return self.simple_test(img, metas)

    def simple_test(self, img, metas):
        """Simple test without augmentation.

        Args:
            img: Input images
            metas: Image metadata and other inputs

        Returns:
            Detection results
        """
        # Extract features
        feature_maps = self.extract_feat(img, False, metas)

        # Get model outputs
        model_outs = self.head(feature_maps, metas)

        # Post-process results
        results = self.head.post_process(model_outs)

        # Format results
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, metas):
        """Test with augmentation.

        Args:
            img: List of input images
            metas: Image metadata and other inputs

        Returns:
            Detection results
        """
        # For consistency with original implementation, process the first image
        # as a simple implementation of test-time augmentation
        for key in metas.keys():
            if isinstance(metas[key], list):
                metas[key] = metas[key][0]

        return self.simple_test(img[0], metas)


def build_model(experiment_config, export=False):
    """Build Sparse4D model from experiment configuration.

    Args:
        experiment_config: Experiment configuration
        export: Whether the model will be exported

    Returns:
        Sparse4D model instance
    """
    # Create model instance
    model = Sparse4D(
        config=experiment_config
    )

    return model
