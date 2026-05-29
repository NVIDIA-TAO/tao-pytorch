# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EdgeNeXt backbone module.

This module provides EdgeNeXt implementations for the TAO PyTorch framework.
EdgeNeXt is a hybrid CNN-Transformer architecture designed for mobile vision
applications, combining the efficiency of convolutional layers with the
representational power of transformer blocks.

The EdgeNeXt architecture introduces Split Depth-wise Transpose Attention (SDTA)
mechanisms that efficiently capture both local and global information. It uses
a hierarchical structure with four stages, each containing a mix of convolutional
and transformer blocks.

Key Features:
- Hybrid CNN-Transformer architecture for mobile applications
- Split Depth-wise Transpose Attention (SDTA) for efficient attention
- Cross-Covariance Attention (XCA) for global information
- Configurable global blocks for long-range dependencies
- Support for multiple model sizes (XX-Small, X-Small, Small, Base)
- BN-HS variants with BatchNorm and Hard Swish activation
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design for mobile and edge devices

Classes:
    EdgeNeXt: Main EdgeNeXt model with hybrid architecture
    EdgeNeXtBNHS: EdgeNeXt variant with BatchNorm and Hard Swish

Functions:
    edgenext_xx_small: EdgeNeXt XX-Small model
    edgenext_x_small: EdgeNeXt X-Small model
    edgenext_small: EdgeNeXt Small model
    edgenext_base: EdgeNeXt Base model
    edgenext_xx_small_bn_hs: EdgeNeXt XX-Small with BN-HS
    edgenext_x_small_bn_hs: EdgeNeXt X-Small with BN-HS
    edgenext_small_bn_hs: EdgeNeXt Small with BN-HS

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import edgenext_small
    >>> model = edgenext_small(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

References:
    - [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](
      https://arxiv.org/abs/2206.10589)
    - [https://github.com/mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)
"""
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import LayerNorm2d
from nvidia_tao_pytorch.cv.backbone_v2.edgenext_utils import (
    PositionalEncodingFourier,
    SDTAEncoder, ConvEncoder, SDTAEncoderBNHS, ConvEncoderBNHS
)
from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class EdgeNeXt(BackboneBase):
    """EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications.

    EdgeNeXt is a hybrid CNN-Transformer architecture designed for mobile vision applications.
    It combines the efficiency of convolutional layers with the representational power of
    transformer blocks through Split Depth-wise Transpose Attention (SDTA) mechanisms.

    The architecture consists of:
    - Stem layer with 4x4 convolution and stride 4
    - Four stages with configurable depths and dimensions
    - Global blocks (SDTA) for capturing long-range dependencies
    - Convolutional blocks for local feature extraction

    Reference:
        EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications
        https://arxiv.org/abs/2206.10589

    Args:
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
        depths (List[int], optional): Number of blocks in each stage. Defaults to [3, 3, 9, 3].
        dims (List[int], optional): Channel dimensions for each stage. Defaults to [24, 48, 88, 168].
        global_block (List[int], optional): Number of global blocks in each stage. Defaults to [0, 0, 0, 3].
        global_block_type (List[str], optional): Type of global block for each stage.
            Must be 'None' or 'SDTA'. Defaults to ['None', 'None', 'None', 'SDTA'].
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        head_init_scale (float, optional): Initial scaling for classification head. Defaults to 1.0.
        expan_ratio (int, optional): Expansion ratio for inverted bottleneck blocks. Defaults to 4.
        kernel_sizes (List[int], optional): Kernel sizes for convolutional blocks in each stage.
            Defaults to [7, 7, 7, 7].
        heads (List[int], optional): Number of attention heads for each stage. Defaults to [8, 8, 8, 8].
        use_pos_embd_xca (List[bool], optional): Whether to use positional embedding in XCA
            for each stage. Defaults to [False, False, False, False].
        use_pos_embd_global (bool, optional): Whether to use global positional embedding. Defaults to False.
        d2_scales (List[int], optional): Scale factors for split depth-wise convolution in each stage.
            Defaults to [2, 3, 4, 5].
        activation_checkpoint (bool, optional): Whether to use activation checkpointing for memory efficiency.
            Defaults to False.
        freeze_at (List[int] or str, optional): Stages to freeze. If "all", freezes entire model.
            If None, no freezing. Defaults to None.
        freeze_norm (bool, optional): Whether to freeze all normalization layers. Defaults to False.
        **kwargs: Additional keyword arguments passed to parent class.

    Raises:
        AssertionError: If global_block_type contains invalid values (not 'None' or 'SDTA').
        NotImplementedError: If unsupported global_block_type is specified.
    """

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3],
                 global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7],
                 heads=[8, 8, 8, 8],
                 use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False,
                 d2_scales=[2, 3, 4, 5],
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 **kwargs):
        """Initialize the EdgeNeXt backbone model.

        See class docstring for detailed parameter descriptions.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.dims = dims
        for g in global_block_type:
            assert g in ['None', 'SDTA']
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA':
                        stage_blocks.append(SDTAEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                        expan_ratio=expan_ratio, scales=d2_scales[i],
                                                        use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]))
                    else:
                        raise NotImplementedError(f"Global block type {global_block_type[i]} not implemented. Only SDTA is supported.")
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        self.num_features = dims[-1]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs.get("classifier_dropout", 0.0))
        if num_classes > 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def get_stage_dict(self):
        """Get a dictionary mapping stage indices to their corresponding modules.

        Returns:
            Dict[int, nn.Module]: Dictionary with stage indices (0-3) as keys and
                stage modules as values. Each stage contains multiple encoder blocks.
        """
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, layer in enumerate(self.stages, start=0):
            stage_dict[i] = layer
        return stage_dict

    def _init_weights(self, m):
        """Initialize weights for different layer types.

        Applies truncated normal initialization to Conv2d and Linear layers,
        and constant initialization to normalization layers.

        Args:
            m (nn.Module): Module to initialize.

        Note:
            TODO: MobileViT uses 'kaiming_normal' for initializing conv layers.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through backbone without classification head.

        Processes input through stem, stages, and global average pooling to produce
        feature representations before the final classification layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (N, num_features) after global average pooling.
        """
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, _, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass through backbone without classification head.

        Processes input through stem, stages, and global average pooling to produce
        feature representations before the final classification layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (N, num_features) after global average pooling.
        """
        outs = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, _, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs

    def forward(self, x):
        """Forward pass through the complete EdgeNeXt model.

        Processes input through the backbone and classification head to produce
        class predictions.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Class logits of shape (N, num_classes).
        """
        x = self.forward_pre_logits(x)
        x = self.head(self.head_dropout(x))
        return x


class EdgeNeXtBNHS(BackboneBase):
    """EdgeNeXt with Batch Normalization and Hard-Swish activation for mobile deployment.

    This is a mobile-optimized variant of EdgeNeXt that replaces Layer Normalization with
    Batch Normalization and GELU activation with Hard-Swish activation. These changes
    improve inference efficiency on mobile devices while maintaining competitive accuracy.

    Key differences from standard EdgeNeXt:
    - Uses BatchNorm2d instead of LayerNorm for all normalization layers
    - Uses Hard-Swish activation instead of GELU in encoder blocks
    - Uses SDTA_BN_HS blocks instead of standard SDTA blocks
    - Optimized for mobile inference with better hardware acceleration support

    The architecture follows the same structure as EdgeNeXt but with mobile-friendly
    components that provide faster inference on edge devices.

    Args:
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
        depths (List[int], optional): Number of blocks in each stage. Defaults to [3, 3, 9, 3].
        dims (List[int], optional): Channel dimensions for each stage.
            Defaults to [96, 192, 384, 768] (larger than standard EdgeNeXt).
        global_block (List[int], optional): Number of global blocks in each stage.
            Defaults to [0, 0, 0, 3].
        global_block_type (List[str], optional): Type of global block for each stage.
            Must be 'None' or 'SDTA_BN_HS'. Defaults to ['None', 'None', 'None', 'SDTA_BN_HS'].
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        head_init_scale (float, optional): Initial scaling for classification head. Defaults to 1.0.
        expan_ratio (int, optional): Expansion ratio for inverted bottleneck blocks. Defaults to 4.
        kernel_sizes (List[int], optional): Kernel sizes for convolutional blocks in each stage.
            Defaults to [7, 7, 7, 7].
        heads (List[int], optional): Number of attention heads for each stage.
            Defaults to [8, 8, 8, 8].
        use_pos_embd_xca (List[bool], optional): Whether to use positional embedding in XCA
            for each stage. Defaults to [False, False, False, False].
        use_pos_embd_global (bool, optional): Whether to use global positional embedding.
            Defaults to False.
        d2_scales (List[int], optional): Scale factors for split depth-wise convolution in each stage.
            Defaults to [2, 3, 4, 5].
        **kwargs: Additional keyword arguments for configuration.

    Raises:
        AssertionError: If global_block_type contains invalid values (not 'None' or 'SDTA_BN_HS').
        NotImplementedError: If unsupported global_block_type is specified.
    """

    def __init__(self, in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 global_block=[0, 0, 0, 3],
                 global_block_type=['None', 'None', 'None', 'SDTA_BN_HS'],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7],
                 heads=[8, 8, 8, 8],
                 use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False,
                 d2_scales=[2, 3, 4, 5],
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 export=False,
                 **kwargs):
        """Initialize the EdgeNeXtBNHS backbone model.

        See class docstring for detailed parameter descriptions.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )
        self.dims = dims
        for g in global_block_type:
            assert g in ['None', 'SDTA_BN_HS']

        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA_BN_HS':
                        stage_blocks.append(SDTAEncoderBNHS(dim=dims[i], drop_path=dp_rates[cur + j],
                                                            expan_ratio=expan_ratio, scales=d2_scales[i],
                                                            use_pos_emb=use_pos_embd_xca[i],
                                                            num_heads=heads[i]))
                    else:
                        raise NotImplementedError(f"Global block type {global_block_type[i]} not implemented. Only SDTA_BN_HS is supported.")
                else:
                    stage_blocks.append(ConvEncoderBNHS(dim=dims[i], drop_path=dp_rates[cur + j],
                                                        layer_scale_init_value=layer_scale_init_value,
                                                        expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        self.num_features = dims[-1]
        self.norm = nn.BatchNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(kwargs.get("classifier_dropout", 0.0))
        if num_classes > 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def get_stage_dict(self):
        """Get a dictionary mapping stage indices to their corresponding modules.

        Returns:
            Dict[int, nn.Module]: Dictionary with stage indices (0-3) as keys and
                stage modules as values. Each stage contains multiple encoder blocks.
        """
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, layer in enumerate(self.stages, start=0):
            stage_dict[i] = layer
        return stage_dict

    def _init_weights(self, m):
        """Initialize weights for different layer types.

        Applies truncated normal initialization to Conv2d and Linear layers,
        and constant initialization to normalization layers.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through backbone without classification head.

        Processes input through stem, stages, and global average pooling to produce
        feature representations before the final classification layer. Uses BatchNorm
        instead of LayerNorm for mobile optimization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature tensor of shape (N, num_features) after global average pooling.
        """
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, _, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x).mean([-2, -1])

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass to extract multi-scale feature maps.

        This method should return feature maps from multiple stages for tasks
        like object detection and segmentation that require multi-scale features.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature maps from different stages.

        Note:
            This method is not yet implemented and needs to be completed for
            multi-scale feature extraction tasks.
        """
        pass

    def forward(self, x):
        """Forward pass through the complete EdgeNeXt-BNHS model.

        Processes input through the mobile-optimized backbone (with BatchNorm and Hard-Swish)
        and classification head to produce class predictions.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Class logits of shape (N, num_classes).
        """
        x = self.forward_pre_logits(x)
        x = self.head(self.head_dropout(x))
        return x


@BACKBONE_REGISTRY.register()
def edgenext_xx_small(**kwargs):
    """Create EdgeNeXt XX-Small model variant.

    This is the smallest EdgeNeXt variant optimized for extremely resource-constrained
    environments while maintaining competitive accuracy.

    Model Specifications:
        - Parameters: 1.33M
        - FLOPs: 260.58M @ 256x256 resolution
        - Top-1 Accuracy: 71.23%

    Training Configuration:
        - No AutoAugment, Color Jitter=0.4
        - No Mixup & Cutmix
        - DropPath=0.0, Batch Size=4096, Learning Rate=0.006
        - Multi-scale sampler

    Performance Benchmarks:
        - Jetson FPS: 51.66 (vs 47.67 for MobileViT_XXS)
        - A100 FPS @ BS=1: 212.13 (vs 96.68 for MobileViT_XXS)
        - A100 FPS @ BS=256: 7042.06 (vs 4624.71 for MobileViT_XXS)

    Architecture:
        - Depths: [2, 2, 6, 2]
        - Dimensions: [24, 48, 88, 168]
        - Global blocks: [0, 1, 1, 1] with SDTA attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXt constructor.

    Returns:
        EdgeNeXt: Configured EdgeNeXt XX-Small model instance.
    """
    model = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_x_small(**kwargs):
    """Create EdgeNeXt X-Small model variant.

    A balanced EdgeNeXt variant offering good accuracy-efficiency trade-off for
    mobile and edge deployment scenarios.

    Model Specifications:
        - Parameters: 2.34M
        - FLOPs: 538.0M @ 256x256 resolution
        - Top-1 Accuracy: 75.00%

    Training Configuration:
        - No AutoAugment, No Mixup & Cutmix
        - DropPath=0.0, Batch Size=4096, Learning Rate=0.006
        - Multi-scale sampler

    Performance Benchmarks:
        - Jetson FPS: 31.61 (vs 28.49 for MobileViT_XS)
        - A100 FPS @ BS=1: 179.55 (vs 94.55 for MobileViT_XS)
        - A100 FPS @ BS=256: 4404.95 (vs 2361.53 for MobileViT_XS)

    Architecture:
        - Depths: [3, 3, 9, 3]
        - Dimensions: [32, 64, 100, 192]
        - Global blocks: [0, 1, 1, 1] with SDTA attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXt constructor.

    Returns:
        EdgeNeXt: Configured EdgeNeXt X-Small model instance.
    """
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     heads=[4, 4, 4, 4],
                     d2_scales=[2, 2, 3, 4],
                     **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_small(**kwargs):
    """Create EdgeNeXt Small model variant.

    A medium-capacity EdgeNeXt variant providing higher accuracy while maintaining
    reasonable computational requirements for mobile deployment.

    Model Specifications:
        - Parameters: 5.59M
        - FLOPs: 1260.59M @ 256x256 resolution
        - Top-1 Accuracy: 79.43%

    Training Configuration:
        - AutoAugment=True, No Mixup & Cutmix
        - DropPath=0.1, Batch Size=4096, Learning Rate=0.006
        - Multi-scale sampler

    Performance Benchmarks:
        - Jetson FPS: 20.47 (vs 18.86 for MobileViT_S)
        - A100 FPS @ BS=1: 172.33 (vs 93.84 for MobileViT_S)
        - A100 FPS @ BS=256: 3010.25 (vs 1785.92 for MobileViT_S)

    Architecture:
        - Depths: [3, 3, 9, 3]
        - Dimensions: [48, 96, 160, 304]
        - Global blocks: [0, 1, 1, 1] with SDTA attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXt constructor.

    Returns:
        EdgeNeXt: Configured EdgeNeXt Small model instance.
    """
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_base(**kwargs):
    """Create EdgeNeXt Base model variant.

    The largest EdgeNeXt variant offering the highest accuracy, suitable for
    applications where computational resources are less constrained.

    Model Specifications:
        - Parameters: 18.51M
        - FLOPs: 3840.93M @ 256x256 resolution
        - Top-1 Accuracy: 82.5% (normal), 83.7% (with USI)

    Training Configuration:
        - AutoAugment=True, Mixup & Cutmix enabled
        - DropPath=0.1, Batch Size=4096, Learning Rate=0.006
        - Multi-scale sampler

    Performance Benchmarks:
        - Jetson FPS: Not yet benchmarked
        - A100 FPS: Not yet benchmarked

    Architecture:
        - Depths: [3, 3, 9, 3]
        - Dimensions: [80, 160, 288, 584]
        - Global blocks: [0, 1, 1, 1] with SDTA attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXt constructor.

    Returns:
        EdgeNeXt: Configured EdgeNeXt Base model instance.
    """
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_xx_small_bn_hs(**kwargs):
    """Create EdgeNeXt XX-Small model with Batch Normalization and Hard-Swish activation.

    This is the smallest mobile-optimized EdgeNeXt variant using BatchNorm and Hard-Swish
    for improved inference efficiency on mobile devices and edge hardware.

    Model Specifications:
        - Parameters: 1.33M
        - FLOPs: 259.53M @ 256x256 resolution
        - Top-1 Accuracy: 70.33%

    Performance Benchmarks:
        - A100 FPS @ BS=1: 219.66
        - A100 FPS @ BS=256: 10359.98

    Mobile Optimizations:
        - BatchNorm2d instead of LayerNorm for better hardware acceleration
        - Hard-Swish activation instead of GELU for mobile efficiency
        - SDTA_BN_HS blocks for attention mechanisms

    Architecture:
        - Depths: [2, 2, 6, 2]
        - Dimensions: [24, 48, 88, 168]
        - Global blocks: [0, 1, 1, 1] with SDTA_BN_HS attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXtBNHS constructor.

    Returns:
        EdgeNeXtBNHS: Configured EdgeNeXt XX-Small BNHS model instance.
    """
    model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_x_small_bn_hs(**kwargs):
    """Create EdgeNeXt X-Small model with Batch Normalization and Hard-Swish activation.

    Mobile-optimized X-Small EdgeNeXt variant providing balanced accuracy and efficiency
    with BatchNorm and Hard-Swish for better mobile hardware acceleration.

    Model Specifications:
        - Parameters: 2.34M
        - FLOPs: 535.84M @ 256x256 resolution
        - Top-1 Accuracy: 74.87%

    Performance Benchmarks:
        - A100 FPS @ BS=1: 179.25
        - A100 FPS @ BS=256: 6059.59

    Mobile Optimizations:
        - BatchNorm2d instead of LayerNorm for better hardware acceleration
        - Hard-Swish activation instead of GELU for mobile efficiency
        - SDTA_BN_HS blocks for attention mechanisms

    Architecture:
        - Depths: [3, 3, 9, 3]
        - Dimensions: [32, 64, 100, 192]
        - Global blocks: [0, 1, 1, 1] with SDTA_BN_HS attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXtBNHS constructor.

    Returns:
        EdgeNeXtBNHS: Configured EdgeNeXt X-Small BNHS model instance.
    """
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@BACKBONE_REGISTRY.register()
def edgenext_small_bn_hs(**kwargs):
    """Create EdgeNeXt Small model with Batch Normalization and Hard-Swish activation.

    Mobile-optimized Small EdgeNeXt variant offering higher accuracy while maintaining
    mobile efficiency through BatchNorm and Hard-Swish optimizations.

    Model Specifications:
        - Parameters: 5.58M
        - FLOPs: 1257.28M @ 256x256 resolution
        - Top-1 Accuracy: 78.39%

    Performance Benchmarks:
        - A100 FPS @ BS=1: 174.68
        - A100 FPS @ BS=256: 3808.19

    Mobile Optimizations:
        - BatchNorm2d instead of LayerNorm for better hardware acceleration
        - Hard-Swish activation instead of GELU for mobile efficiency
        - SDTA_BN_HS blocks for attention mechanisms

    Architecture:
        - Depths: [3, 3, 9, 3]
        - Dimensions: [48, 96, 160, 304]
        - Global blocks: [0, 1, 1, 1] with SDTA_BN_HS attention
        - Kernel sizes: [3, 5, 7, 9]
        - Attention heads: [4, 4, 4, 4]

    Args:
        **kwargs: Additional keyword arguments passed to EdgeNeXtBNHS constructor.

    Returns:
        EdgeNeXtBNHS: Configured EdgeNeXt Small BNHS model instance.
    """
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model
