# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for backbone models.

This module provides the foundational infrastructure for backbone models in the TAO PyTorch framework.
It defines the common interface and functionality that all backbone models must implement.

The BackboneBase class serves as the abstract base class that provides:
- Common initialization parameters for all backbone models
- Standardized methods for feature extraction and classification
- Support for activation checkpointing and layer freezing
- Integration with the TAO backbone registry system

Key Features:
- Abstract interface for backbone model implementations
- Support for gradient checkpointing to trade compute for memory
- Layer freezing capabilities for transfer learning
- Batch normalization freezing for improved training stability
- Standardized forward pass methods for different use cases

Classes:
    BackboneMeta: Metaclass for BackboneBase that handles post-initialization
    BackboneBase: Abstract base class for all backbone models

Example:
    ```python
    class MyBackbone(BackboneBase):
        def __init__(self, in_chans=3, num_classes=1000, **kwargs):
            super().__init__(in_chans=in_chans, num_classes=num_classes, **kwargs)
            # Define your model architecture here

        def get_stage_dict(self):
            # Return dictionary mapping stage names to modules
            return {0: self.stage0, 1: self.stage1, ...}

        def forward_pre_logits(self, x):
            # Forward pass without classification head
            return features

        def forward(self, x):
            # Complete forward pass with classification
            return logits
    ```
"""
import abc
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import FrozenBatchNorm2d


class BackboneMeta(abc.ABCMeta, type):
    """Metaclass for BackboneBase that handles post-initialization.

    This metaclass ensures that the `_post_init` method is called after object creation,
    allowing for proper setup of backbone models including freezing layers and setting
    gradient checkpointing.

    The metaclass combines ABCMeta (for abstract base class functionality) with type
    to provide both abstract method enforcement and custom initialization behavior.
    """

    def __call__(cls, *args, **kwargs):
        """Called when you call `BackboneBase()` or any subclass constructor.

        This method ensures that the `_post_init` method is automatically called
        after the object is created, providing a standardized initialization flow
        for all backbone models.

        Args:
            *args: Positional arguments passed to the constructor
            **kwargs: Keyword arguments passed to the constructor

        Returns:
            BackboneBase: The initialized backbone model instance
        """
        obj = type.__call__(cls, *args, **kwargs)
        obj._post_init()  # Call `_post_init` after the object is created.
        return obj


class BackboneBase(nn.Module, metaclass=BackboneMeta):
    """Abstract base class for backbone models.

    This class defines the common interface and functionality for all backbone models.
    All backbone models should inherit from this class and implement the required methods:

    - `set_grad_checkpointing` (optional if the model inherits from Timm library)
    - `get_stage_dict`
    - `get_classifier` (optional if the model inherits from Timm library)
    - `reset_classifier` (optional if the model inherits from Timm library)
    - `forward_pre_logits`
    - `forward_feature_pyramid` (can raise `NotImplementedError` if not needed)
    - `forward`

    Examples:

        ```python
        # When using `TimmModel` as a base class.
        class MyBackbone(TimmModel, BackboneBase):
            def __init__(self, ...):
                ...
                super().__init__(...)  # `TimmModel` initialization.
                BackboneBase.__init__(
                    self,
                    in_chans=in_chans,
                    num_classes=num_classes,
                    activation_checkpoint=activation_checkpoint,
                    freeze_at=freeze_at,
                    freeze_norm=freeze_norm,
                )
        ```

        ```python
        # Not using `TimmModel` as a base class.
        class MyBackbone(BackboneBase):
            def __init__(self, ...):
                super().__init__(
                    in_chans=in_chans,
                    num_classes=num_classes,
                    activation_checkpoint=activation_checkpoint,
                    freeze_at=freeze_at,
                    freeze_norm=freeze_norm,
                )
                # Define your model architecture here.
                ...

        ```
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        activation_checkpoint: bool = False,
        freeze_at: Optional[List[Union[int, str]]] = None,
        freeze_norm: bool = False,
        export: bool = False,
    ):
        """Initialize the backbone base class.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN
        """
        if not self._module_is_initialized:
            nn.Module.__init__(self)

        freeze_at = freeze_at or []
        if isinstance(freeze_at, str):
            if freeze_at != "all":
                raise ValueError(f"If freeze_at is a string, it should be 'all'. Received: {freeze_at}.")
        else:
            if not isinstance(freeze_at, list):
                freeze_at = list(freeze_at)
            if not all(isinstance(i, (int, str)) for i in freeze_at):
                raise ValueError(f"Invalid freeze_at value: {freeze_at}. It should be a list of integers or strings.")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.freeze_norm = bool(freeze_norm)
        self.freeze_at = freeze_at
        self.export = export

    @property
    def _module_is_initialized(self):
        """Whether the nn.Module is initialized.

        This property checks if the nn.Module has been properly initialized by
        checking for the presence of standard nn.Module attributes.

        Returns:
            bool: True if the module is initialized, False otherwise
        """
        if hasattr(self, "_parameters") or hasattr(self, "_buffers") or hasattr(self, "_modules"):
            return True
        return False

    def _post_init(self):
        """Post-initialization method.

        This method is called after the module is initialized and handles:
        - Freezing specified layers of the backbone
        - Setting up gradient checkpointing for memory efficiency

        This method is automatically called by the BackboneMeta metaclass
        after object creation.
        """
        self.freeze_backbone()
        if self.export:
            self._freeze_bn_norm(self)
        self.set_grad_checkpointing(self.activation_checkpoint)

    def _init_weights(self, m):
        """Initialize weights for different layer types.

        Applies truncated normal initialization to Conv2d and Linear layers,
        and zero initialization to bias terms.

        Args:
            m (nn.Module): Module to initialize weights for
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Set the gradient (activation) checkpointing for the model.

        In short, this technique allows you to trade compute for memory. It saves memory by not storing intermediate
        activations. Please refer to https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.

        Note that some Timm models (such as Hiera, ResNet and ViT) have already implemented this method. You can safely
        call this method on those models.

        Args:
            enable (bool): If `True`, enable gradient (activation) checkpointing. Default: `True`.
        """
        self.activation_checkpoint = enable

    @abc.abstractmethod
    def get_stage_dict(self) -> Dict[Union[int, str], nn.Module]:
        """Get the stage dictionary.

        This method is useful for freezing specific stages of the backbone.

        Returns:
            dict: Dictionary containing the stages of the backbone. The keys are
                the stage names (e.g. the index of the stage) and the values are
                the corresponding modules.
        """
        pass

    def no_weight_decay(self) -> Set[str]:
        """Get the set of parameter names to exclude from weight decay.

        This method returns parameter names that should be excluded from weight decay
        during training. This is commonly used for bias terms and normalization
        parameters.

        Returns:
            Set[str]: Set of parameter names to exclude from weight decay
        """
        return set()

    def no_weight_decay_keywords(self) -> Set[str]:
        """Get the set of parameter keywords to exclude from weight decay.

        This method returns parameter name keywords that should be excluded from
        weight decay. This is useful for excluding parameters based on naming patterns.

        Returns:
            Set[str]: Set of parameter keywords to exclude from weight decay
        """
        return set()

    @abc.abstractmethod
    def get_classifier(self) -> nn.Module:
        """Get the classifier module.

        Note that some Timm models (such as Hiera, ResNet and ViT) have already implemented this method. You can safely
        call this method on those models.

        Returns:
            nn.Module: The classifier module
        """
        pass

    @abc.abstractmethod
    def reset_classifier(self, num_classes: int = 0, **kwargs):
        """Reset the classifier head.

        Note that some Timm models (such as Hiera, ResNet and ViT) have already implemented this method. You can safely
        call this method on those models.

        Args:
            num_classes (int): Number of classes for the new classifier. Default: `0`.
            **kwargs: Additional arguments passed to the classifier implementation.
        """
        pass

    def _freeze_module(self, m: nn.Module):
        """Freeze the given module.

        Sets all parameters in the module to not require gradients and sets
        the module to evaluation mode.

        Args:
            m (nn.Module): Module to freeze
        """
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    def _freeze_bn_norm(self, m: nn.Module):
        """Recursively freeze the batch normalization layers in the given module.

        Replaces all BatchNorm2d layers with FrozenBatchNorm2d to prevent
        their statistics from being updated during training.

        Args:
            m (nn.Module): Module to process

        Returns:
            nn.Module: Module with frozen batch normalization layers
        """
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_bn_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def freeze_backbone(self):
        """Freeze specific parts of the backbone and batch normalization layers.

        This method used `self.freeze_at` and `self.freeze_norm` to determine
        which parts of the backbone to freeze.

        The `get_stage_dict` method must be implemented in the subclass to
        provide the mapping of stage keys to modules.
        """
        if self.freeze_norm:
            self._freeze_bn_norm(self)
            if get_global_rank() == 0:
                logging.warning("All batch normalization layers are frozen.")

        if self.freeze_at == "all":
            self._freeze_module(self)
            if get_global_rank() == 0:
                logging.warning("The backbone is frozen.")
        elif isinstance(self.freeze_at, list) and len(self.freeze_at) != 0:
            stage_dict = self.get_stage_dict()
            for key in self.freeze_at:
                if key in stage_dict:
                    self._freeze_module(stage_dict[key])
                    if get_global_rank() == 0:
                        logging.warning(f"Stage {key} is frozen.")
                else:
                    if get_global_rank() == 0:
                        logging.warning(f"Stage {key} not found. Freezing options: {stage_dict.keys()}")

    @abc.abstractmethod
    def forward_pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output pre-logits tensor.
        """
        pass

    @abc.abstractmethod
    def forward_feature_pyramid(
        self, x: torch.Tensor, indices: Optional[Union[int, List[int]]] = None, **kwargs
    ) -> List[torch.Tensor]:
        """Forward pass through the backbone to extract intermediate feature maps.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            **kwargs: Additional arguments.

        Returns:
            List[torch.Tensor]: List of intermediate feature maps.
        """
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor.
        """
        pass
