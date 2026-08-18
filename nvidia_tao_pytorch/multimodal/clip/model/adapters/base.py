# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base adapter class for CLIP-compatible model training.

This module provides the abstract base class for all CLIP-compatible model
adapters, defining the common interface and shared functionality for
vision-language models.

Classes:
    BaseCLIPAdapter: Abstract base class for CLIP-compatible model adapters
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tabulate import tabulate

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.model.logit_calibration import (
    DEFAULT_MAX_LOGIT_SCALE,
    default_logit_calibration,
    named_logit_parameters,
    scalar_override,
    validate_logit_parameter,
)


class BaseCLIPAdapter(nn.Module, ABC):
    """Abstract base class for CLIP-compatible model adapters.

    This class defines the common interface for all vision-language model
    adapters used in CLIP training. Concrete implementations must provide
    encode_image and encode_text methods.

    Args:
        logit_scale_init: Adapter-owned fallback raw scale override.
        logit_bias_init: Adapter-owned fallback bias override.
        loss_type: Loss family used to select missing-value fallbacks.
        owns_logit_parameters: Whether this adapter owns calibration locally.

    Attributes:
        tokenizer: Tokenizer for text encoding (set by subclasses)
    """

    logit_scale_max = DEFAULT_MAX_LOGIT_SCALE

    def __init__(
        self,
        logit_scale_init=None,
        logit_bias_init=None,
        loss_type='siglip',
        owns_logit_parameters=True,
    ):
        """Initialize shared state and optional local calibration."""
        super().__init__()
        self.tokenizer = None
        self.logit_scale_max = DEFAULT_MAX_LOGIT_SCALE
        if owns_logit_parameters:
            self._create_local_logit_parameters(
                logit_scale_init=logit_scale_init,
                logit_bias_init=logit_bias_init,
                loss_type=loss_type,
            )

    def _create_local_logit_parameters(
        self, logit_scale_init=None, logit_bias_init=None, loss_type='siglip'
    ):
        """Create one adapter-owned fallback calibration pair."""
        fallback_scale, fallback_bias = default_logit_calibration(loss_type)
        scale = fallback_scale if logit_scale_init is None else logit_scale_init
        bias = fallback_bias if logit_bias_init is None else logit_bias_init
        self.logit_scale = nn.Parameter(torch.tensor(scalar_override(
            scale, 'init_logit_scale'
        )))
        self.logit_bias = nn.Parameter(torch.tensor(scalar_override(
            bias, 'init_logit_bias'
        )))

    @staticmethod
    def _format_params(n: int) -> str:
        """Format parameter count with M/B suffix.

        Args:
            n: Number of parameters.

        Returns:
            Formatted string (e.g., "1.2B", "304.5M", "1,234").
        """
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        return f"{n:,}"

    def _log_model_summary(
        self,
        model_name: str,
        vision_total: int,
        vision_trainable: int,
        text_total: int,
        text_trainable: int,
        freeze_vision: bool,
        freeze_text: bool,
    ):
        """Log a formatted model parameter summary table.

        Args:
            model_name: Name of the model to display in the header.
            vision_total: Total vision encoder parameters.
            vision_trainable: Trainable vision encoder parameters.
            text_total: Total text encoder parameters.
            text_trainable: Trainable text encoder parameters.
            freeze_vision: Whether vision encoder is frozen.
            freeze_text: Whether text encoder is frozen.
        """
        fmt = self._format_params
        logit_named_params = list(self.named_logit_parameters())
        logit_params = sum(param.numel() for _, param in logit_named_params)
        logit_trainable = sum(
            param.numel() for _, param in logit_named_params if param.requires_grad
        )
        total_params = vision_total + text_total + logit_params
        total_trainable = vision_trainable + text_trainable + logit_trainable

        vision_status = "frozen" if freeze_vision else "trainable"
        text_status = "frozen" if freeze_text else "trainable"

        table_data = [
            ["Vision encoder", fmt(vision_trainable), fmt(vision_total), vision_status],
            ["Text encoder", fmt(text_trainable), fmt(text_total), text_status],
            ["Logit params", fmt(logit_trainable), fmt(logit_params), "trainable"],
            ["Total", fmt(total_trainable), fmt(total_params), ""],
        ]
        headers = ["Component", "Trainable", "Total", "Status"]
        table = tabulate(table_data, headers=headers, tablefmt="simple")

        scale = self.get_logit_scale_parameter()
        bias = self.get_logit_bias_parameter()
        bias_info = (
            f"{bias.item():.2f}" if bias is not None else "not present"
        )
        logit_info = (
            f"Logit scale: {scale.exp().item():.2f}, "
            f"Logit bias: {bias_info}"
        )
        logging.info(f"{model_name}\n{table}\n{logit_info}")

    def get_encoder_blocks(self, tower: str):
        """Return transformer blocks in forward order for LoRA injection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encoder blocks for {tower!r}."
        )

    @abstractmethod
    def vision_named_parameters(self):
        """Return named parameters for the vision encoder.

        Used by the optimizer to create per-tower parameter groups.

        Returns:
            Iterator of (name, parameter) tuples for vision encoder params.
        """
        pass

    @abstractmethod
    def text_named_parameters(self):
        """Return named parameters for the text encoder.

        Used by the optimizer to create per-tower parameter groups.

        Returns:
            Iterator of (name, parameter) tuples for text encoder params.
        """
        pass

    def get_logit_scale_parameter(self):
        """Return the canonical raw logit-scale parameter."""
        return validate_logit_parameter(
            getattr(self, 'logit_scale', None),
            'logit_scale',
            required=True,
        )

    def get_logit_bias_parameter(self):
        """Return the optional canonical logit-bias parameter."""
        return validate_logit_parameter(
            getattr(self, 'logit_bias', None),
            'logit_bias',
            required=False,
        )

    def named_logit_parameters(self):
        """Yield canonical calibration parameters in stable optimizer order."""
        yield from named_logit_parameters(self)

    def other_named_parameters(self):
        """Return named parameters not belonging to either tower.

        Kept as a compatibility alias for existing optimizer callers.

        Returns:
            Iterator of (name, parameter) tuples.
        """
        yield from self.named_logit_parameters()

    def clamp_logit_scale_(self):
        """Apply this model family's raw logit-scale bounds in place."""
        scale = self.get_logit_scale_parameter()
        with torch.no_grad():
            scale.clamp_(min=0, max=self.logit_scale_max)

    @abstractmethod
    def encode_image(
        self, image, normalize: bool = True
    ) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            image: Input image tensor or dict (for HF processor outputs).
            normalize: Whether to L2-normalize output features. Default: True.

        Returns:
            Image feature tensor of shape (B, D).
        """
        pass

    @abstractmethod
    def encode_text(
        self, text, normalize: bool = True
    ) -> torch.Tensor:
        """Encode text to feature vectors.

        Args:
            text: Tokenized text dict with 'input_ids' and 'attention_mask'.
            normalize: Whether to L2-normalize output features. Default: True.

        Returns:
            Text feature tensor of shape (B, D).
        """
        pass

    def forward(self, image=None, text=None):
        """Forward pass supporting both image and text inputs.

        Args:
            image: Input image tensor or dict. Optional.
            text: Tokenized text dict. Optional.

        Returns:
            If both image and text:
                Tuple of image features, text features, exponentiated scalar
                logit scale, and optional scalar logit bias.
            If only image:
                Dict with 'image_features' key.
            If only text:
                Dict with 'text_features' key.

        Raises:
            ValueError: If neither image nor text is provided.
        """
        if image is not None and text is not None:
            image_features = self.encode_image(image, normalize=True)
            text_features = self.encode_text(text, normalize=True)
            outputs = (
                image_features,
                text_features,
                self.get_logit_scale_parameter().exp().reshape(()),
            )
            logit_bias = self.get_logit_bias_parameter()
            if logit_bias is not None:
                outputs += (logit_bias.reshape(()),)
            return outputs
        elif image is not None:
            image_features = self.encode_image(image, normalize=True)
            return {"image_features": image_features}
        elif text is not None:
            text_features = self.encode_text(text, normalize=True)
            return {"text_features": text_features}
        else:
            raise ValueError("Either image or text must be provided")

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency.

        Override in subclasses if the underlying model supports
        checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing. Default: True.
        """
        logging.warning(
            "%s does not implement set_grad_checkpointing; "
            "gradient checkpointing will have no effect.",
            self.__class__.__name__,
        )
