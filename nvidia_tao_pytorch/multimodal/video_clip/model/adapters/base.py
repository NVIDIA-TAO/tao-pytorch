# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class BaseCLIPAdapter(nn.Module, ABC):
    """Abstract base class for CLIP-compatible model adapters.

    This class defines the common interface for all vision-language model
    adapters used in CLIP training. Concrete implementations must provide
    encode_image and encode_text methods.

    Args:
        logit_scale_init: Initial value for logit scale. Default: 2.3026
        logit_bias_init: Initial value for logit bias parameter. Default: -10.0

    Attributes:
        logit_scale: Learnable temperature parameter for contrastive loss
        logit_bias: Learnable bias parameter for SigLIP loss
        tokenizer: Tokenizer for text encoding (set by subclasses)
    """

    def __init__(
        self,
        logit_scale_init: float = 2.3026,
        logit_bias_init: float = -10.0,
    ):
        """Initialize the base adapter with logit scale and bias parameters."""
        super().__init__()

        # Initialize logit_scale and logit_bias for SigLIP/CLIP loss
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_init)

        # Tokenizer should be set by subclasses
        self.tokenizer = None

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

    @staticmethod
    def _param_status(trainable: int, total: int) -> str:
        """Status from actual counts so it can't contradict the numbers.

        'frozen' when nothing trains, 'trainable' when everything trains, and
        'partial' when only part of a component trains (e.g. a frozen backbone
        with a trainable alignment/projection head).
        """
        if trainable <= 0:
            return "frozen"
        if trainable >= total:
            return "trainable"
        return "partial"

    @staticmethod
    def _count_params(params):
        """Return ``(trainable, total)`` parameter counts for an iterable.

        Accepts either ``module.parameters()`` or the parameter values of
        ``module.named_parameters()`` (pass ``(p for _, p in named)``).
        """
        params = list(params)
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        return trainable, total

    def _warn_if_fully_frozen(self, freeze_vision, freeze_text):
        """Warn when both encoders are frozen (only the logits would train)."""
        if freeze_vision and freeze_text:
            logging.warning(
                "Both vision and text encoders are frozen. "
                "Only logit_scale and logit_bias will be trained."
            )

    def _log_model_summary(
        self,
        model_name: str,
        vision_total: int,
        vision_trainable: int,
        text_total: int,
        text_trainable: int,
        freeze_vision: bool,
        freeze_text: bool,
        vision_subrows=None,
    ):
        """Log a formatted model parameter summary table.

        Args:
            model_name: Name of the model to display in the header.
            vision_total: Total vision encoder parameters.
            vision_trainable: Trainable vision encoder parameters.
            text_total: Total text encoder parameters.
            text_trainable: Trainable text encoder parameters.
            freeze_vision: Whether vision encoder is frozen (kept for API
                compatibility; the Status column is derived from the actual
                trainable/total counts, not this flag).
            freeze_text: Whether text encoder is frozen (see freeze_vision).
            vision_subrows: Optional list of ``(label, trainable, total)`` rows
                that replace the single "Vision encoder" row. Lets a model split
                the vision tower into, e.g., a frozen backbone and a trainable
                alignment head so the report is not misleading.
        """
        del freeze_vision, freeze_text  # status now derived from counts
        fmt = self._format_params
        status = self._param_status
        logit_params = self.logit_scale.numel() + self.logit_bias.numel()
        total_params = vision_total + text_total + logit_params
        total_trainable = vision_trainable + text_trainable + logit_params

        if vision_subrows:
            vision_rows = [
                [label, fmt(tr), fmt(tot), status(tr, tot)]
                for label, tr, tot in vision_subrows
            ]
        else:
            vision_rows = [
                ["Vision encoder", fmt(vision_trainable), fmt(vision_total),
                 status(vision_trainable, vision_total)],
            ]

        table_data = [
            *vision_rows,
            ["Text encoder", fmt(text_trainable), fmt(text_total), status(text_trainable, text_total)],
            ["Logit params", fmt(logit_params), fmt(logit_params), "trainable"],
            ["Total", fmt(total_trainable), fmt(total_params), ""],
        ]
        headers = ["Component", "Trainable", "Total", "Status"]
        table = tabulate(table_data, headers=headers, tablefmt="simple")

        logit_info = (
            f"Logit scale: {self.logit_scale.exp().detach().item():.2f}, "
            f"Logit bias: {self.logit_bias.detach().item():.2f}"
        )
        logging.info(f"{model_name}\n{table}\n{logit_info}")

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

    def other_named_parameters(self):
        """Return named parameters not belonging to either tower.

        By default returns logit_scale and logit_bias. Override if needed.

        Returns:
            Iterator of (name, parameter) tuples.
        """
        yield 'logit_scale', self.logit_scale
        yield 'logit_bias', self.logit_bias

    def get_encoder_blocks(self, tower: str):
        """Return an ordered list of transformer blocks for a given tower.

        Used by LoRA injection (see ``multimodal.clip.model.lora.inject_lora``)
        to identify the last N blocks of a tower to adapt.

        Args:
            tower: 'vision' or 'text'.

        Returns:
            List of nn.Module, one per transformer block, in forward order.

        Raises:
            ValueError: If tower is not 'vision' or 'text'.
            NotImplementedError: If the subclass does not support this tower.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            f"get_encoder_blocks for tower='{tower}'."
        )

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
                Tuple of (image_features, text_features, logit_scale,
                logit_bias)
            If only image:
                Dict with 'image_features' key
            If only text:
                Dict with 'text_features' key

        Raises:
            ValueError: If neither image nor text is provided.
        """
        if image is not None and text is not None:
            image_features = self.encode_image(image, normalize=True)
            text_features = self.encode_text(text, normalize=True)
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
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
