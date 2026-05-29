# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP2 model adapter for CLIP-compatible training.

Supports Google's SigLIP2 models with both vision and text encoders:
- siglip2-giant-opt-patch16-384 (SigLIP2-G, 384x384 images)
- siglip2-so400m-patch16-naflex (SigLIP2-SO400M, flexible resolution)
"""

import math

import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    SigLIP2WrappedTokenizer,
    CLIPCompatibleTokenizer,
)


class SigLIP2(BaseCLIPAdapter):
    """Adapter to make backbone_v2.siglip2 compatible with CLIP training.

    This wraps the backbone_v2 SigLIP2Wrapper to provide the same interface
    as the CLIP SigLIP2Wrapper for training compatibility.

    Args:
        backbone_model: The backbone_v2 SigLIP2 model
        processor: HuggingFace processor for tokenization
        logit_scale_init: Initial value for logit scale parameter
        logit_bias_init: Initial value for logit bias parameter
        freeze_vision_encoder: Freeze vision encoder parameters
        freeze_text_encoder: Freeze text encoder parameters
        canonicalize_text: Apply text canonicalization before tokenization
    """

    def __init__(
        self,
        backbone_model,
        processor,
        logit_scale_init=2.3026,
        logit_bias_init=-10.0,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
        canonicalize_text=False,
    ):
        """Initialize SigLIP2 adapter."""
        super().__init__(
            logit_scale_init=logit_scale_init,
            logit_bias_init=logit_bias_init,
        )

        self.backbone = backbone_model
        self.processor = processor
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_text_encoder = freeze_text_encoder

        # Create wrapped tokenizer using shared utilities
        self._siglip2_tokenizer = SigLIP2WrappedTokenizer(
            processor, canonicalize=canonicalize_text
        )
        self.tokenizer = CLIPCompatibleTokenizer(self._siglip2_tokenizer)

        # Configure trainable parameters
        self._configure_trainable_params()

        # Log parameters
        self._log_parameters()

    def _configure_trainable_params(self):
        """Configure trainable params based on freeze settings."""
        # Warning if both encoders are frozen
        if self.freeze_vision_encoder and self.freeze_text_encoder:
            logging.warning(
                "Both vision and text encoders are frozen. "
                "Only logit_scale and logit_bias will be trained."
            )

        # Freeze vision encoder if requested
        if self.freeze_vision_encoder:
            for param in self.backbone.inner.vision_model.parameters():
                param.requires_grad = False
            self.backbone.inner.vision_model.eval()

        # Freeze text encoder if requested
        if self.freeze_text_encoder:
            for param in self.backbone.inner.text_model.parameters():
                param.requires_grad = False
            self.backbone.inner.text_model.eval()

    def _log_parameters(self):
        """Log parameter configuration summary."""
        vision_total = sum(
            p.numel() for p in self.backbone.inner.vision_model.parameters()
        )
        vision_trainable = sum(
            p.numel() for p in self.backbone.inner.vision_model.parameters()
            if p.requires_grad
        )
        text_total = sum(
            p.numel() for p in self.backbone.inner.text_model.parameters()
        )
        text_trainable = sum(
            p.numel() for p in self.backbone.inner.text_model.parameters()
            if p.requires_grad
        )
        self._log_model_summary(
            model_name="SigLIP2 Model",
            vision_total=vision_total,
            vision_trainable=vision_trainable,
            text_total=text_total,
            text_trainable=text_trainable,
            freeze_vision=self.freeze_vision_encoder,
            freeze_text=self.freeze_text_encoder,
        )

    def vision_named_parameters(self):
        """Return named parameters for the vision encoder."""
        prefix = 'backbone.inner.vision_model'
        for name, param in self.backbone.inner.vision_model.named_parameters():
            yield f'{prefix}.{name}', param

    def text_named_parameters(self):
        """Return named parameters for the text encoder."""
        prefix = 'backbone.inner.text_model'
        for name, param in self.backbone.inner.text_model.named_parameters():
            yield f'{prefix}.{name}', param

    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing for memory efficiency."""
        vision = self.backbone.inner.vision_model
        if hasattr(vision, 'gradient_checkpointing_enable'):
            if enable:
                vision.gradient_checkpointing_enable()
            else:
                vision.gradient_checkpointing_disable()
        text = self.backbone.inner.text_model
        if hasattr(text, 'gradient_checkpointing_enable'):
            if enable:
                text.gradient_checkpointing_enable()
            else:
                text.gradient_checkpointing_disable()

    def encode_image(self, image, normalize=True):
        """Encode images using backbone_v2 SigLIP2.

        Handles both:
        1. 4D tensor [B, C, H, W] - standard image format
        2. 3D tensor [B, num_patches, patch_dim] - NaFlex pre-flattened
           patches from HF processor
        3. Dict with 'pixel_values' and optional 'pixel_attention_mask',
           'spatial_shapes'
        """
        device = next(self.backbone.parameters()).device

        # Handle dict input (from HF processor)
        if isinstance(image, dict):
            pixel_values = image['pixel_values'].to(device)
            # Get optional NaFlex-specific inputs
            pixel_attention_mask = image.get('pixel_attention_mask')
            if pixel_attention_mask is not None:
                pixel_attention_mask = pixel_attention_mask.to(device)
            spatial_shapes = image.get('spatial_shapes')
            if spatial_shapes is not None:
                spatial_shapes = spatial_shapes.to(device)
        else:
            pixel_values = image.to(device)
            pixel_attention_mask = None
            spatial_shapes = None

        # Check if it's NaFlex format (3D) or standard format (4D)
        if pixel_values.dim() == 3:
            # NaFlex format: [B, num_patches, patch_dim]
            # Call inner model's vision_model directly since backbone's
            # forward expects 4D input for dynamic mode
            batch_size = pixel_values.shape[0]
            num_patches = pixel_values.shape[1]

            # Create attention mask if not provided (int32 to match HF)
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    (batch_size, num_patches),
                    dtype=torch.int32,
                    device=device
                )

            # Use spatial_shapes from processor if available, otherwise compute
            if spatial_shapes is None:
                side = int(math.sqrt(num_patches))
                spatial_shapes = torch.tensor(
                    [[side, side]] * batch_size,
                    dtype=torch.int64,
                    device=device
                )

            output = self.backbone.inner.vision_model(
                pixel_values=pixel_values,
                attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            # output is BaseModelOutputWithPooling - has .pooler_output
            image_features = output.pooler_output
        else:
            # Standard 4D format - use backbone's forward
            image_features = self.backbone.forward(
                pixel_values, return_features=False, return_logits=True
            )

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(self, text, normalize=True):
        """Encode text using backbone_v2 SigLIP2."""
        device = next(self.backbone.parameters()).device
        text = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in text.items()
        }

        text_features = self.backbone.encode_text(text, normalize=False)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features
