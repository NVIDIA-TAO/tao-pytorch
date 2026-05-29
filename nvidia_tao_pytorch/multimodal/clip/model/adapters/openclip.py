# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCLIP model adapter for CLIP-compatible training.

Supports OpenCLIP/NV-CLIP models with both vision and text encoders:
- ViT-L-14-SigLIP-CLIPA-224
- ViT-L-14-SigLIP-CLIPA-336
- ViT-H-14-SigLIP-CLIPA-224
- And other OpenCLIP models
"""

import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    OpenCLIPWrappedTokenizer,
    CLIPCompatibleTokenizer,
)


class OpenCLIP(BaseCLIPAdapter):
    """Adapter to make backbone_v2.open_clip compatible with CLIP training.

    This wraps the backbone_v2 OpenCLIP model to provide the same
    interface as other CLIP adapters for training compatibility.

    Args:
        backbone_model: The backbone_v2 OpenCLIP model
        logit_scale_init: Initial value for logit scale parameter
        logit_bias_init: Initial value for logit bias parameter
        freeze_vision_encoder: Freeze vision encoder parameters
        freeze_text_encoder: Freeze text encoder parameters
        canonicalize_text: Apply text canonicalization before tokenization
    """

    def __init__(
        self,
        backbone_model,
        logit_scale_init=2.3026,
        logit_bias_init=-10.0,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
        canonicalize_text=False,
    ):
        """Initialize OpenCLIP adapter."""
        super().__init__(
            logit_scale_init=logit_scale_init,
            logit_bias_init=logit_bias_init,
        )

        self.backbone = backbone_model
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_text_encoder = freeze_text_encoder

        # Create wrapped tokenizer for CLIP dataloader compatibility
        self._wrapped_tokenizer = OpenCLIPWrappedTokenizer(
            self.backbone.tokenizer, canonicalize=canonicalize_text
        )
        self.tokenizer = CLIPCompatibleTokenizer(self._wrapped_tokenizer)

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
            for param in self.backbone.model.visual.parameters():
                param.requires_grad = False
            self.backbone.model.visual.eval()

        # Freeze text encoder if requested
        if self.freeze_text_encoder:
            # OpenCLIP text encoder components
            if hasattr(self.backbone.model, 'transformer'):
                for param in self.backbone.model.transformer.parameters():
                    param.requires_grad = False
                self.backbone.model.transformer.eval()
            if hasattr(self.backbone.model, 'token_embedding'):
                self.backbone.model.token_embedding.requires_grad_(False)
            if hasattr(self.backbone.model, 'positional_embedding'):
                pe = self.backbone.model.positional_embedding
                if isinstance(pe, torch.nn.Parameter):
                    pe.requires_grad = False
            if hasattr(self.backbone.model, 'ln_final'):
                for param in self.backbone.model.ln_final.parameters():
                    param.requires_grad = False
                self.backbone.model.ln_final.eval()
            if hasattr(self.backbone.model, 'text_projection'):
                tp = self.backbone.model.text_projection
                if tp is not None:
                    if isinstance(tp, torch.nn.Parameter):
                        tp.requires_grad = False
                    else:
                        tp.requires_grad_(False)

    def _log_parameters(self):
        """Log parameter configuration summary."""
        vision_total = sum(
            p.numel() for p in self.backbone.model.visual.parameters()
        )
        vision_trainable = sum(
            p.numel() for p in self.backbone.model.visual.parameters()
            if p.requires_grad
        )
        text_total, text_trainable = self._count_text_params()

        self._log_model_summary(
            model_name=f"OpenCLIP Model: {self.backbone._model_name}",
            vision_total=vision_total,
            vision_trainable=vision_trainable,
            text_total=text_total,
            text_trainable=text_trainable,
            freeze_vision=self.freeze_vision_encoder,
            freeze_text=self.freeze_text_encoder,
        )

    def _count_text_params(self):
        """Count text encoder parameters."""
        text_params = []
        if hasattr(self.backbone.model, 'transformer'):
            text_params.extend(self.backbone.model.transformer.parameters())
        if hasattr(self.backbone.model, 'token_embedding'):
            text_params.extend(
                self.backbone.model.token_embedding.parameters()
            )
        if hasattr(self.backbone.model, 'positional_embedding'):
            pe = self.backbone.model.positional_embedding
            if isinstance(pe, torch.nn.Parameter):
                text_params.append(pe)
        if hasattr(self.backbone.model, 'ln_final'):
            text_params.extend(self.backbone.model.ln_final.parameters())
        if hasattr(self.backbone.model, 'text_projection'):
            tp = self.backbone.model.text_projection
            if tp is not None:
                if isinstance(tp, torch.nn.Parameter):
                    text_params.append(tp)
                else:
                    text_params.extend(tp.parameters())

        text_total = sum(p.numel() for p in text_params)
        text_trainable = sum(p.numel() for p in text_params if p.requires_grad)
        return text_total, text_trainable

    def vision_named_parameters(self):
        """Return named parameters for the vision encoder."""
        for name, param in self.backbone.model.visual.named_parameters():
            yield f'backbone.model.visual.{name}', param

    def text_named_parameters(self):
        """Return named parameters for the text encoder."""
        text_modules = {
            'transformer': 'backbone.model.transformer',
            'token_embedding': 'backbone.model.token_embedding',
            'ln_final': 'backbone.model.ln_final',
        }
        for attr, prefix in text_modules.items():
            module = getattr(self.backbone.model, attr, None)
            if module is not None:
                for name, param in module.named_parameters():
                    yield f'{prefix}.{name}', param
        pe = getattr(self.backbone.model, 'positional_embedding', None)
        if pe is not None and isinstance(pe, torch.nn.Parameter):
            yield 'backbone.model.positional_embedding', pe
        tp = getattr(self.backbone.model, 'text_projection', None)
        if tp is not None:
            if isinstance(tp, torch.nn.Parameter):
                yield 'backbone.model.text_projection', tp
            else:
                for name, param in tp.named_parameters():
                    yield f'backbone.model.text_projection.{name}', param

    def encode_image(self, image, normalize=True):
        """Encode images using backbone_v2 OpenCLIP.

        Args:
            image: Input image tensor of shape (B, C, H, W).
            normalize: Whether to L2-normalize the output. Default: True.

        Returns:
            Image features of shape (B, D).
        """
        device = next(self.backbone.parameters()).device
        image = image.to(device)

        # Use backbone's forward_pre_logits which calls model.encode_image
        image_features = self.backbone.forward_pre_logits(image)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(self, text, normalize=True):
        """Encode text using backbone_v2 OpenCLIP.

        Args:
            text: Tokenized text dict with 'input_ids', or tensor directly.
            normalize: Whether to L2-normalize the output. Default: True.

        Returns:
            Text features of shape (B, D).
        """
        device = next(self.backbone.parameters()).device

        # Handle dict (from wrapper) and tensor (direct) input
        if isinstance(text, dict):
            text = text['input_ids']
        text = text.to(device)

        # Use backbone's encode_text method
        text_features = self.backbone.encode_text(text, normalize=False)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            enable: Whether to enable gradient checkpointing. Default: True.
        """
        self.backbone.set_grad_checkpointing(enable)
