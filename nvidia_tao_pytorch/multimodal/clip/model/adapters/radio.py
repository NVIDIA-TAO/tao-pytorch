# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C-RADIO model adapter for CLIP-compatible training.

Uses torch.hub C-RADIO model with built-in text adaptors for both vision
and text encoding.

Based on: https://github.com/NVlabs/RADIO

Supported models (Commercial License):
  - C-RADIOv3: c-radio_v3-h, c-radio_v3-l, c-radio_v3-b, c-radio_v3-g

Pre-aligned adaptors:
  - 'siglip2' (SigLIP2 text encoder)
  - 'clip' (DFN CLIP text encoder)

NOTE: This implementation uses torch.hub because the text adaptors are
only available via torch.hub, not on HuggingFace.
"""

import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.logit_calibration import (
    configure_source_logit_calibration,
)
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    CLIPCompatibleTokenizer,
    OpenCLIPWrappedTokenizer,
    SigLIP2WrappedTokenizer,
)


class CRADIO(BaseCLIPAdapter):
    """Adapter using torch.hub C-RADIO with built-in text adaptors.

    The adaptor provides both a vision projection head and a text encoder.
    Tokenization and text encoding are delegated to the adaptor's built-in
    interface, with a thin normalization layer to unify the output format
    (OpenCLIP returns raw tensors; SigLIP2 returns dicts).

    Args:
        model_version: C-RADIO model version (e.g., 'c-radio_v3-l').
        adaptor_name: Name of the text adaptor ('siglip2' or 'clip').
        logit_scale_init: Optional raw scale override; None preserves native
            calibration or uses the selected loss-family fallback.
        logit_bias_init: Optional bias override; None preserves native
            absence or uses the loss-family fallback for local ownership.
        loss_type: Contrastive loss family used for fallback initialization.
        freeze_vision_encoder: Freeze vision encoder parameters.
        freeze_text_encoder: Freeze text encoder parameters.
        canonicalize_text: Apply text canonicalization before tokenization.
    """

    # Map adaptor names to text model attribute inside the adaptor module.
    _TEXT_MODEL_ATTR = {
        'siglip2': 'text_model',
        'siglip2-g': 'text_model',
        'clip': 'oc_model',
    }

    def __init__(
        self,
        model_version='c-radio_v3-l',
        adaptor_name='siglip2',
        logit_scale_init=None,
        logit_bias_init=None,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
        canonicalize_text=False,
        loss_type='siglip',
    ):
        """Initialize CRADIO adapter."""
        super().__init__(
            loss_type=loss_type,
            owns_logit_parameters=False,
        )

        self.model_version = model_version
        self.adaptor_name = adaptor_name
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_text_encoder = freeze_text_encoder

        logging.info(
            f"Loading RADIO model: {model_version} (adaptor: {adaptor_name})"
        )

        self.radio_model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=model_version,
            progress=True,
            skip_validation=True,
            adaptor_names=adaptor_name,
        )
        self.radio_model.make_preprocessor_external()

        source_owner = self._source_logit_owner()
        self._uses_source_logit_calibration = source_owner is not None
        if source_owner is None:
            self._create_local_logit_parameters(
                logit_scale_init=logit_scale_init,
                logit_bias_init=logit_bias_init,
                loss_type=loss_type,
            )
        else:
            configure_source_logit_calibration(
                source_owner,
                logit_scale_init=logit_scale_init,
                logit_bias_init=logit_bias_init,
                loss_type=loss_type,
                bias_required=False,
            )
            self.logit_scale_max = source_owner.logit_scale_max

        # Wrap the adaptor's built-in tokenizer for dataloader compatibility.
        # Apply canonicalization wrapper based on config.
        raw_tokenizer = self.adaptor.tokenizer
        if adaptor_name == 'clip':
            # OpenCLIP tokenizer returns raw tensors; wrap to produce dicts
            raw_tokenizer = OpenCLIPWrappedTokenizer(
                raw_tokenizer, canonicalize=canonicalize_text
            )
        else:
            # RADIO's SigLIP2 adaptor already wraps the HF processor in
            # its own SigLIP2WrappedTokenizer (stored as ._proc). Extract
            # the underlying processor to avoid double-wrapping and to
            # give us control over canonicalization.
            processor = getattr(raw_tokenizer, '_proc', raw_tokenizer)
            raw_tokenizer = SigLIP2WrappedTokenizer(
                processor, canonicalize=canonicalize_text
            )
        self.tokenizer = CLIPCompatibleTokenizer(raw_tokenizer)

        self._configure_trainable_params()
        self._log_parameters()

    @property
    def adaptor(self):
        """Return the selected adaptor without registering a second alias."""
        return self.radio_model.adaptors[self.adaptor_name]

    def _source_logit_owner(self):
        """Return native adaptor calibration owner when one is available."""
        source = self.text_model
        logit_scale = getattr(source, 'logit_scale', None)
        logit_bias = getattr(source, 'logit_bias', None)
        if logit_scale is None:
            if logit_bias is not None:
                raise ValueError(
                    "RADIO adaptor exposes logit_bias without logit_scale."
                )
            return None
        return source

    def get_logit_scale_parameter(self):
        """Return native or local canonical raw logit scale."""
        if self._uses_source_logit_calibration:
            return self._source_logit_owner().logit_scale
        return super().get_logit_scale_parameter()

    def get_logit_bias_parameter(self):
        """Return native or local canonical logit bias."""
        if self._uses_source_logit_calibration:
            return getattr(self._source_logit_owner(), 'logit_bias', None)
        return super().get_logit_bias_parameter()

    @property
    def text_model(self):
        """Return the text encoder sub-module inside the adaptor."""
        attr = self._TEXT_MODEL_ATTR.get(self.adaptor_name)
        if attr and hasattr(self.adaptor, attr):
            return getattr(self.adaptor, attr)
        raise AttributeError(
            f"Unknown adaptor '{self.adaptor_name}'. "
            f"Supported: {list(self._TEXT_MODEL_ATTR)}"
        )

    def _configure_trainable_params(self):
        """Configure trainable params without freezing native calibration."""
        if self.freeze_vision_encoder and self.freeze_text_encoder:
            logging.warning(
                "Both vision and text encoders are frozen. "
                "Only available logit calibration parameters will be trained."
            )

        text_param_ids = {id(param) for param in self.text_model.parameters()}
        logit_param_ids = {
            id(param) for _, param in self.named_logit_parameters()
        }
        for param in self.radio_model.parameters():
            if id(param) in logit_param_ids:
                param.requires_grad = True
                continue
            is_text = id(param) in text_param_ids
            freeze = (
                self.freeze_text_encoder if is_text
                else self.freeze_vision_encoder
            )
            param.requires_grad = not freeze

        if self.freeze_vision_encoder:
            self.radio_model.model.eval()
        if self.freeze_text_encoder:
            self.text_model.eval()

    def _log_parameters(self):
        """Log parameter configuration summary without double-counting."""
        vision_params = [
            param for _, param in self.vision_named_parameters()
        ]
        text_params = [
            param for _, param in self.text_named_parameters()
        ]
        self._log_model_summary(
            model_name=f"RADIO: {self.model_version} ({self.adaptor_name})",
            vision_total=sum(param.numel() for param in vision_params),
            vision_trainable=sum(
                param.numel() for param in vision_params
                if param.requires_grad
            ),
            text_total=sum(param.numel() for param in text_params),
            text_trainable=sum(
                param.numel() for param in text_params if param.requires_grad
            ),
            freeze_vision=self.freeze_vision_encoder,
            freeze_text=self.freeze_text_encoder,
        )

    def get_encoder_blocks(self, tower):
        """Return ordered list of transformer blocks for a given tower."""
        if tower == 'vision':
            return list(self.radio_model.model.blocks)
        elif tower == 'text':
            return list(self.text_model.encoder.layers)
        raise ValueError(f"Unknown tower: {tower}")

    # -- Parameter enumeration for per-tower optimizer groups --

    def vision_named_parameters(self):
        """Yield named parameters for the vision encoder."""
        text_param_ids = {id(param) for param in self.text_model.parameters()}
        for name, param in self.radio_model.named_parameters():
            if id(param) not in text_param_ids:
                yield f'radio_model.{name}', param

    def text_named_parameters(self):
        """Yield text parameters, excluding canonical calibration."""
        attr = self._TEXT_MODEL_ATTR.get(self.adaptor_name)
        if attr is None:
            return
        logit_param_ids = {
            id(param) for _, param in self.named_logit_parameters()
        }
        prefix = (
            f'radio_model.adaptors.{self.adaptor_name}.{attr}'
        )
        for name, param in self.text_model.named_parameters():
            if id(param) not in logit_param_ids:
                yield f'{prefix}.{name}', param

    # -- Forward pass --

    def encode_image(self, image, normalize=True):
        """Encode images through RADIO backbone + adaptor projection."""
        output = self.radio_model(image)
        features = output[self.adaptor_name].summary

        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text, normalize=True):
        """Encode text using the adaptor's built-in text encoder.

        Args:
            text: Dict with 'input_ids' (and optionally 'attention_mask').
            normalize: Whether to L2-normalize output features.

        Returns:
            Text feature tensor.
        """
        device = next(self.adaptor.parameters()).device
        text = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in text.items()
        }

        if self.adaptor_name == 'clip':
            return self.adaptor.encode_text(
                text['input_ids'], normalize=normalize
            )
        return self.adaptor.encode_text(text, normalize=normalize)

    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.radio_model, 'set_grad_checkpointing'):
            self.radio_model.set_grad_checkpointing(enable)
