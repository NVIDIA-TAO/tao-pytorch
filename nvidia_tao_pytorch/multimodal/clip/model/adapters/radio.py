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
        logit_scale_init: Initial value for logit scale parameter.
        logit_bias_init: Initial value for logit bias parameter.
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
        logit_scale_init=2.3026,
        logit_bias_init=-10.0,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
        canonicalize_text=False,
    ):
        """Initialize CRADIO adapter."""
        super().__init__(
            logit_scale_init=logit_scale_init,
            logit_bias_init=logit_bias_init,
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

        self.adaptor = self.radio_model.adaptors[adaptor_name]

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
        """Configure trainable params based on freeze settings."""
        if self.freeze_vision_encoder and self.freeze_text_encoder:
            logging.warning(
                "Both vision and text encoders are frozen. "
                "Only logit_scale and logit_bias will be trained."
            )

        text_param_ids = {id(p) for p in self.text_model.parameters()}

        for param in self.radio_model.parameters():
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
        """Log parameter configuration summary."""
        text_total = sum(p.numel() for p in self.text_model.parameters())
        text_train = sum(
            p.numel() for p in self.text_model.parameters() if p.requires_grad
        )
        radio_total = sum(p.numel() for p in self.radio_model.parameters())
        radio_train = sum(
            p.numel() for p in self.radio_model.parameters() if p.requires_grad
        )

        self._log_model_summary(
            model_name=f"RADIO: {self.model_version} ({self.adaptor_name})",
            vision_total=radio_total - text_total,
            vision_trainable=radio_train - text_train,
            text_total=text_total,
            text_trainable=text_train,
            freeze_vision=self.freeze_vision_encoder,
            freeze_text=self.freeze_text_encoder,
        )

    # -- Parameter enumeration for per-tower optimizer groups --

    def vision_named_parameters(self):
        """Yield named parameters for the vision encoder."""
        text_param_ids = {id(p) for p in self.text_model.parameters()}
        for name, param in self.radio_model.named_parameters():
            if id(param) not in text_param_ids:
                yield f'radio_model.{name}', param

    def text_named_parameters(self):
        """Yield named parameters for the text encoder."""
        attr = self._TEXT_MODEL_ATTR.get(self.adaptor_name)
        if attr is None:
            return
        prefix = f'adaptor.{attr}'
        for name, param in self.text_model.named_parameters():
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
