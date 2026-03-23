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

"""Model builder functions for CLIP-compatible models.

This module provides factory functions to build various vision-language models
with their preprocessing transforms and tokenizers.

Functions:
    build_radio_model: Build C-RADIO model (SigLIP2 or DFN CLIP adaptor)
    build_siglip2_model: Build Google SigLIP2 model
    build_openclip_model: Build OpenCLIP/NV-CLIP model
"""

from open_clip.transform import image_transform
from transformers import AutoProcessor

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2.open_clip import get_openclip_model
from nvidia_tao_pytorch.cv.backbone_v2.siglip2 import get_siglip2_model
from nvidia_tao_pytorch.multimodal.clip.model.adapters.radio import CRADIO
from nvidia_tao_pytorch.multimodal.clip.model.adapters.siglip2 import SigLIP2
from nvidia_tao_pytorch.multimodal.clip.model.adapters.openclip import OpenCLIP
from nvidia_tao_pytorch.multimodal.clip.model.transforms import (
    SigLIP2ImageTransform,
)
from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import (
    radio_model_configs,
    siglip2_model_configs,
    openclip_model_configs,
)

# Normalization constants
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _parse_aug_config(aug_cfg):
    """Parse augmentation config.

    Args:
        aug_cfg: Augmentation configuration dict or None.

    Returns:
        dict: Augmentation config for open_clip.

    Config format:
        - scale: [min, max] for random resized crop. [1.0, 1.0] to disable.
        - color_jitter: [prob, brightness, contrast, saturation, hue].
          [] to disable.
        - grayscale: probability of grayscale. 0.0 to disable.
    """
    if aug_cfg is None:
        return None

    result = {
        'scale': aug_cfg.get('scale', [0.4, 1.0]),
    }

    # Parse grayscale - support both 'grayscale' (new) and
    # 'grayscale_prob'/'gray_scale_prob' (legacy)
    result['gray_scale_prob'] = aug_cfg.get(
        'grayscale',
        aug_cfg.get('grayscale_prob', aug_cfg.get('gray_scale_prob', 0.2))
    )

    # Parse color_jitter: [prob, brightness, contrast, saturation, hue] or []
    color_jitter = aug_cfg.get('color_jitter', [0.8, 0.32, 0.32, 0.32, 0.08])
    if color_jitter and len(color_jitter) == 5:
        result['color_jitter_prob'] = color_jitter[0]
        result['color_jitter'] = color_jitter[1:5]
    elif color_jitter and len(color_jitter) == 4:
        # Legacy format: [brightness, contrast, saturation, hue]
        result['color_jitter_prob'] = aug_cfg.get('color_jitter_prob', 0.8)
        result['color_jitter'] = color_jitter
    elif color_jitter:
        logging.warning(
            "color_jitter has %d elements (expected 4 or 5), disabling.",
            len(color_jitter),
        )
        result['color_jitter_prob'] = 0.0
        result['color_jitter'] = None
    else:
        result['color_jitter_prob'] = 0.0
        result['color_jitter'] = None

    return result


def _build_image_transforms(image_size, aug_cfg, mean, std):
    """Build train and val image transforms.

    Args:
        image_size: Target image size.
        aug_cfg: Augmentation configuration dict or None.
        mean: Normalization mean tuple.
        std: Normalization std tuple.

    Returns:
        Tuple of (preprocess_train, preprocess_val).
    """
    preprocess_train = image_transform(
        image_size=image_size,
        is_train=True,
        mean=mean,
        std=std,
        aug_cfg=_parse_aug_config(aug_cfg)
    )

    preprocess_val = image_transform(
        image_size=image_size,
        is_train=False,
        mean=mean,
        std=std,
    )

    return preprocess_train, preprocess_val


# User-facing adaptor names and their internal RADIO equivalents.
# Users set model.adaptor_name to 'siglip' or 'clip'; the actual
# internal name varies by model version.
_USER_ADAPTOR_NAMES = {'siglip', 'clip'}

_INTERNAL_ADAPTOR_NAMES = {'clip', 'siglip2', 'siglip2-g'}

# Legacy aliases that still work
_ADAPTOR_ALIASES = {
    'dfn_clip': 'clip',
    'dfn': 'clip',
}


def _resolve_adaptor_name(adaptor_name, model_version):
    """Resolve user-facing adaptor name to the internal RADIO adaptor name.

    'siglip' resolves to the correct version-specific name (siglip2/siglip2-g).
    'clip' passes through unchanged. Legacy aliases like 'dfn_clip'
    also resolve.
    """
    adaptor_name = _ADAPTOR_ALIASES.get(adaptor_name, adaptor_name)

    if adaptor_name == 'siglip':
        model_config = radio_model_configs.get(model_version, {})
        resolved = model_config.get('adaptor_name', 'siglip2')
        logging.info(
            f"Resolved adaptor 'siglip' -> '{resolved}' for {model_version}"
        )
        return resolved

    if adaptor_name not in _INTERNAL_ADAPTOR_NAMES:
        logging.warning(
            f"Unknown adaptor '{adaptor_name}'. "
            f"Supported user-facing names: {sorted(_USER_ADAPTOR_NAMES)}."
        )

    return adaptor_name


def build_radio_model(
    model_version='c-radio_v3-l',
    adaptor_name=None,
    aug_cfg=None,
    freeze_vision_encoder=False,
    freeze_text_encoder=False,
    image_size=224,
    logit_scale_init=2.6592,
    logit_bias_init=-10.0,
    canonicalize_text=False,
):
    """Build C-RADIO model with preprocessing transforms and tokenizer.

    Uses RADIO's built-in text adaptor for both vision and text encoding.

    Args:
        model_version: C-RADIO model version
            (e.g., 'c-radio_v3-l', 'c-radio_v3-h')
        adaptor_name: User-facing adaptor name: 'siglip', 'clip', or None
            (default). 'siglip' auto-resolves to the correct internal name
            per model.
        aug_cfg: Augmentation configuration (optional)
        freeze_vision_encoder: Freeze vision encoder parameters
        freeze_text_encoder: Freeze text encoder parameters
        image_size: Input image resolution
        logit_scale_init: Initial logit scale (log-space)
        logit_bias_init: Initial logit bias
        canonicalize_text: Apply text canonicalization before tokenization

    Returns:
        Tuple of (model, preprocess_train, preprocess_val, tokenizer)
    """
    if adaptor_name is None:
        model_config = radio_model_configs.get(model_version, {})
        adaptor_name = model_config.get('adaptor_name', 'siglip2')
    else:
        adaptor_name = _resolve_adaptor_name(adaptor_name, model_version)

    logging.info(
        f"Building RADIO model: {model_version} with adaptor: {adaptor_name}"
    )
    model = CRADIO(
        model_version=model_version,
        adaptor_name=adaptor_name,
        logit_scale_init=logit_scale_init,
        logit_bias_init=logit_bias_init,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_text_encoder=freeze_text_encoder,
        canonicalize_text=canonicalize_text,
    )

    # RADIO uses OpenAI CLIP normalization regardless of adaptor
    mean, std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    preprocess_train, preprocess_val = _build_image_transforms(
        image_size=image_size,
        aug_cfg=aug_cfg,
        mean=mean,
        std=std
    )

    return model, preprocess_train, preprocess_val, model.tokenizer


def build_siglip2_model(
    model_version='siglip2-g-384',
    aug_cfg=None,
    freeze_vision_encoder=False,
    freeze_text_encoder=False,
    image_size=384,
    logit_scale_init=2.6592,
    logit_bias_init=-10.0,
    canonicalize_text=False,
):
    """Build SigLIP2 model with preprocessing transforms and tokenizer.

    Uses backbone_v2/siglip2.py for the model implementation.

    Args:
        model_version: SigLIP2 model version ('siglip2-g-384',
            'siglip2-so400m', etc.)
        aug_cfg: Augmentation configuration (optional, not used with
            HF processor)
        freeze_vision_encoder: Freeze vision encoder parameters
        freeze_text_encoder: Freeze text encoder parameters
        image_size: Input image resolution (not used with HF processor)
        logit_scale_init: Initial logit scale (log-space)
        logit_bias_init: Initial logit bias
        canonicalize_text: Apply text canonicalization before tokenization

    Returns:
        Tuple of (model, preprocess_train, preprocess_val, tokenizer)

    Raises:
        ValueError: If model_version is not recognized.
    """
    if model_version not in siglip2_model_configs:
        raise ValueError(
            f"Unknown SigLIP2 model: {model_version}. "
            f"Available: {list(siglip2_model_configs.keys())}"
        )

    model_config = siglip2_model_configs[model_version]

    # Use backbone_v2/siglip2.py
    logging.info(f"Building SigLIP2 model: {model_version}")

    # Must match versions supported by backbone_v2/siglip2.py
    # get_siglip2_model()
    _SUPPORTED_SIGLIP2 = {
        'siglip2-so400m-patch16-naflex',  # NaFlex (dynamic resolution)
        'siglip2-so400m-patch14-224',
        'siglip2-so400m-patch14-384',
        'siglip2-so400m-patch16-256',
        'siglip2-so400m-patch16-384',
        'siglip2-so400m-patch16-512',
    }
    if model_version not in _SUPPORTED_SIGLIP2:
        raise ValueError(
            f"SigLIP2 model '{model_version}' is not supported. "
            f"Supported: {sorted(_SUPPORTED_SIGLIP2)}"
        )
    backbone_version = model_version

    backbone_model = get_siglip2_model(backbone_version)

    # Get processor for image transforms
    hf_model_name = model_config['hf_model']
    processor = AutoProcessor.from_pretrained(
        hf_model_name, trust_remote_code=True
    )

    # Wrap backbone in adapter
    model = SigLIP2(
        backbone_model,
        processor,
        logit_scale_init=logit_scale_init,
        logit_bias_init=logit_bias_init,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_text_encoder=freeze_text_encoder,
        canonicalize_text=canonicalize_text,
    )

    # Use HuggingFace processor for image transforms
    preprocess_train = SigLIP2ImageTransform(processor, is_train=True)
    preprocess_val = SigLIP2ImageTransform(processor, is_train=False)

    tokenizer = model.tokenizer

    # Note: aug_cfg is not used with HF processor as it handles
    # preprocessing natively
    if aug_cfg is not None:
        logging.warning(
            "aug_cfg is provided but not used with HuggingFace processor. "
            "SigLIP2 uses native HF preprocessing."
        )

    return model, preprocess_train, preprocess_val, tokenizer


def build_openclip_model(
    model_version='ViT-L-14-SigLIP-CLIPA-336',
    aug_cfg=None,
    freeze_vision_encoder=False,
    freeze_text_encoder=False,
    image_size=224,
    logit_scale_init=2.6592,
    logit_bias_init=-10.0,
    canonicalize_text=False,
):
    """Build OpenCLIP model with preprocessing transforms and tokenizer.

    Uses backbone_v2/open_clip.py for the model implementation.

    Args:
        model_version: OpenCLIP model version
            (e.g., 'ViT-L-14-SigLIP-CLIPA-336')
        aug_cfg: Augmentation configuration (optional)
        freeze_vision_encoder: Freeze vision encoder parameters
        freeze_text_encoder: Freeze text encoder parameters
        image_size: Input image resolution
        logit_scale_init: Initial logit scale (log-space)
        logit_bias_init: Initial logit bias
        canonicalize_text: Apply text canonicalization before tokenization

    Returns:
        Tuple of (model, preprocess_train, preprocess_val, tokenizer)

    Raises:
        ValueError: If model_version is not recognized.
    """
    if model_version not in openclip_model_configs:
        raise ValueError(
            f"Unknown OpenCLIP model: {model_version}. "
            f"Available: {list(openclip_model_configs.keys())}"
        )

    logging.info(f"Building OpenCLIP model: {model_version}")
    # Disable canonicalization at backbone level - we handle it in the adapter
    # wrapper to centralize control via the experiment config
    backbone_model = get_openclip_model(
        model_version, canonicalize_text=False
    )

    model = OpenCLIP(
        backbone_model,
        logit_scale_init=logit_scale_init,
        logit_bias_init=logit_bias_init,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_text_encoder=freeze_text_encoder,
        canonicalize_text=canonicalize_text,
    )

    # OpenCLIP/NV-CLIP uses OpenAI CLIP normalization
    preprocess_train, preprocess_val = _build_image_transforms(
        image_size=image_size,
        aug_cfg=aug_cfg,
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD
    )

    return model, preprocess_train, preprocess_val, model.tokenizer
