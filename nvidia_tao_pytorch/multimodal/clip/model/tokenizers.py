# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer utilities for CLIP-compatible training.

This module provides the canonical implementations for text processing
utilities used across C-RADIO, SigLIP2, and other CLIP-compatible model
implementations.

Classes:
    SigLIP2WrappedTokenizer: Tokenizer wrapper for SigLIP2 with text
        canonicalization
    OpenCLIPWrappedTokenizer: Tokenizer wrapper for DFN CLIP with text
        canonicalization
    CLIPCompatibleTokenizer: Wrapper for CLIP dataloader compatibility

Functions:
    canonicalize_text: Text normalization (lowercase, punctuation removal)
    save_tokenizer: Save tokenizer to disk for deployment
    load_tokenizer: Load tokenizer from disk
"""

import os
from typing import List, Optional

from transformers import AutoTokenizer

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2.text_utils import canonicalize_text


class SigLIP2WrappedTokenizer:
    """Tokenizer wrapper for SigLIP2 with optional text canonicalization.

    This wrapper optionally applies text canonicalization before tokenization.
    Canonicalization (lowercase + punctuation removal) can improve zero-shot
    classification but may hurt retrieval tasks where punctuation matters.

    Args:
        processor: The underlying processor from HuggingFace.
        max_length: Maximum sequence length for tokenization. Default: 64.
        canonicalize: Whether to apply text canonicalization. Default: False.
    """

    def __init__(self, processor, max_length: int = 64, canonicalize: bool = False):
        """Initialize the tokenizer wrapper."""
        # Guard against double-wrapping: if processor is already a wrapped
        # tokenizer (e.g. RADIO's SigLIP2WrappedTokenizer), extract the
        # underlying HuggingFace processor.
        for attr in ('_proc', '_processor'):
            if hasattr(processor, attr):
                processor = getattr(processor, attr)
                break
        self._processor = processor
        self._max_length = max_length
        self._canonicalize = canonicalize

    def __call__(self, text: List[str]):
        """Tokenize text with optional canonicalization.

        Args:
            text: List of strings to tokenize.

        Returns:
            BatchEncoding dict with 'input_ids' and 'attention_mask'.
        """
        if self._canonicalize:
            text = [canonicalize_text(t) for t in text]
        ret = self._processor(
            text=text,
            return_tensors='pt',
            max_length=self._max_length,
            padding='max_length',
            truncation=True
        )
        return ret


class CLIPCompatibleTokenizer:
    """Wrapper to make tokenizers compatible with CLIP dataloader.

    The CLIP dataloader expects tokenizer(text)[0] to return a tensor or
    dict. This wrapper normalizes the interface across different tokenizer
    types:
    - SigLIP2WrappedTokenizer: returns dict with 'input_ids', 'attention_mask'
    - OpenCLIPWrappedTokenizer: returns dict with 'input_ids'

    This wrapper makes tokenizer(text) return a list where [0] gives the
    dict, allowing it to work with the existing dataloader pattern.

    Args:
        wrapped_tokenizer: A SigLIP2WrappedTokenizer or
            OpenCLIPWrappedTokenizer instance.
    """

    def __init__(self, wrapped_tokenizer):
        """Initialize the CLIP-compatible tokenizer wrapper."""
        self._tokenizer = wrapped_tokenizer

    def __call__(self, text):
        """Tokenize text and return in CLIP-compatible format.

        Args:
            text: Single string or list of strings.

        Returns:
            List where [0] is the tokenized dict.
        """
        if isinstance(text, str):
            # Single text - wrap in list, then squeeze the batch dimension
            result = self._tokenizer([text])
            result = {k: v.squeeze(0) for k, v in result.items()}
        else:
            # List of texts - keep batch dimension
            result = self._tokenizer(text)

        return [result]


class OpenCLIPWrappedTokenizer:
    """Tokenizer wrapper for OpenCLIP/DFN-CLIP with optional text canonicalization.

    This wrapper optionally applies text canonicalization before tokenization
    and converts the output to a dict format matching SigLIP2/RADIO for
    consistency.

    Used for:
    - RADIO 'clip' adaptor (DFN CLIP)
    - backbone_v2 OpenCLIP models

    Args:
        tokenizer: The raw OpenCLIP tokenizer (callable that returns tensor).
        canonicalize: Whether to apply text canonicalization. Default: False.
    """

    def __init__(self, tokenizer, canonicalize: bool = False):
        """Initialize the tokenizer wrapper."""
        self._tokenizer = tokenizer
        self._canonicalize = canonicalize

    def __call__(self, text: List[str]):
        """Tokenize text with optional canonicalization and return dict format.

        Args:
            text: List of strings to tokenize.

        Returns:
            Dict with 'input_ids' key containing the tokenized tensor.
        """
        if self._canonicalize:
            text = [canonicalize_text(t) for t in text]

        # OpenCLIP tokenizer returns tensor directly
        result = self._tokenizer(text)

        # Wrap tensor in dict for consistency with SigLIP2/RADIO tokenizers
        return {'input_ids': result}


def save_tokenizer(
    tokenizer: CLIPCompatibleTokenizer,
    output_dir: str,
    model_type: str,
    adaptor_name: Optional[str] = None,
) -> str:
    """Save tokenizer to disk for deployment.

    For HuggingFace-based tokenizers (SigLIP2), saves using save_pretrained().
    For OpenCLIP-based tokenizers (RADIO CLIP), saves the equivalent HuggingFace
    tokenizer which produces identical token IDs.

    Args:
        tokenizer: The CLIPCompatibleTokenizer from the model.
        output_dir: Directory to save tokenizer files.
        model_type: Model type string (e.g., 'siglip2-so400m-patch16-256').
        adaptor_name: For RADIO models, the adaptor name (e.g., 'clip', 'siglip').

    Returns:
        Path to the saved tokenizer directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    inner_tokenizer = tokenizer._tokenizer

    if isinstance(inner_tokenizer, SigLIP2WrappedTokenizer):
        # SigLIP2: save the HuggingFace processor's tokenizer
        processor = inner_tokenizer._processor
        if hasattr(processor, 'tokenizer'):
            hf_tokenizer = processor.tokenizer
        else:
            # Processor is the tokenizer itself
            hf_tokenizer = processor
        hf_tokenizer.model_max_length = inner_tokenizer._max_length  # e.g. 64 for SigLIP2
        hf_tokenizer.save_pretrained(output_dir)
        logging.info("Saved SigLIP2 tokenizer to %s", output_dir)
    else:
        # OpenCLIP-based (RADIO CLIP, OpenCLIP): save equivalent HuggingFace tokenizer
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import (
            map_clip_model_cfg,
        )

        model_type_lower = model_type.lower()

        if 'radio' in model_type_lower:
            # adaptor_name=None defaults to siglip at runtime (see builders.py)
            if adaptor_name is None or 'siglip' in adaptor_name.lower():
                hf_tokenizer_name = "google/siglip2-so400m-patch14-384"
            else:
                hf_tokenizer_name = "openai/clip-vit-large-patch14"
            ctx_len = 77
        else:
            # OpenCLIP models: look up the correct tokenizer from the
            # detailed clip model config (map_clip_model_cfg has text_cfg)
            cfg = map_clip_model_cfg.get(model_type, {})
            text_cfg = cfg.get("text_cfg", {})
            hf_tokenizer_name = text_cfg.get(
                "hf_tokenizer_name", "openai/clip-vit-large-patch14"
            )
            ctx_len = text_cfg.get("context_length", 77)

        hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        hf_tokenizer.model_max_length = ctx_len
        hf_tokenizer.save_pretrained(output_dir)
        logging.info(
            "Saved equivalent HuggingFace tokenizer (%s) to %s",
            hf_tokenizer_name, output_dir
        )

    return output_dir


def load_tokenizer(tokenizer_dir: str) -> AutoTokenizer:
    """Load tokenizer from disk.

    Args:
        tokenizer_dir: Directory containing saved tokenizer files.

    Returns:
        HuggingFace AutoTokenizer instance.

    Raises:
        FileNotFoundError: If tokenizer directory doesn't exist.
    """
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    return AutoTokenizer.from_pretrained(tokenizer_dir)


def get_tokenizer_dir(checkpoint_path: str) -> str:
    """Get the tokenizer directory path from a checkpoint path.

    Args:
        checkpoint_path: Path to model checkpoint file.

    Returns:
        Path to tokenizer directory (sibling to checkpoint).
    """
    return os.path.join(os.path.dirname(checkpoint_path), "tokenizer")
