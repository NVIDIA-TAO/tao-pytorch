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

"""SigLIP2 backbone module.

This module provides SigLIP2 implementations for the TAO PyTorch framework.
SigLIP2 is Google's improved vision-language model that supports both image
classification and text encoding for zero-shot classification.

Key Features:
- Support for multiple SigLIP2 model variants (SO400M, Giant)
- Dynamic resolution support via NaFlex
- Text encoding with canonicalization
- Zero-shot classification support
- Integration with TAO backbone framework

Classes:
    SigLIP2Wrapper: SigLIP2 model wrapper with TAO integration
    WrappedTokenizer: Tokenizer wrapper with text canonicalization

Functions:
    siglip2_so400m_patch16_512: SigLIP2 SO400M with 512x512 images
    siglip2_so400m: SigLIP2 SO400M with NaFlex (flexible resolution)

References:
    - https://huggingface.co/google/siglip2-so400m-patch16-naflex
    - https://huggingface.co/google/siglip2-giant-opt-patch16-384
"""

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel, AutoProcessor

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.text_utils import canonicalize_text


class SigLIP2Wrapper(BackboneBase):
    """SigLIP2 model wrapper with TAO integration.

    This class provides a wrapper around Google's SigLIP2 models with additional
    functionality for integration with the TAO PyTorch framework. It supports
    both vision encoding and text encoding for multimodal applications.

    Key Features:
    - Dynamic image size support via NaFlex
    - Text encoding with canonicalization
    - Zero-shot classification support
    - Integration with TAO backbone framework
    - Support for activation checkpointing and layer freezing

    Args:
        clip_model: The underlying SigLIP2 model from HuggingFace.
        tokenizer: Tokenizer for text encoding.
        num_classes (int): Number of classes for classification head. Default: 0.
        patch_size (int): Patch size for vision encoder. Default: 16.
        is_dynamic (bool): Whether to use dynamic resolution (NaFlex). Default: True.
        in_chans (int): Number of input image channels. Default: 3.
        activation_checkpoint (bool): Whether to use activation checkpointing.
            Default: False.
        freeze_at: List of keys corresponding to the stages or layers to freeze.
            Default: None.
        freeze_norm (bool): If True, all normalization layers will be frozen.
            Default: False.
        head_init_scale (float): Initialization scale for the head. Default: 1.0.
        **kwargs: Additional arguments.

    Attributes:
        inner: The underlying SigLIP2 model.
        tokenizer: The wrapped tokenizer.
        num_features (int): Number of features from the vision encoder.
        head (nn.Module): Classification head (Linear layer or Identity).
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        num_classes: int = 0,
        patch_size: int = 16,
        is_dynamic: bool = True,
        in_chans: int = 3,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        head_init_scale=1.0,
        **kwargs,
    ):
        """Initialize the SigLIP2Wrapper."""
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.inner = clip_model
        self.tokenizer = tokenizer

        self._patch_size = patch_size
        self._is_dynamic = is_dynamic
        self.num_features = self.inner.vision_model.config.hidden_size
        # Ensure vision model returns dict outputs by default so we don't
        # need to pass return_dict=True (which breaks ONNX tracing).
        self.inner.vision_model.config.return_dict = True
        self.register_buffer('mask', torch.ones(1, 1, dtype=torch.int32))

        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def patch_size(self):
        """Return the patch size."""
        return self._patch_size

    def get_stage_dict(self):
        """Get the stage dictionary for feature extraction.

        Returns:
            dict: Dictionary mapping stage indices to model components.
        """
        stage_dict = {0: self.inner.vision_model.embeddings}
        for i, block in enumerate(self.inner.vision_model.encoder.layers, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
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
            global_pool (str, optional): Global pooling type (unused).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Summary tensor of shape (B, D).
        """
        summary = self.forward(x, return_features=False, return_logits=True)
        return summary

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_logits: bool = False
    ):
        """Forward pass through the vision encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_features (bool): If True, also return spatial features.
                Default: False.
            return_logits (bool): If True, return pooled features without head.
                Default: False.

        Returns:
            If return_features is True:
                Tuple[torch.Tensor, torch.Tensor]: (summary, features)
            If return_logits is True:
                torch.Tensor: Pooled features of shape (B, D).
            Otherwise:
                torch.Tensor: Classification logits of shape (B, num_classes).
        """
        out_h = x.shape[-2] // self._patch_size
        out_w = x.shape[-1] // self._patch_size

        extra = {}

        if self._is_dynamic:
            pixel_values = rearrange(
                x,
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=self._patch_size,
                p2=self._patch_size,
                h=out_h,
                w=out_w
            )
            mask = self.mask.expand(*pixel_values.shape[:2])
            shapes = torch.tensor(
                [(out_h, out_w)] * pixel_values.shape[0],
                dtype=torch.int64,
                device=x.device
            )
            extra = dict(attention_mask=mask, spatial_shapes=shapes)
        else:
            pixel_values = x

        output = self.inner.vision_model(
            pixel_values=pixel_values,
            **extra
        )

        summary = output.pooler_output

        if return_features:
            features = output.last_hidden_state
            features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)
            return summary, features
        elif return_logits:
            return summary
        else:
            return self.head(summary)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass to extract feature maps.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature maps of shape (B, C, H, W).
        """
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features

    def encode_text(
        self,
        inputs: Dict[str, torch.Tensor],
        normalize: bool = False
    ) -> torch.Tensor:
        """Encode text inputs using the text encoder.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary with 'input_ids' and
                'attention_mask' tensors.
            normalize (bool): Whether to L2-normalize the output. Default: False.

        Returns:
            torch.Tensor: Text features of shape (B, D).
        """
        output = self.inner.text_model(**inputs, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def zero_shot_postproc(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply zero-shot post-processing (logit scale and bias).

        Args:
            logits (torch.Tensor): Raw logits from image-text similarity.

        Returns:
            torch.Tensor: Processed logits with scale and bias applied.
        """
        logit_scale = self.inner.logit_scale.to(logits.device)
        logit_bias = self.inner.logit_bias.to(logits.device)
        logits = logits * logit_scale.exp() + logit_bias
        return logits


class WrappedTokenizer:
    """Tokenizer wrapper with text canonicalization.

    This wrapper applies text canonicalization before tokenization to improve
    zero-shot classification performance.

    Args:
        proc: The underlying processor from HuggingFace.
    """

    def __init__(self, proc):
        """Initialize the WrappedTokenizer."""
        self._proc = proc

    def __call__(self, text: List[str]):
        """Tokenize text with canonicalization.

        Args:
            text (List[str]): List of text strings to tokenize.

        Returns:
            Dict[str, torch.Tensor]: Tokenized outputs with 'input_ids' and
                'attention_mask'.
        """
        c_text = [canonicalize_text(t) for t in text]
        return self._proc(
            text=c_text,
            return_tensors='pt',
            max_length=64,
            padding='max_length',
            truncation=True
        )


def get_siglip2_model(version: str):
    """Create a SigLIP2 model from HuggingFace.

    Args:
        version (str): Model version identifier.

    Returns:
        SigLIP2Wrapper: Wrapped SigLIP2 model.
    """
    # Format: (hf_model_name, is_dynamic, patch_size)
    version_map = {
        # NaFlex (dynamic resolution)
        'siglip2-so400m-patch16-naflex': ('google/siglip2-so400m-patch16-naflex', True, 16),
        # Fixed resolution patch14 variants
        'siglip2-so400m-patch14-224': ('google/siglip2-so400m-patch14-224', False, 14),
        'siglip2-so400m-patch14-384': ('google/siglip2-so400m-patch14-384', False, 14),
        # Fixed resolution patch16 variants
        'siglip2-so400m-patch16-256': ('google/siglip2-so400m-patch16-256', False, 16),
        'siglip2-so400m-patch16-384': ('google/siglip2-so400m-patch16-384', False, 16),
        'siglip2-so400m-patch16-512': ('google/siglip2-so400m-patch16-512', False, 16),
    }

    if version not in version_map:
        raise ValueError(
            f"Unknown SigLIP2 version: {version}. "
            f"Available: {sorted(version_map.keys())}"
        )

    hf_model_name, is_dynamic, patch_size = version_map[version]

    model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)

    tokenizer = WrappedTokenizer(proc)

    model = SigLIP2Wrapper(
        model,
        tokenizer,
        num_classes=0,
        is_dynamic=is_dynamic,
        patch_size=patch_size
    )

    return model


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_naflex(**kwargs):
    """Create SigLIP2 SO400M model with NaFlex (flexible resolution).

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch16-naflex")


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch14_224(**kwargs):
    """Create SigLIP2 SO400M Patch14 224 model.

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch14-224")


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch14_384(**kwargs):
    """Create SigLIP2 SO400M Patch14 384 model.

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch14-384")


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_256(**kwargs):
    """Create SigLIP2 SO400M Patch16 256 model.

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch16-256")


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_384(**kwargs):
    """Create SigLIP2 SO400M Patch16 384 model.

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch16-384")


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_512(**kwargs):
    """Create SigLIP2 SO400M Patch16 512 model.

    Args:
        **kwargs: Additional arguments (unused).

    Returns:
        SigLIP2Wrapper: SigLIP2 model.
    """
    return get_siglip2_model("siglip2-so400m-patch16-512")
