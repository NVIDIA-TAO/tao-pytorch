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

"""OpenCLIP backbone module.

This module provides OpenCLIP implementations for the TAO PyTorch framework.
OpenCLIP is a multimodal embeddings model for image and text, based on the
Contrastive Language-Image Pre-Training (CLIP) architecture.

The OpenCLIP implementation supports NV-CLIP models, which are NVIDIA's commercial
version of OpenAI CLIP trained on 700M proprietary images. These models can be
applied to various areas such as multimodal search, zero-shot image classification,
and downstream computer vision tasks such as object detection.

Key Features:
- Support for multiple CLIP model variants (ViT-L, ViT-H)
- Dynamic image size support with positional encoding interpolation
- Text encoding with canonicalization support
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Multimodal capabilities for image-text understanding
- Pre-trained on large-scale proprietary datasets

Classes:
    OpenCLIP: OpenCLIP model wrapper with TAO integration
    WrappedTokenizer: Tokenizer wrapper with text canonicalization

Functions:
    vit_l_14_siglip_clipa_224: ViT Large Patch14 SigLIP CLIPA 224 model
    vit_l_14_siglip_clipa_336: ViT Large Patch14 SigLIP CLIPA 336 model
    vit_h_14_siglip_clipa_224: ViT Huge Patch14 SigLIP CLIPA 224 model

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import vit_l_14_siglip_clipa_224
    >>> model = vit_l_14_siglip_clipa_224(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

References:
    - [https://build.nvidia.com/nvidia/nvclip/modelcard](https://build.nvidia.com/nvidia/nvclip/modelcard)
    - [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
"""

import math
import string
import types
from typing import List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import timm_model
from open_clip.transformer import VisionTransformer as OpenCLIPVisionTransformer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Return canonicalized text (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/image_text/prompt_engineering.py

    Args:
        text: String to be canonicalized.
        keep_punctuation_exact_string: If provided, this exact string is kept.
        trans_punctuation: Translation table for punctuation removal.

    Returns:
        Canonicalized text string.
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


class WrappedTokenizer:
    """Tokenizer wrapper with text canonicalization.

    This wrapper applies text canonicalization before tokenization to improve
    zero-shot classification performance.

    Args:
        tokenizer: The underlying tokenizer from open_clip.
        canonicalize: Whether to apply text canonicalization. Default: True.
    """

    def __init__(self, tokenizer, canonicalize: bool = True):
        """Initialize the WrappedTokenizer."""
        self._tokenizer = tokenizer
        self._canonicalize = canonicalize

    def __call__(self, text: List[str]):
        """Tokenize text with optional canonicalization.

        Args:
            text (List[str]): List of text strings to tokenize.

        Returns:
            torch.Tensor: Tokenized text tensor of shape (B, context_length).
        """
        if self._canonicalize:
            text = [canonicalize_text(t) for t in text]
        return self._tokenizer(text)


# Configuration dictionaries for NV-CLIP models
NVCLIP_COMMON_VISION_CONFIG = {
    "patch_size": 14,
    "no_ln_pre": True,
    "pool_type": "avg",
    "final_ln_after_pool": True,
    "pos_embed_type": "sin_cos_2d",
    "patch_dropout": 0.0,
}
NVCLIP_COMMON_TEXT_CONFIG = {
    "context_length": 77,
    "vocab_size": 32000,
    "hf_tokenizer_name": "bert-base-uncased",
    "tokenizer_kwargs": {"strip_sep_token": True},
    "pool_type": "last",
    "no_causal_mask": True,
}
NVCLIP_MODEL_CONFIG = {
    "ViT-H-14-SigLIP-CLIPA-224": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 1024, "heads": 16, "layers": 24, **NVCLIP_COMMON_TEXT_CONFIG},
    },
    "ViT-L-14-SigLIP-CLIPA-336": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 336,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 768, "heads": 12, "layers": 12, **NVCLIP_COMMON_TEXT_CONFIG},
    },
    "ViT-L-14-SigLIP-CLIPA-224": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            **NVCLIP_COMMON_VISION_CONFIG,
        },
        "text_cfg": {"width": 768, "heads": 12, "layers": 12, **NVCLIP_COMMON_TEXT_CONFIG},
    },
}


class OpenCLIP(BackboneBase):
    """OpenCLIP model wrapper with TAO integration.

    This class provides a wrapper around OpenCLIP models with additional functionality
    for integration with the TAO PyTorch framework. It supports NV-CLIP models, which
    are NVIDIA's commercial version of OpenAI CLIP trained on 700M proprietary images.

    The OpenCLIP model is a multimodal embeddings model for image and text that can
    be applied to various areas such as multimodal search, zero-shot image classification,
    and downstream computer vision tasks such as object detection.

    Key Features:
    - Dynamic image size support with positional encoding interpolation
    - Text encoding with canonicalization support
    - Integration with TAO backbone framework
    - Support for activation checkpointing and layer freezing
    - Multimodal capabilities for image-text understanding
    - Pre-trained on large-scale proprietary datasets

    Args:
        in_chans (int): Number of input image channels. Must be 3 for OpenCLIP models.
            Default: `3`.
        num_classes (int): Number of classes for classification head. Default: `0`.
        model_name (str): The name of the model to load. Must be one of the supported
            NV-CLIP model names. Default: `"ViT-L-14-SigLIP-CLIPA-336"`.
        activation_checkpoint (bool): Whether to use activation checkpointing to reduce
            memory usage at the cost of computation time. Default: `False`.
        freeze_at (list, optional): List of keys corresponding to the stages or layers
            to freeze. If `None`, no specific layers are frozen. Default: `None`.
        freeze_norm (bool): If `True`, all normalization layers in the backbone will
            be frozen. Default: `False`.
        canonicalize_text (bool): Whether to canonicalize text (lowercase, remove
            punctuation) during tokenization. Default: `True`.
        **kwargs: Additional arguments passed to `open_clip.create_model`.

    Attributes:
        model: The underlying OpenCLIP model.
        tokenizer: Tokenizer with optional text canonicalization.
        head (nn.Module): Classification head (Linear layer or Identity).
        num_features (int): Number of features from the visual encoder.

    Raises:
        ValueError: If `in_chans` is not 3.
        NotImplementedError: If the model type is not supported for dynamic image size.

    Example:
        >>> model = OpenCLIP(model_name="ViT-L-14-SigLIP-CLIPA-224", num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    References:
        - [https://build.nvidia.com/nvidia/nvclip/modelcard](https://build.nvidia.com/nvidia/nvclip/modelcard)
        - [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 0,
        model_name: str = "ViT-L-14-SigLIP-CLIPA-336",
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        canonicalize_text=True,
        **kwargs,
    ):
        """Initialize the OpenCLIP model.

        This method initializes the OpenCLIP model with the specified configuration.
        It registers NV-CLIP model configurations, creates the model, enables dynamic
        image size support, and sets up the classification head and tokenizer.

        Args:
            in_chans (int): Number of input image channels. Must be 3 for OpenCLIP models.
                Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `0`.
            model_name (str): The name of the model to load. Must be one of the supported
                NV-CLIP model names. Default: `"ViT-L-14-SigLIP-CLIPA-336"`.
            activation_checkpoint (bool): Whether to use activation checkpointing to reduce
                memory usage at the cost of computation time. Default: `False`.
            freeze_at (list, optional): List of keys corresponding to the stages or layers
                to freeze. If `None`, no specific layers are frozen. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will
                be frozen. Default: `False`.
            canonicalize_text (bool): Whether to canonicalize text (lowercase, remove
                punctuation) during tokenization. Default: `True`.
            **kwargs: Additional arguments passed to `open_clip.create_model`.

        Raises:
            ValueError: If `in_chans` is not 3.
            NotImplementedError: If the model type is not supported for dynamic image size.
        """
        if in_chans != 3:
            raise ValueError(f"in_chans must be 3 for OpenCLIP backbones. Received: in_chans={in_chans}")
        export = kwargs.pop("export", False)
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )
        self._model_name = model_name
        self._register_nvclip_configs()
        self.model = open_clip.create_model(model_name, **kwargs)

        # Create tokenizer with optional canonicalization
        base_tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer = WrappedTokenizer(base_tokenizer, canonicalize=canonicalize_text)

        # Enable dynamic image size after the model is initialized.
        if isinstance(self.model.visual, timm_model.TimmModel):
            self.model.visual.trunk.dynamic_img_size = True
            self.model.visual.trunk.patch_embed.strict_img_size = False
            self.model.visual.trunk.patch_embed.output_fmt = "NHWC"
            self.model.visual.trunk.patch_embed.flatten = False
        elif isinstance(self.model.visual, OpenCLIPVisionTransformer):
            self._enable_interpolated_forward(self.model.visual)
        else:
            raise NotImplementedError(f"Unsupported model type {model_name} for dynamic image size.")
        self.num_features = self.model.visual.output_dim
        self.head = nn.Linear(self.model.visual.output_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _register_nvclip_configs(self):
        """Register NVCLIP model configurations.

        This method registers the NV-CLIP model configurations with the OpenCLIP
        factory to enable loading of these models. It iterates through the
        `NVCLIP_MODEL_CONFIG` dictionary and adds any configurations that are
        not already registered.

        Note:
            This method modifies the global `open_clip.factory._MODEL_CONFIGS`
            dictionary to include NV-CLIP model configurations.
        """
        for model_name, model_cfg in NVCLIP_MODEL_CONFIG.items():
            if model_name not in open_clip.factory._MODEL_CONFIGS:
                open_clip.factory._MODEL_CONFIGS[model_name] = model_cfg

    def _enable_interpolated_forward(self, model: OpenCLIPVisionTransformer):
        """Enable interpolated forward for the model.

        This method enables dynamic image size support by replacing the model's
        forward method with an interpolated version that can handle arbitrary
        input sizes. It also adds a positional encoding interpolation function.

        Args:
            model (OpenCLIPVisionTransformer): The OpenCLIP vision transformer model
                to enable interpolated forward for.

        Note:
            This method modifies the model's forward method and adds an interpolate_pos_encoding
            method to handle dynamic input sizes.
        """

        def interpolate_pos_encoding(self: OpenCLIPVisionTransformer, w, h):
            """Interpolate the position encodings for the model to accept any image size.

            This function interpolates the positional embeddings to match the input
            image size. It handles both the class token and patch tokens separately.

            Args:
                w (int): Width of the input image.
                h (int): Height of the input image.

            Returns:
                torch.Tensor: Interpolated positional embeddings of shape (N+1, D)
                    where N is the number of patches and D is the embedding dimension.

            Note:
                This function adds a small number (0.1) to avoid floating point errors
                in the interpolation, as discussed in the DINO issue #8.
            """
            npatch = (w * h) // (self.patch_size[0] * self.patch_size[1])
            N = self.positional_embedding.shape[0] - 1
            if npatch == N and w == h:
                return self.positional_embedding
            class_pos_embed = self.positional_embedding[0, :]
            patch_pos_embed = self.positional_embedding[1:, :]
            dim = self.positional_embedding.shape[-1]
            w0 = w // self.patch_size[0]
            h0 = h // self.patch_size[1]
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            # We need fp32 for the interpolation
            reshaped_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            )
            patch_pos_embed = nn.functional.interpolate(
                reshaped_pos_embed,
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1], (
                "The interpolated value does not match the positional embedding size."
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)

        def interpolated_forward(self: OpenCLIPVisionTransformer, x: torch.Tensor):
            """Forward using interpolated positional encodings.

            This function performs the forward pass with interpolated positional
            encodings to support dynamic input sizes. It follows the standard
            CLIP vision transformer forward pass but with dynamic positional
            encoding interpolation.

            Args:
                x (torch.Tensor): Input tensor of shape (B, C, H, W).

            Returns:
                torch.Tensor: Output features of shape (B, D) where D is the
                    embedding dimension.
            """
            _, _, W, H = x.shape
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # class embeddings and positional embeddings
            expand_token = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
            x = torch.cat([expand_token.to(x.dtype), x], dim=1)
            # shape = [*, grid ** 2 + 1, width]
            # Support dynamic input size by interpolating the positional encoding
            dynamic_positional_embedding = interpolate_pos_encoding(self, int(W), int(H))
            x = x + dynamic_positional_embedding.to(x.dtype)

            x = self.patch_dropout(x)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            if self.attn_pool is not None:
                if self.attn_pool_contrastive is not None:
                    # This is untested, WIP pooling that should match paper
                    x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                    tokens = self.attn_pool(x)
                    if self.attn_pool_type == "parallel":
                        pooled = self.attn_pool_contrastive(x)
                    else:
                        assert self.attn_pool_type == "cascade"
                        pooled = self.attn_pool_contrastive(tokens)
                else:
                    # this is the original OpenCLIP CoCa setup, does not match paper
                    x = self.attn_pool(x)
                    x = self.ln_post(x)
                    pooled, tokens = self._global_pool(x)
            elif self.final_ln_after_pool:
                pooled, tokens = self._global_pool(x)
                pooled = self.ln_post(pooled)
            else:
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
            if self.proj is not None:
                pooled = pooled @ self.proj
            if self.output_tokens:
                return pooled, tokens
            return pooled

        model.forward = types.MethodType(interpolated_forward, model)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Set the gradient checkpointing for the model.

        This method enables or disables gradient checkpointing in the underlying
        OpenCLIP model. Gradient checkpointing trades compute for memory by
        recomputing intermediate activations during the backward pass.

        Args:
            enable (bool): Whether to enable gradient checkpointing. Default: `True`.
        """
        self.model.set_grad_checkpointing(enable)

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        This method loads the state dictionary into the underlying OpenCLIP model.
        It delegates the loading to the model's load_state_dict method.

        Args:
            state_dict (dict): A dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.

        Returns:
            NamedTuple with missing_keys and unexpected_keys fields.
        """
        return self.model.load_state_dict(state_dict, **kwargs)

    def get_stage_dict(self):
        """Get the stage dictionary for feature extraction.

        This method returns a dictionary mapping stage indices to model components
        for feature extraction. Currently only supports OpenCLIPVisionTransformer
        models.

        Returns:
            dict: Dictionary mapping stage indices to model components.

        Raises:
            NotImplementedError: If the model type is not supported for stage
                dictionary extraction.
        """
        if isinstance(self.model.visual, timm_model.TimmModel):
            # TODO(@hongyuc): Does TimmModel have a unified signature?
            raise NotImplementedError("Stage dictionary is not implemented for TimmModel.")
        elif isinstance(self.model.visual, OpenCLIPVisionTransformer):
            stage_dict = {0: self.model.visual.conv1}
            for i, block in enumerate(self.model.visual.transformer.resblocks, start=1):
                stage_dict[i] = block
        else:
            raise NotImplementedError("Stage dictionary is not implemented for this model type.")
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        This method returns the classification head, which is either a Linear layer
        or an Identity module depending on the number of classes.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        This method resets the classification head to accommodate a new number
        of classes. If num_classes is 0, it uses an Identity module.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current
                implementation). Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the visual encoder.

        This method performs the forward pass through the visual encoder of the
        OpenCLIP model without applying the classification head. It returns the
        raw features from the visual encoder.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Features of shape (B, D) where D is the embedding dimension.
        """
        return self.model.encode_image(x, normalize=False)

    def encode_text(
        self,
        inputs: torch.Tensor,
        normalize: bool = False
    ) -> torch.Tensor:
        """Encode text inputs using the text encoder.

        Args:
            inputs (torch.Tensor): Tokenized text tensor of shape (B, context_length).
            normalize (bool): Whether to L2-normalize the output. Default: False.

        Returns:
            torch.Tensor: Text features of shape (B, D).
        """
        text_features = self.model.encode_text(inputs, normalize=False)

        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def zero_shot_postproc(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply zero-shot post-processing (logit scale and bias).

        Args:
            logits (torch.Tensor): Raw logits from image-text similarity.

        Returns:
            torch.Tensor: Processed logits with scale and bias applied.
        """
        logit_scale = self.model.logit_scale.to(logits.device)
        # OpenCLIP models may or may not have logit_bias
        if hasattr(self.model, 'logit_bias') and self.model.logit_bias is not None:
            logit_bias = self.model.logit_bias.to(logits.device)
            logits = logits * logit_scale.exp() + logit_bias
        else:
            logits = logits * logit_scale.exp()
        return logits

    def forward_feature_pyramid(self, *args, **kwargs):
        """Forward pass through the backbone to extract intermediate feature maps.

        This method is not implemented for OpenCLIP models as they are designed
        for classification rather than feature pyramid extraction.

        Raises:
            NotImplementedError: This method is not implemented for OpenCLIP models.
        """
        raise NotImplementedError("forward_feature_pyramid is not implemented.")

    def forward(self, x):
        """Forward pass through the visual encoder.

        This method performs the complete forward pass through the OpenCLIP model,
        including both the visual encoder and the classification head.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes) if
                num_classes > 0, otherwise features of shape (B, D).
        """
        x = self.model.encode_image(x, normalize=False)
        x = self.head(x)
        return x


def get_openclip_model(model_name: str, canonicalize_text: bool = True, **kwargs):
    """Create an OpenCLIP model.

    Args:
        model_name (str): Model name (e.g., 'ViT-L-14-SigLIP-CLIPA-336').
        canonicalize_text (bool): Whether to canonicalize text. Default: True.
        **kwargs: Additional arguments passed to OpenCLIP constructor.

    Returns:
        OpenCLIP: Wrapped OpenCLIP model with tokenizer.
    """
    return OpenCLIP(model_name=model_name, canonicalize_text=canonicalize_text, **kwargs)


@BACKBONE_REGISTRY.register()
def vit_l_14_siglip_clipa_224(**kwargs):
    """Create a ViT Large Patch14 SigLIP CLIPA 224 model.

    This function creates a ViT Large model with the following specifications:
    - Patch size: 14x14 pixels
    - Image size: 224x224 pixels
    - Embedding dimension: 768
    - Number of layers: 24
    - Width: 1024
    - Head width: 64

    Args:
        **kwargs: Additional arguments passed to OpenCLIP constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `0`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        OpenCLIP: Configured ViT Large model for 224x224 images.

    Example:
        >>> model = vit_l_14_siglip_clipa_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is optimized for
        224x224 input images with SigLIP training.
    """
    return OpenCLIP(model_name="ViT-L-14-SigLIP-CLIPA-224", **kwargs)


@BACKBONE_REGISTRY.register()
def vit_l_14_siglip_clipa_336(**kwargs):
    """Create a ViT Large Patch14 SigLIP CLIPA 336 model.

    This function creates a ViT Large model with the following specifications:
    - Patch size: 14x14 pixels
    - Image size: 336x336 pixels
    - Embedding dimension: 768
    - Number of layers: 24
    - Width: 1024
    - Head width: 64

    Args:
        **kwargs: Additional arguments passed to OpenCLIP constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `0`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        OpenCLIP: Configured ViT Large model for 336x336 images.

    Example:
        >>> model = vit_l_14_siglip_clipa_336(num_classes=1000)
        >>> x = torch.randn(1, 3, 336, 336)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 304M parameters and is optimized for
        336x336 input images with SigLIP training. The larger input size
        provides better performance for high-resolution tasks.
    """
    return OpenCLIP(model_name="ViT-L-14-SigLIP-CLIPA-336", **kwargs)


@BACKBONE_REGISTRY.register()
def vit_h_14_siglip_clipa_224(**kwargs):
    """Create a ViT Huge Patch14 SigLIP CLIPA 224 model.

    This function creates a ViT Huge model with the following specifications:
    - Patch size: 14x14 pixels
    - Image size: 224x224 pixels
    - Embedding dimension: 1024
    - Number of layers: 32
    - Width: 1280
    - Head width: 80

    Args:
        **kwargs: Additional arguments passed to OpenCLIP constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `0`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        OpenCLIP: Configured ViT Huge model for 224x224 images.

    Example:
        >>> model = vit_h_14_siglip_clipa_224(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 632M parameters and is the largest
        available NV-CLIP model. It provides the best performance but
        requires more computational resources.
    """
    return OpenCLIP(model_name="ViT-H-14-SigLIP-CLIPA-224", **kwargs)
