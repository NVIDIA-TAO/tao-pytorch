# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP2 backbone module."""
import inspect
import string
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel, AutoProcessor

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class SigLIP2Wrapper(BackboneBase):
    """Wrapper exposing a HuggingFace SigLIP2 model as a TAO ``BackboneBase``.

    Supports both fixed-resolution and ``naflex`` (dynamic-resolution) SigLIP2
    variants. When ``is_dynamic`` is ``True``, inputs are unfolded into per-patch
    tokens with an attention mask and spatial shapes; otherwise inputs are passed
    directly as ``pixel_values``. The wrapper also exposes the SigLIP2 text tower
    via :meth:`encode_text` and the standard logit scale/bias used at zero-shot
    inference time via :meth:`zero_shot_postproc`.
    """

    # Pretrained weights are loaded inside ``AutoModel.from_pretrained`` (either
    # from the HF hub or from a local snapshot directory), so the generic
    # post-construction state-dict load in ``build_model`` is skipped.
    _consumes_pretrained_path = True

    def __init__(self,
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
        """Initialize the SigLIP2 wrapper.

        Args:
            clip_model: HuggingFace SigLIP2 model containing ``vision_model`` and
                ``text_model`` submodules.
            tokenizer: Callable tokenizer (e.g. :class:`WrappedTokenizer`) used by
                :meth:`encode_text`.
            num_classes (int, optional): Number of classes for the linear head;
                ``0`` means no head (Identity). Defaults to ``0``.
            patch_size (int, optional): Spatial patch size of the vision tower.
                Defaults to ``16``.
            is_dynamic (bool, optional): If ``True``, run the vision tower in
                ``naflex`` (dynamic-resolution) mode with per-patch tokens.
                Defaults to ``True``.
            in_chans (int, optional): Number of input channels. Defaults to ``3``.
            activation_checkpoint (bool, optional): Forwarded to ``BackboneBase``.
                Defaults to ``False``.
            freeze_at: Forwarded to ``BackboneBase``. Defaults to ``None``.
            freeze_norm (bool, optional): Forwarded to ``BackboneBase``.
                Defaults to ``False``.
            head_init_scale (float, optional): Multiplier applied to the head
                weight and bias at initialization. Defaults to ``1.0``.
            **kwargs: Unused, accepted for compatibility with backbone factories.
        """
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
        self.register_buffer('mask', torch.ones(1, 1, dtype=torch.int32))
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def patch_size(self):
        """Spatial patch size of the SigLIP2 vision tower."""
        return self._patch_size

    def get_stage_dict(self):
        """Get the stage dictionary."""
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
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (Tensor): Input tensor.

        Returns:
            summary (Tensor): Summary tensor.
            features (Tensor): Features tensor.
        """
        summary = self.forward(x, return_features=False, return_logits=True)
        return summary

    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        """Run the SigLIP2 vision tower and optionally return spatial features or head logits.

        Args:
            x (Tensor): Input image tensor of shape ``[B, C, H, W]``.
            return_features (bool, optional): If ``True``, also return the spatial
                feature map reshaped to ``[B, C, H, W]``. Defaults to ``False``.
            return_logits (bool, optional): If ``True`` (and ``return_features`` is
                ``False``), return the pooled summary instead of head logits.
                Defaults to ``False``.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - ``(summary, features)`` when ``return_features=True``.
                - ``summary`` when ``return_logits=True``.
                - ``head(summary)`` otherwise.
        """
        out_h = x.shape[-2] // self._patch_size
        out_w = x.shape[-1] // self._patch_size

        extra = dict()

        if self._is_dynamic:
            pixel_values = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=self._patch_size, p2=self._patch_size,
                                     h=out_h, w=out_w)
            mask = self.mask.expand(*pixel_values.shape[:2])
            shapes = torch.tensor([(out_h, out_w)] * pixel_values.shape[0], dtype=torch.int64, device=x.device)

            extra = dict(attention_mask=mask, spatial_shapes=shapes)
        else:
            pixel_values = x
        sig = inspect.signature(self.inner.vision_model.forward)
        if 'return_dict' in sig.parameters:
            extra['return_dict'] = True
        output = self.inner.vision_model(pixel_values=pixel_values, **extra)

        summary = output.pooler_output
        if return_features:
            features = output.last_hidden_state
            features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)
            return summary, features
        else:
            if return_logits:
                return summary
            else:
                return self.head(summary)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Return the spatial feature map ``[B, C, H, W]`` from the SigLIP2 vision tower.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Spatial feature map.
        """
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features

    def encode_text(self, inputs: Dict[str, torch.Tensor], normalize: bool = False):
        """Encode tokenized text inputs using the SigLIP2 text tower.

        Args:
            inputs (Dict[str, Tensor]): Tokenizer output (``input_ids``,
                ``attention_mask``, ...).
            normalize (bool, optional): If ``True``, L2-normalize the pooled
                token embedding along the last dim. Defaults to ``False``.

        Returns:
            Tensor: Pooled text embedding.
        """
        output = self.inner.text_model(**inputs, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def zero_shot_postproc(self, logits: torch.Tensor):
        """Apply SigLIP2's learned ``logit_scale``/``logit_bias`` to similarity logits.

        Args:
            logits (Tensor): Image-text similarity logits.

        Returns:
            Tensor: ``logits * exp(logit_scale) + logit_bias``.
        """
        logit_scale, logit_bias = self.inner.logit_scale.to(logits.device), self.inner.logit_bias.to(logits.device)
        logits = logits * logit_scale.exp() + logit_bias
        return logits


class WrappedTokenizer:
    """Thin wrapper around a HuggingFace ``AutoProcessor`` that canonicalizes text first.

    Lower-cases inputs, strips punctuation, then tokenizes with fixed
    ``max_length=64`` padding/truncation so the resulting tensors are batchable
    with SigLIP2's text tower.
    """

    def __init__(self, proc):
        """Store the underlying HuggingFace processor.

        Args:
            proc: A HuggingFace ``AutoProcessor`` (or compatible callable).
        """
        self._proc = proc

    def __call__(self, text: List[str]):
        """Canonicalize and tokenize a batch of text strings.

        Args:
            text (List[str]): Input strings.

        Returns:
            BatchEncoding: Tokenizer output with ``return_tensors='pt'`` and
                fixed ``max_length=64`` padding/truncation.
        """
        c_text = [canonicalize_text(t) for t in text]
        return self._proc(text=c_text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
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


def get_siglip2_model(version: str, pretrained_backbone_path: Optional[str] = None):
    """Build a :class:`SigLIP2Wrapper` for a known SigLIP2 release alias.

    Looks up ``version`` in an internal map of supported releases, loads the
    HuggingFace model and processor (from the hub or a local snapshot
    directory), wraps the processor in a :class:`WrappedTokenizer`, and returns
    the wrapped backbone.

    Args:
        version (str): One of ``'siglip2'``, ``'siglip2-so400m'``,
            ``'siglip2-so400m-512'``, ``'siglip2-g'``, or ``'siglip2-g-384'``.
        pretrained_backbone_path (str, optional): Local path to a HuggingFace
            snapshot directory (containing ``config.json``, weights, and
            processor files) for the chosen ``version``. When ``None`` (the
            default), the model and processor are downloaded from the HF hub
            using the registered repo id.

    Returns:
        SigLIP2Wrapper: Configured backbone (with ``num_classes=0``).

    Raises:
        KeyError: If ``version`` is not a recognized release alias.
    """
    version_map = {
        'siglip2-so400m-512': ('google/siglip2-so400m-patch16-512', False, 16),
        'siglip2-so400m': ('google/siglip2-so400m-patch16-naflex', True, 16),
        'siglip2-g-384': ('google/siglip2-giant-opt-patch16-384', False, 16),
    }
    version_map['siglip2'] = version_map['siglip2-so400m']
    version_map['siglip2-g'] = version_map['siglip2-g-384']

    version, is_dynamic, patch_size = version_map[version]

    # ``from_pretrained`` accepts either a hub repo id or a local snapshot dir,
    # so an explicit local path simply replaces the hub id when provided.
    source = pretrained_backbone_path if pretrained_backbone_path else version

    model = AutoModel.from_pretrained(source, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(source, trust_remote_code=True)

    tokenizer = WrappedTokenizer(proc)

    model = SigLIP2Wrapper(model, tokenizer, num_classes=0, is_dynamic=is_dynamic, patch_size=patch_size)

    return model


@BACKBONE_REGISTRY.register()
def siglip2_so400m_patch16_512(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """SigLIP2 SO400M Patch16 512.

    Args:
        pretrained_backbone_path (str, optional): Local HuggingFace snapshot
            directory to load instead of downloading from the hub.
        **kwargs: Forwarded by the backbone registry; accepted for compatibility.
    """
    return get_siglip2_model("siglip2-so400m-512", pretrained_backbone_path=pretrained_backbone_path)


@BACKBONE_REGISTRY.register()
def siglip2_so400m(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """SigLIP2 SO400M.

    Args:
        pretrained_backbone_path (str, optional): Local HuggingFace snapshot
            directory to load instead of downloading from the hub.
        **kwargs: Forwarded by the backbone registry; accepted for compatibility.
    """
    return get_siglip2_model("siglip2-so400m", pretrained_backbone_path=pretrained_backbone_path)


@BACKBONE_REGISTRY.register()
def siglip2_g_384(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """SigLIP2 Giant 384.

    Args:
        pretrained_backbone_path (str, optional): Local HuggingFace snapshot
            directory to load instead of downloading from the hub.
        **kwargs: Forwarded by the backbone registry; accepted for compatibility.
    """
    return get_siglip2_model("siglip2-g-384", pretrained_backbone_path=pretrained_backbone_path)
