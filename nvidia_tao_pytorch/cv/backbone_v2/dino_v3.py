# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 backbone module.

This module provides DINOv3 implementations for the TAO PyTorch framework.
DINOv3 is a self-supervised learning method that learns robust visual features
without supervision. It extends the Vision Transformer architecture with additional
components for improved feature learning.
"""
from typing import Tuple, Optional, List, Union

import torch
from torch import nn
import timm

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights


_CONVERSION_HINT = (
    "Run `dinov3 convert -e <dinov3_spec.yaml> "
    "convert.checkpoint=<ssl_checkpoint> "
    "convert.output_path=<converted_backbone.safetensors>` first."
)


def validate_dinov3_checkpoint(state_dict, allow_partial=False):
    """Validate that a DINOv3 state dict uses the downstream timm layout.

    TAO DINOv3 SSL checkpoints use a different key layout and must pass through
    the existing `dinov3 convert` task before a downstream task consumes them.
    Hugging Face `timm/*dinov3*` checkpoints and converted TAO checkpoints
    both use the accepted timm layout.

    Args:
        state_dict (Mapping): DINOv3 state dict to validate.
        allow_partial (bool): Permit incomplete timm-format state dicts while
            retaining raw SSL checkpoint rejection.

    Raises:
        ValueError: If the state dict is empty, is a raw TAO SSL checkpoint, or
            does not contain the required timm DINOv3 backbone keys.
    """
    if not state_dict and not allow_partial:
        raise ValueError(f"DINOv3 checkpoint is empty. {_CONVERSION_HINT}")

    keys = []
    for key in state_dict:
        normalized = key
        for prefix in ("vit.", "inner."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        keys.append(normalized)

    raw_ssl_markers = ("mask_token", "register_tokens")
    is_raw_ssl = (
        any(key in raw_ssl_markers for key in keys) or
        any(key.endswith((".ls1.gamma", ".ls2.gamma")) for key in keys) or
        any(key.startswith(("teacher.", "student.")) for key in keys)
    )
    if is_raw_ssl:
        raise ValueError(
            "Raw TAO DINOv3 SSL checkpoints are not accepted by downstream "
            f"task backbones. {_CONVERSION_HINT}"
        )

    required_keys = ("cls_token", "patch_embed.proj.weight")
    if allow_partial:
        return
    if not all(key in keys for key in required_keys) or not any(
        key.startswith("blocks.0.") for key in keys
    ):
        raise ValueError(
            "DINOv3 checkpoint is not a compatible timm-format backbone. "
            f"{_CONVERSION_HINT}"
        )


class DINOV3Wrapper(BackboneBase):
    """DINOv3 wrapper for TAO PyTorch framework.

    This class wraps the DINOv3 model to provide a TAO PyTorch backbone interface.
    It extends the BackboneBase class to provide additional functionality for backbone
    integration and feature extraction.
    """

    # Pretrained weights are loaded inside ``timm.create_model`` (either
    # downloaded via ``pretrained=True`` or read from ``checkpoint_path``), so
    # the generic post-construction state-dict load in ``build_model`` is
    # skipped.
    _consumes_pretrained_path = True

    def __init__(self,
                 dino_model: nn.Module,
                 num_classes: int = 0,
                 in_chans: int = 3,
                 activation_checkpoint: bool = False,
                 freeze_at: Optional[List[Union[int, str]]] = None,
                 freeze_norm: bool = False,
                 head_init_scale: float = 1.0,
                 **kwargs):
        """Initialize the DINOv3 wrapper.

        Args:
            dino_model (nn.Module): The DINOv3 model to wrap.
            num_classes (int): The number of classes for classification.
            in_chans (int): The number of input channels.
            activation_checkpoint (bool): Whether to use activation checkpointing.
            freeze_at (list): The layers to freeze.
            freeze_norm (bool): Whether to freeze the normalization layers.
            head_init_scale (float): The scale for the head initialization.
            **kwargs: Additional arguments passed to the BackboneBase initializer.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.inner = dino_model
        self.num_features = self.inner.num_features
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def embed_dim(self):
        """Get the embedding dimension."""
        return self.inner.embed_dim

    @property
    def patch_size(self):
        """Get the patch size."""
        return 16

    @property
    def num_summary_tokens(self):
        """Get the number of summary tokens."""
        return self.inner.num_prefix_tokens

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.inner.patch_embed}
        for i, block in enumerate(self.inner.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        The wrapped timm model lives under `self.inner`, so the wrapper's parameters are named
        `inner.*`. A timm DINOv3 checkpoint (or an SSL->timm converted backbone) uses bare keys
        (`blocks.*`, `cls_token`, `patch_embed.*`, `norm.*`, ...); route those to the
        wrapped model by prefixing `inner.`.

        Note: the prefixing is all-or-nothing. It only triggers when *no* key is already
        `inner.`/`head.`-prefixed, i.e. it expects a uniformly-bare timm state dict (the
        convert/timm load paths). A mixed dict (some keys bare, some already `inner.`-prefixed)
        skips prefixing and will fail the strict load below; that case is not supported.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.
        """
        validate_dinov3_checkpoint(
            state_dict,
            allow_partial=not kwargs.get("strict", True),
        )
        if state_dict and not any(k.startswith(("inner.", "head.")) for k in state_dict):
            state_dict = {f"inner.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, **kwargs)

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

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass through the backbone to extract intermediate feature maps."""
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features

    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        """Forward pass through the backbone.

        Args:
            x (Tensor): Input tensor.
            return_features (bool): Whether to return the features.
            return_logits (bool): Whether to return the logits.
        """
        B, _, height, width = x.shape
        x = self.inner.forward_features(x)

        cls_token = x[:, 0]
        features = x[:, self.inner.num_prefix_tokens:]

        if return_features:
            # reshape to BCHW output format
            H, W = self.inner.patch_embed.dynamic_feat_size((height, width))
            features = features.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # features = rearrange(features, 'b (h w) c -> b c h w', h=out_h, w=out_w)
            return cls_token, features
        else:
            if return_logits:
                return cls_token
            else:
                return self.head(cls_token)


def create_dinov3_model(dino_v3_model: str, pretrained: bool = False, **kwargs):
    """Create a timm DINOv3 model with actionable pretrained-weight errors."""
    try:
        return timm.create_model(dino_v3_model, pretrained=pretrained, **kwargs)
    except Exception as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            "Unable to load pretrained DINOv3 weights from timm/Hugging Face. "
            "Ensure HF_TOKEN grants access to the gated model, or set "
            "model.backbone.pretrained_backbone_path to a local converted TAO "
            f"checkpoint. {_CONVERSION_HINT}"
        ) from exc


def _load_dino_v3(
    dino_v3_model: str,
    pretrained_backbone_path: Optional[str] = None,
    pretrained: bool = False,
):
    """Load the DINOv3 model.

    Args:
        dino_v3_model (str): The timm model name.
        pretrained_backbone_path (str, optional): Path to a local DINOv3
            checkpoint file. When set, weights are validated and loaded from
            disk; otherwise `pretrained` controls whether timm/HF weights are
            downloaded.
        pretrained (bool): Fetch timm/HF weights when no local path is set.
    """
    # option1: use torch.hub
    # model = torch.hub.load(
    #     'facebookresearch/dinov3',
    #     dino_v3_model,
    #     pretrained=True,
    # )
    # if pretrained:
    #     chk_path = os.path.join(base_path, _CHECKPOINT_MAP[dino_v3_model])
    #     chk = torch.load(chk_path, map_location='cpu')
    #     model.load_state_dict(chk)
    # return model
    # option2: use transformers
    # if is_local:
    #     ckpt_path = os.path.join(_DEFAULT_BASE_PATH, dino_v3_model)
    # else:
    #     ckpt_path = dino_v3_model
    # print(f"Loading DINOv3 model from {ckpt_path}")
    # model = AutoModel.from_pretrained(ckpt_path)
    # option3: use timm
    if pretrained_backbone_path:
        state_dict = load_pretrained_weights(
            pretrained_backbone_path,
            weights_only=True,
        )
        validate_dinov3_checkpoint(state_dict)
        model = create_dinov3_model(dino_v3_model, pretrained=False)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = create_dinov3_model(dino_v3_model, pretrained=pretrained)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vit7b16(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with 7B parameters.

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_7b_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vitl16(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with L parameters.

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_large_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vits16(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with S parameters.

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_small_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vits16plus(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with S+ parameters (SwiGLU FFN).

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_small_plus_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vitb16(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with B parameters.

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_base_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model


@BACKBONE_REGISTRY.register()
def dinov3_vith16plus(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Load the DINOv3 model with H parameters.

    Args:
        pretrained_backbone_path (str, optional): Local DINOv3 checkpoint path.
            Pass ``pretrained=True`` to download timm/HF weights when unset.
        **kwargs: Forwarded to :class:`DINOV3Wrapper`.
    """
    pretrained = kwargs.pop("pretrained", False)
    model = _load_dino_v3("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained_backbone_path, pretrained)
    model = DINOV3Wrapper(model, **kwargs)
    return model
