# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 backbone module."""
from typing import Optional, Tuple, Union, Any

import torch
from torch import nn

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase

try:
    from sam3.model_builder import build_sam3_image_model
except Exception as e:
    print(f'Failed to import SAM3. Error: {e}')
    build_sam3_image_model = None
norm_t = Union[Tuple[float, float, float], torch.Tensor]


def _to_tensor(v: norm_t):
    """Convert a triplet or tensor of channel-wise values to a (C, 1, 1) float tensor."""
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)


class InputConditioner(nn.Module):
    """Image input conditioner that scales and normalizes inputs for the SAM3 backbone.

    Applies ``y = (x - mean) / std`` where ``mean`` and ``std`` are pre-divided by
    ``input_scale`` so the conditioner is equivalent to ``(x * input_scale - mean) / std``
    without an explicit multiplication.
    """

    def __init__(self,
                 input_scale: float,
                 norm_mean: norm_t,
                 norm_std: norm_t,
                 dtype: torch.dtype = torch.float32,):
        """Initialize the input conditioner.

        Args:
            input_scale (float): Scalar applied implicitly to the input by dividing the
                normalization buffers; use 1.0 if inputs are already in [0, 1].
            norm_mean (norm_t): Per-channel normalization mean.
            norm_std (norm_t): Per-channel normalization std.
            dtype (torch.dtype, optional): Output dtype. Defaults to ``torch.float32``.
        """
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        """Normalize ``x`` and cast it to ``self.dtype``."""
        y = (x - self.norm_mean) / self.norm_std
        return y.to(self.dtype)

    def backward(self, x: torch.Tensor):
        """Invert the normalization, mapping ``x`` back to the original input space."""
        y = x * self.norm_std + self.norm_mean
        return y.to(self.dtype)

    def to(self, *args, **kwargs):
        """Move/cast the module and synchronize ``self.dtype`` when the dtype changes."""
        super().to(*args, **kwargs)

        dtype_kwarg = kwargs.get('dtype', None)
        if dtype_kwarg is not None:
            self.dtype = dtype_kwarg
        else:
            dtype_args = [arg for arg in args if isinstance(arg, torch.dtype)]
            if len(dtype_args) == 1:
                self.dtype = dtype_args[0]

        return self

    def float(self):
        """Cast the module to ``torch.float`` and update ``self.dtype``."""
        super().float()
        self.dtype = torch.float
        return self

    def double(self):
        """Cast the module to ``torch.double`` and update ``self.dtype``."""
        super().double()
        self.dtype = torch.double
        return self

    def half(self):
        """Cast the module to ``torch.half`` and update ``self.dtype``."""
        super().half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        """Cast the module to ``torch.bfloat16`` and update ``self.dtype``."""
        super().bfloat16()
        self.dtype = torch.bfloat16
        return self

    @staticmethod
    def default():
        """Build a default conditioner using OpenAI CLIP mean/std and unit input scale."""
        from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        return InputConditioner(
            input_scale=1.0,
            norm_mean=OPENAI_CLIP_MEAN,
            norm_std=OPENAI_CLIP_STD,
        )


class SAM3Wrapper(BackboneBase):
    """Wrapper exposing the SAM3 ViT trunk as a TAO ``BackboneBase``.

    The wrapper bypasses SAM3's ``Sam3DualViTDetNeck`` neck (which applies a
    dimensional bottleneck) and runs the underlying ViT trunk directly so the
    raw ``[B, C, H, W]`` global-attention features are available for downstream
    use, with an optional linear classification head on top of pooled features.
    """

    # SAM3 loads its own pretrained weights inside ``build_sam3_image_model``,
    # so the generic post-construction state-dict load in ``build_model`` should
    # be skipped when this flag is set.
    _consumes_pretrained_path = True

    def __init__(self,
                 sam3_vision_encoder,
                 num_classes: int = 0,
                 in_chans: int = 3,
                 activation_checkpoint=False,
                 freeze_at=None,
                 freeze_norm=False,
                 head_init_scale=1.0,
                 **kwargs,):
        """Initialize the SAM3 wrapper.

        Args:
            sam3_vision_encoder: SAM3 vision backbone (``Sam3DualViTDetNeck``); only
                its ``trunk`` is used.
            num_classes (int, optional): Number of classes for the linear head; ``0``
                means no head (Identity). Defaults to ``0``.
            in_chans (int, optional): Number of input channels. Defaults to ``3``.
            activation_checkpoint (bool, optional): Forwarded to ``BackboneBase``.
                Defaults to ``False``.
            freeze_at: Forwarded to ``BackboneBase``. Defaults to ``None``.
            freeze_norm (bool, optional): Forwarded to ``BackboneBase``.
                Defaults to ``False``.
            head_init_scale (float, optional): Multiplier applied to the head weight
                and bias at initialization. Defaults to ``1.0``.
            **kwargs: Unused, accepted for compatibility with backbone factory calls.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        # Extract the ViT trunk directly from the vision backbone (Sam3DualViTDetNeck)
        # This gets us features before the neck applies dimensional bottleneck
        self.inner = sam3_vision_encoder.trunk
        self.num_features = self.inner.patch_embed.proj.out_channels
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @property
    def embed_dim(self):
        """Embedding dimension of the underlying ViT trunk."""
        return self.inner.patch_embed.proj.out_channels

    @property
    def patch_size(self):
        """Spatial patch size (stride of the patch embedding) of the ViT trunk."""
        return self.inner.patch_embed.proj.stride[0]

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
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

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Return the spatial feature map ``[B, C, H, W]`` from the ViT trunk.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Final global-attention feature map.
        """
        _, features = self.forward(x, return_features=True, return_logits=False)
        return features

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False):
        """Run the SAM3 ViT trunk and optionally pool/classify the output.

        Args:
            x (Tensor): Input image tensor of shape ``[B, C, H, W]``.
            return_features (bool, optional): If ``True``, return both the pooled
                summary and the spatial feature map. Defaults to ``False``.
            return_logits (bool, optional): If ``True`` (and ``return_features`` is
                ``False``), return the pooled summary instead of head logits.
                Defaults to ``False``.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - ``(summary, features)`` when ``return_features=True``.
                - ``summary`` when ``return_logits=True``.
                - ``head(summary)`` otherwise.
        """
        # Run the ViT trunk directly to get features before dimensional bottleneck
        # The trunk returns a list of outputs from global attention blocks
        # (typically just the final output unless return_interm_layers is True)
        features_list = self.inner(x)

        # Features are in [B, C, H, W] format from SAM3's ViT output
        features = features_list[-1]

        summary = features.mean(dim=(2, 3))
        if return_features:
            return summary, features
        else:
            if return_logits:
                return summary
            else:
                return self.head(summary)


"""
    sam3_model, input_conditioner = get_sam3_model(model, chk_base_path=chk_base_path, load_from_HF=load_from_HF, wrap=False)
    img_encoder = SAM3Wrapper(sam3_model.backbone.vision_backbone)
"""


def get_sam3_model(chk_base_path: str = None,
                   wrap: bool = True,
                   load_from_HF: bool = True,
                   **kwargs) -> Tuple[Union[SAM3Wrapper, Any], InputConditioner]:
    """Build a SAM3 image model, optionally wrapped as a TAO backbone.

    Args:
        chk_base_path (str, optional): Local checkpoint path. Used only when
            ``load_from_HF`` is ``False``. Defaults to ``None``.
        wrap (bool, optional): If ``True``, return a ``SAM3Wrapper`` around the
            vision encoder; if ``False``, return the raw ``(model, conditioner)``
            pair. Defaults to ``True``.
        load_from_HF (bool, optional): If ``True``, download the model from
            HuggingFace; otherwise load from ``chk_base_path``. Defaults to ``True``.
        **kwargs: Forwarded to ``build_sam3_image_model``.

    Returns:
        SAM3Wrapper when ``wrap=True``; otherwise a tuple
        ``(model, InputConditioner)`` containing the raw SAM3 model and the
        matching input conditioner (0.5 mean/std).

    Raises:
        ImportError: If the ``sam3`` package is not installed.
    """
    if build_sam3_image_model is None:
        raise ImportError('Unable to import Sam3 module. Please install: pip install git+https://github.com/facebookresearch/sam3.git')

    if chk_base_path and not load_from_HF:
        checkpoint_path = chk_base_path
    else:
        checkpoint_path = None

    # Build the SAM3 image model
    # This returns a Sam3Image model with a backbone (SAM3VLBackbone)
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        eval_mode=True,
        device='cuda',
        enable_inst_interactivity=False,  # Don't need SAM1 task for teacher
        **kwargs
    )

    # SAM3 uses 0.5 mean/std normalization (not ImageNet stats)
    # See: https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py
    conditioner = InputConditioner(
        input_scale=1.0,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
    )

    if not wrap:
        return model, conditioner

    # Extract the vision encoder (Sam3DualViTDetNeck) from the backbone
    # model.backbone is SAM3VLBackbone, which has vision_backbone as Sam3DualViTDetNeck
    vision_encoder = model.backbone.vision_backbone
    img_encoder = SAM3Wrapper(vision_encoder)

    return img_encoder


@BACKBONE_REGISTRY.register()
def sam3_default(pretrained_backbone_path: Optional[str] = None, **kwargs):
    """Registered factory returning the default SAM3 backbone.

    Args:
        pretrained_backbone_path (str, optional): Local path to a SAM3 checkpoint
            (``.pt``). When set, the checkpoint is loaded from disk; when
            ``None`` (the default), the model is downloaded from HuggingFace.
            This is sourced from ``model.backbone.pretrained_backbone_path`` in
            the experiment config.
        **kwargs: Forwarded by the backbone registry (e.g. ``num_classes``,
            ``freeze_at``); accepted for compatibility and otherwise ignored.
    """
    if pretrained_backbone_path:
        return get_sam3_model(
            chk_base_path=pretrained_backbone_path,
            wrap=True,
            load_from_HF=False,
        )
    return get_sam3_model(wrap=True, load_from_HF=True)
