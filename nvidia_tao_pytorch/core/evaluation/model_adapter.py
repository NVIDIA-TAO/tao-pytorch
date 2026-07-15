# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-adapter contract for shared embedding evaluation.

Unifies every backbone family (RADIO, NV-DINOv2, MAE, ViT/FasterViT students)
behind ONE forward signature, ported from the vfm-eval/c-radiov4 harness
(`common/model_utils.py`):

    adapter(images) -> (summary, features)
        summary  : [B, D]      pooled / CLS embedding  -> KNN, retrieval, cls-probe
        features : [B, T, D]   patch tokens            -> seg probe (reshape to [B, D, H', W'])

Each adapter owns its own normalization, padding-to-patch-divisor, and output
reshaping. This replaces the earlier `get_embed_fn(global|dense)` design.

REUSE NOTE — `cv/backbone_v2` already implements this contract. Its `BackboneBase`
backbones (dino_v3, siglip2, swin, edgenext, ViT/FasterViT RADIO students,
`VisionTransformerMAE`) expose:
    forward(x, return_features=True, return_logits=False) -> (summary, features[B,C,H,W])
    forward_pre_logits(x)  ·  forward_feature_pyramid(x)  ·  get_spatial_feat(x)
So for any backbone_v2-derived model the adapter is a thin passthrough
(`BackboneV2Adapter`), returning the spatial map directly. c-radiov4's
`_ViTRadioStudentAdapter` does the same (`student(x, return_features=True)`).

PORT NOTE: the torch.hub upstream RADIO adapters + `_load_tao_state_dict` live in
`vfm-eval/c-radiov4/common/model_utils.py`; vendor those here for the hub path.
The DINOv2 adapter below is bespoke because NV-DINOv2's ViT returns a token dict
rather than the backbone_v2 contract.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# OpenAI CLIP normalization — the distribution RADIO students expect. Vendored
# from vfm-eval/c-radiov4 common/model_utils.py (Apache-2.0). Used by the
# torch.hub RADIO adapter when the backbone has no internal input_conditioner.
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ModelAdapter(nn.Module, ABC):
    """Wrap a backbone so it returns ``(summary, features)``.

    Attributes:
        patch_size: token stride, used to reshape features to ``[B, D, H', W']``.
        feature_dim: channel dim ``D`` of summary/features.
        preferred_resolution: default square eval resolution.
    """

    patch_size: int = 16
    feature_dim: int = 0
    preferred_resolution: int = 224

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """images [B,3,H,W] -> (summary [B,D], features [B,T,D])."""
        raise NotImplementedError


def features_to_map(features: torch.Tensor, images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[B, T, D]`` patch tokens -> ``[B, D, H', W']`` spatial map (for seg).

    Mirrors c-radiov4 `eval_seg._extract_spatial`. Idempotent: if an adapter
    already returns a ``[B, C, H, W]`` map (e.g. backbone_v2 via
    ``get_spatial_feat`` / ``forward(return_features=True)``), it is returned
    unchanged.
    """
    import math
    if features.dim() == 4:                      # already [B, C, H, W]
        return features.float()
    b, _t, d = features.shape
    h = math.ceil(images.shape[2] / patch_size)
    w = math.ceil(images.shape[3] / patch_size)
    return features.float().view(b, h, w, d).permute(0, 3, 1, 2).contiguous()


class DinoV2Adapter(ModelAdapter):
    """Adapter for NV-DINOv2 (`DinoV2PlModel`): teacher backbone → (CLS, patches)."""

    def __init__(self, pl_model: nn.Module, patch_size: int = 16, feature_dim: int = 1024):
        """Wrap a loaded DinoV2 LightningModule (frozen, eval)."""
        super().__init__()
        self.backbone = pl_model.teacher.backbone
        self.patch_size = patch_size
        self.feature_dim = feature_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (CLS summary [B,D], patch features [B,T,D])."""
        out = self.backbone(images)
        cls = out["x_norm_clstoken"]
        summary = cls[:, 0] if cls.dim() == 3 else cls
        features = out["x_norm_patchtokens"]          # [B, T, D]
        return summary, features


class BackboneV2Adapter(ModelAdapter):
    """Near-passthrough adapter for any ``cv/backbone_v2`` model.

    backbone_v2 backbones (dino_v3, siglip2, swin, edgenext, ViT/FasterViT RADIO
    students, ``VisionTransformerMAE``) already implement the contract via
    ``BackboneBase``:
      - ``forward(x, return_features=True, return_logits=False) -> (summary, features[B,C,H,W])``
      - ``forward_pre_logits(x)`` / ``forward_feature_pyramid(x)`` / ``get_spatial_feat(x)``

    So features come back as a ``[B, C, H, W]`` spatial map directly (no reshape).
    This is the same path c-radiov4's ``_ViTRadioStudentAdapter`` uses. Prefer this
    over a bespoke adapter whenever the backbone derives from backbone_v2.
    """

    def __init__(self, backbone: nn.Module, patch_size: int = 16, feature_dim: int = 0):
        """Wrap a frozen backbone_v2 module (eval)."""
        super().__init__()
        self.backbone = backbone
        self.patch_size = int(getattr(backbone, "patch_size", patch_size) or patch_size)
        self.feature_dim = feature_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (summary [B,D], features [B,C,H,W]) via backbone_v2's contract."""
        return self.backbone(images, return_features=True, return_logits=False)


def load_tao_state_dict(checkpoint_path: str) -> dict:
    """Extract student ViT weights from a TAO Lightning checkpoint.

    Vendored from vfm-eval/c-radiov4 ``common/model_utils._load_tao_state_dict``
    (Apache-2.0). The TAO ``ClassDistiller`` wraps the student so keys look like
    ``model.radio.radio.model.blocks.0.norm1.weight``. After stripping the
    ``model.radio.radio.`` prefix the keys match the upstream RADIO layout
    (``radio.model.blocks.0.norm1.weight``).

    The upstream RADIO release renames the LayerScale parameter ``gamma`` to
    ``grandma`` (a joke rename in its ``hubconf.py``); TAO training stores it
    under the standard timm name ``gamma``. Rename ``ls{1,2}.gamma`` →
    ``ls{1,2}.grandma`` so ``load_state_dict`` picks it up against a hub model.

    Args:
        checkpoint_path: Path to a TAO Lightning ``.pth`` checkpoint.

    Returns:
        A state dict with student weights re-keyed to the upstream RADIO layout.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    prefix = "model.radio.radio."
    student_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    student_sd = {
        k.replace(".ls1.gamma", ".ls1.grandma").replace(".ls2.gamma", ".ls2.grandma"): v
        for k, v in student_sd.items()
    }
    return student_sd


class RadioStudentAdapter(ModelAdapter):
    """Adapter for a torch.hub / TAO RADIO ViT student.

    Vendored from c-radiov4 ``_ViTRadioStudentAdapter`` (Apache-2.0). Used for
    the upstream torch.hub RADIO path and any RADIO student that does **not**
    derive from ``cv/backbone_v2`` (those use :class:`BackboneV2Adapter`). The
    backbone is expected to expose
    ``student(x, return_features=True) -> (summary[B,D], spatial[B,C,H,W])``.

    This adapter owns its normalization (the student's ``input_conditioner`` if
    present, else OpenAI CLIP stats) and pads the input up to the patch divisor,
    then flattens the spatial map to tokens-last ``[B, T, D]`` for the contract.
    """

    def __init__(self, student: nn.Module, input_size: int = 512, patch_size: int = 16,
                 feature_dim: int = 0):
        """Wrap a frozen RADIO student (eval)."""
        super().__init__()
        self.student = student
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.preferred_resolution = input_size
        conditioner = getattr(student, "input_conditioner", None)
        if conditioner is not None and hasattr(conditioner, "norm_mean") and hasattr(conditioner, "norm_std"):
            mean = conditioner.norm_mean.detach().float().view(1, 3, 1, 1)
            std = conditioner.norm_std.detach().float().view(1, 3, 1, 1)
        else:
            mean = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (summary [B,D], features [B,T,D]) with input normalization + padding."""
        x = (images - self._mean) / self._std
        pad_h = (-x.shape[-2]) % self.patch_size
        pad_w = (-x.shape[-1]) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        summary, spatial = self.student(x, return_features=True)
        features = spatial.flatten(2).transpose(1, 2).contiguous()
        return summary, features


class MAEViTAdapter(ModelAdapter):
    """Adapter for an MAE ViT encoder (``ssl/mae/model/mae.py``).

    Calls ``forward_encoder(x, mask_ratio=0.0)`` → all tokens (CLS + patches).
    At ``mask_ratio=0`` no tokens are dropped, but ``random_masking`` still
    permutes patch order; ``ids_restore`` is used to unshuffle the patch tokens
    so the reshaped spatial map is correct for the segmentation probe. (The CLS
    summary used by KNN/retrieval is order-invariant.)

    Only the ViT-MAE family is wired here; FCMAE (ConvNeXtV2) and Hiera-MAE
    produce spatial maps with different pooling and need their own adapters
    (DESIGN open item #1).
    """

    def __init__(self, pl_model: nn.Module, patch_size: int = 16, feature_dim: int = 0):
        """Wrap a loaded MAE LightningModule's encoder (frozen, eval)."""
        super().__init__()
        self.model = getattr(pl_model, "model", pl_model)
        if not hasattr(self.model, "forward_encoder"):
            raise NotImplementedError(
                "MAEViTAdapter expects an MAE ViT with forward_encoder (train.stage='pretrain'). "
                "FCMAE/Hiera-MAE families are not yet wired (DESIGN open item #1)."
            )
        ps = getattr(getattr(self.model, "patch_embed", None), "patch_size", patch_size)
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        self.feature_dim = feature_dim or int(self.model.cls_token.shape[-1])

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (CLS summary [B,D], patch features [B,N,D]) in original patch order."""
        tokens, _mask, ids_restore = self.model.forward_encoder(images, 0.0)
        summary = tokens[:, 0]
        patches = tokens[:, 1:]
        if ids_restore is not None and ids_restore.shape[1] == patches.shape[1]:
            b, n, d = patches.shape
            patches = torch.gather(patches, 1, ids_restore.unsqueeze(-1).expand(b, n, d))
        return summary, patches


# Registry: network name -> adapter factory(pl_model, **cfg) -> ModelAdapter.
# For backbone_v2-derived students (RADIO/ViT/FasterViT/MAE), use
# BackboneV2Adapter(model.backbone) — it's a passthrough to backbone_v2's contract.
ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "nvdinov2": DinoV2Adapter,            # bespoke: DINOv2 ViT returns a token dict, not backbone_v2
    "dinov3": DinoV2Adapter,              # DinoV3PlModel(DinoV2PlModel): teacher.backbone returns the
    #   same x_norm_clstoken/x_norm_patchtokens dict; embed_dim/patch_size come in via cfg.
    "mae": MAEViTAdapter,                 # ViT-MAE encoder (forward_encoder); FCMAE/Hiera TODO (open item #1)
    # RADIO / ViT / FasterViT students: BackboneV2Adapter (backbone_v2 native contract);
    #   only the torch.hub upstream RADIO path needs c-radiov4's bespoke adapter.
}


def build_adapter(network: str, pl_model: nn.Module, **cfg) -> ModelAdapter:
    """Construct the model adapter for ``network``."""
    if network not in ADAPTER_REGISTRY:
        raise KeyError(
            f"No model adapter for '{network}'. Known: {sorted(ADAPTER_REGISTRY)}. "
            f"Port the adapter from vfm-eval/c-radiov4 common/model_utils.py."
        )
    return ADAPTER_REGISTRY[network](pl_model, **cfg).eval()
