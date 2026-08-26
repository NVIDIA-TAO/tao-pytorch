# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""InternVideo2-CLIP adapter for TAO CLIP training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.video_clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.internvideo2_assets import (
    DEFAULT_INTERNVIDEO2CLIP_HF_ID,
    AttrDict,
    build_internvideo2_l14_config,
    resolve_internvideo2_l14_assets,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    load_partial_pretrained_weights,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.backbones.internvideo2 import (
    InternVideo2_CLIP_small,
)


class InternVideo2Tokenizer:
    """Wrapper that matches the CLIP dataloader tokenizer contract."""

    def __init__(self, tokenizer):
        """Initialize wrapper around OpenGVLab tokenizer."""
        self._tokenizer = tokenizer

    def __call__(self, text):
        """Tokenize a string or list and return ``[tokens]``."""
        if isinstance(text, str):
            result = self._tokenizer([text])
            if hasattr(result, "items"):
                result = {k: v.squeeze(0) for k, v in result.items()}
            elif isinstance(result, torch.Tensor) and result.ndim > 1:
                result = result.squeeze(0)
        else:
            result = self._tokenizer(text)
        return [result]


class InternVideo2FrameTransform:
    """InternVideo2 L14 frame preprocessing."""

    def __init__(self, image_size=224):
        """Initialize deterministic frame transform."""
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def __call__(self, image):
        """Transform one PIL image into a normalized CHW tensor."""
        return self.transform(image)


def _to_device(tokens, device):
    """Move token containers to device."""
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in tokens.items()
        }
    return tokens


class InternVideo2CLIP(BaseCLIPAdapter):
    """Adapter around OpenGVLab InternVideo2_CLIP_small."""

    def __init__(
        self,
        internvideo2clip_hf_id=DEFAULT_INTERNVIDEO2CLIP_HF_ID,
        vision_encoder=None,
        text_encoder=None,
        clip_head=None,
        pretrained_ckpt=None,
        num_frames=8,
        image_size=224,
        logit_scale_init=4.605170185988092,
        logit_bias_init=0.0,
        freeze_vision_encoder=False,
        freeze_text_encoder=False,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        log_parameters=True,
    ):
        """Initialize InternVideo2-CLIP L14 from a whole-model or component ckpts.

        The architecture is vendored in-repo
        (``...video_clip.model.backbones.internvideo2``); weights load from the
        ambient HF cache.
        """
        super().__init__(
            logit_scale_init=logit_scale_init,
            logit_bias_init=logit_bias_init,
        )
        self.num_frames = num_frames
        self.image_size = image_size

        # Marker so downstream tooling (e.g. ONNX export) can detect this as an
        # InternVideo2 video adapter and shape inputs accordingly (5D
        # [B, T, C, H, W] vision input, raw-tensor text input).
        self.is_internvideo2 = True

        load_full = bool(pretrained_ckpt)
        if load_full:
            # Whole-model checkpoint: skip component resolution; build the
            # architecture without the upstream component loads, then load the
            # complete state_dict over it (final precedence).
            logging.info(
                "model.pretrained_ckpt set -> loading the complete "
                "InternVideo2-CLIP model from %s (component sources ignored).",
                pretrained_ckpt,
            )
            assets = AttrDict({
                "vision_ckpt": None, "text_ckpt": None, "extra_ckpt": None,
            })
        else:
            assets = resolve_internvideo2_l14_assets(
                AttrDict({
                    "internvideo2clip_hf_id": internvideo2clip_hf_id,
                    "vision_encoder": vision_encoder,
                    "text_encoder": text_encoder,
                    "clip_head": clip_head,
                })
            )
        cfg = build_internvideo2_l14_config(
            assets=assets,
            num_frames=num_frames,
            image_size=image_size,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_text_encoder=freeze_text_encoder,
            use_flash_attn=use_flash_attn,
            use_fused_rmsnorm=use_fused_rmsnorm,
            use_fused_mlp=use_fused_mlp,
        )

        self.backbone = InternVideo2_CLIP_small(
            config=cfg, is_pretrain=True,
        )

        if load_full:
            load_partial_pretrained_weights(
                self.backbone,
                pretrained_ckpt,
                prefixes=("module.", "model.", "backbone."),
                source=pretrained_ckpt,
            )

        self.backbone.temp.requires_grad = False
        self.tokenizer = InternVideo2Tokenizer(self.backbone.tokenizer)
        if log_parameters:
            self._log_parameters(freeze_vision_encoder, freeze_text_encoder)

    def _log_parameters(self, freeze_vision, freeze_text):
        """Log trainable parameter summary."""
        vision_params = list(self.vision_named_parameters())
        text_params = list(self.text_named_parameters())
        vision_trainable, vision_total = self._count_params(
            p for _, p in vision_params
        )
        text_trainable, text_total = self._count_params(
            p for _, p in text_params
        )

        # Split the vision tower so the report attributes trainable params
        # correctly: the ViT backbone (vision_encoder) is frozen while the
        # alignment/projection head (vision_align) is trainable.
        def _split(params, key):
            sub = [(n, p) for n, p in params if key in n]
            trainable = sum(p.numel() for _, p in sub if p.requires_grad)
            total = sum(p.numel() for _, p in sub)
            return trainable, total

        bb_tr, bb_tot = _split(vision_params, ".vision_encoder.")
        al_tr, al_tot = _split(vision_params, ".vision_align.")
        vision_subrows = [
            row for row in (
                ("Vision backbone", bb_tr, bb_tot),
                ("Vision align head", al_tr, al_tot),
            ) if row[2] > 0
        ] or None

        self._log_model_summary(
            model_name="InternVideo2-CLIP L14",
            vision_total=vision_total,
            vision_trainable=vision_trainable,
            text_total=text_total,
            text_trainable=text_trainable,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            vision_subrows=vision_subrows,
        )

    def vision_named_parameters(self):
        """Return vision tower parameters."""
        for name, param in self.backbone.vision_encoder.named_parameters():
            yield f"backbone.vision_encoder.{name}", param
        for name, param in self.backbone.vision_align.named_parameters():
            yield f"backbone.vision_align.{name}", param

    def text_named_parameters(self):
        """Return text tower parameters."""
        for name, param in self.backbone.text_encoder.named_parameters():
            yield f"backbone.text_encoder.{name}", param

    def get_encoder_blocks(self, tower):
        """Return the ordered transformer blocks for a tower (LoRA target).

        Confirmed against a live InternVideo2-CLIP L14 build:

        - Vision tower (``InternVideo2``, 24 blocks): blocks live in
          ``vision_encoder.blocks`` (``nn.ModuleList`` of ``Block``); each block's
          attention exposes a fused ``qkv`` plus ``proj`` -- so the vision
          ``target_modules`` are ``["qkv", "proj"]``.
        - Text tower (MobileCLIP ``TextTransformer``, 12 blocks): the block
          ``nn.ModuleList`` is ``text_encoder.transformer`` itself; each block's
          ``MultiHeadAttention`` exposes a fused ``qkv_proj`` plus ``out_proj`` --
          so the text ``target_modules`` are ``["qkv_proj", "out_proj"]``.

        The text block container is resolved from a prioritized candidate list
        (kept for robustness across InternVideo2-CLIP builds); if none resolves we
        raise with the candidates and the ModuleLists actually present.

        Args:
            tower: 'vision' or 'text'.

        Returns:
            List of nn.Module transformer blocks in forward order.
        """
        if tower == 'vision':
            return list(self.backbone.vision_encoder.blocks)
        if tower == 'text':
            text_encoder = self.backbone.text_encoder
            candidates = (
                'transformer',          # MobileCLIP TextTransformer (confirmed)
                'transformer.resblocks',
                'transformer.layers',
                'encoder.layers',
                'blocks',
            )
            for path in candidates:
                module = text_encoder
                try:
                    for part in path.split('.'):
                        module = getattr(module, part)
                except AttributeError:
                    continue
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    return list(module)
            present = [
                name for name, mod in text_encoder.named_modules()
                if isinstance(mod, nn.ModuleList) and len(mod) > 0
            ]
            raise NotImplementedError(
                "Could not locate the text-encoder transformer blocks for "
                f"InternVideo2-CLIP (MobileCLIP). Tried {candidates}; "
                f"ModuleLists present under text_encoder: {present}. Pin the "
                "correct path against a live model before enabling text LoRA."
            )
        raise ValueError(f"Unknown tower: {tower}")

    def encode_image(self, image, normalize=True):
        """Encode video/image tensor to features.

        Expects video tensor shape ``[B, T, C, H, W]``.
        """
        device = next(self.backbone.parameters()).device
        image = image.to(device)
        features = self.backbone.encode_vision(image)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text, normalize=True):
        """Encode tokenized text to features."""
        device = next(self.backbone.parameters()).device
        text = _to_device(text, device)
        features = self.backbone.encode_text(text)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def forward(self, image=None, text=None):
        """Forward pass compatible with the TAO CLIP module."""
        if image is not None and text is not None:
            image_features = self.encode_image(image, normalize=True)
            text_features = self.encode_text(text, normalize=True)
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        if image is not None:
            return {"image_features": self.encode_image(image, normalize=True)}
        if text is not None:
            return {"text_features": self.encode_text(text, normalize=True)}
        raise ValueError("Either image or text must be provided")
