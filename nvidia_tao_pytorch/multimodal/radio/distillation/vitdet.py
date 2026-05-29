# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViTDet windowed-attention augmentation for distillation training.

Adapted from EVFM's vitdet_aug_v2.py. Registers forward hooks on
the student ViT's transformer blocks to alternate between local
(windowed) and global self-attention during training, acting as a
regularizer that makes the student robust to restricted attention
patterns while reducing the quadratic cost of self-attention.

Reference: "Exploring Plain Vision Transformer Backbones for Object
Detection" (https://arxiv.org/abs/2203.16527)
"""

import logging
import math
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_NUM_GLOBAL = 4


def _merge_if_real(a: Any, b: Any, merger: Callable[[Any, Any], Any]):
    if a is None:
        return b
    if b is None:
        return a
    return merger(a, b)


class VitDetArgs:
    """Configuration for ViTDet windowed-attention augmentation."""

    def __init__(
        self,
        prob: float = 0.0,
        window_sizes: Union[int, List[int]] = (),
        num_global: Optional[int] = None,
        num_windowed: Optional[int] = None,
        training_only: bool = True,
    ):
        self.prob = prob
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else list(window_sizes)
        self.num_global = num_global
        self.num_windowed = num_windowed
        self.training_only = training_only

    @property
    def enabled(self) -> bool:
        """Return True when ViTDet augmentation should be applied."""
        return self.prob > 0 and len(self.window_sizes) > 0

    @staticmethod
    def merge(*args: "VitDetArgs") -> "VitDetArgs":
        """Merge multiple ``VitDetArgs`` instances into one combined configuration."""
        ret = VitDetArgs(0, [])
        for arg in args:
            ret.prob = max(ret.prob, arg.prob)
            ret.window_sizes.extend(arg.window_sizes)
            ret.num_global = _merge_if_real(ret.num_global, arg.num_global, max)
            ret.num_windowed = _merge_if_real(ret.num_windowed, arg.num_windowed, min)
        # Deduplicate window sizes while preserving order
        seen = set()
        deduped = []
        for ws in ret.window_sizes:
            if ws not in seen:
                seen.add(ws)
                deduped.append(ws)
        ret.window_sizes = deduped
        return ret


class ViTDetAugHook:
    """Forward-hook based ViTDet windowed-attention augmentation.

    Registers hooks on the student ViT's transformer blocks to alternate
    between windowed and global self-attention. On each forward pass,
    with probability ``apply_prob``, a random window size is chosen and
    patches are rearranged so that a periodic subset of layers uses
    local windowed attention while the rest use global attention.

    CLS/register tokens are replicated into each window during windowed
    layers and averaged back when returning to global mode.
    """

    def __init__(
        self,
        blocks: nn.Sequential,
        args: VitDetArgs,
        num_cls_tokens: int,
    ):
        self.blocks = blocks
        self.num_cls_tokens = num_cls_tokens
        self.window_sizes = list(args.window_sizes)
        self.training_only = args.training_only
        self.apply_prob = args.prob

        self._enabled = False
        self._window_size = None
        self._num_windows = None
        self._num_cols = 0
        self._num_rows = 0
        self._batch_size = 0

        # Deterministic so that all ranks pick the same window size
        self._rand = np.random.default_rng(42)

        blocks.register_forward_pre_hook(self._enter_model)

        num_global = args.num_global
        if num_global is None and args.num_windowed is None:
            num_global = DEFAULT_NUM_GLOBAL
            logger.info(
                "VitDet layer pattern not specified. "
                f"Defaulting to {DEFAULT_NUM_GLOBAL} global layers."
            )
        if num_global is not None:
            period = max(len(blocks) // num_global, 1)
            logger.info(f"VitDet pattern: {num_global} global layers. Period: {period}")
        elif args.num_windowed is not None:
            period = args.num_windowed + 1
            num_global = int(math.ceil(len(blocks) / period))
            logger.info(
                f"VitDet pattern: {args.num_windowed} windowed per block. "
                f"Period: {period}. Num global: {num_global}"
            )

        is_global = True
        for i, layer in enumerate(blocks[:-1]):
            ctr = i % period
            if ctr == 0:
                layer.register_forward_pre_hook(self._to_windows)
                is_global = False
            elif ctr == period - 1:
                layer.register_forward_pre_hook(self._to_global)
                is_global = True

        if not is_global:
            blocks[-1].register_forward_pre_hook(self._to_global)

    @property
    def disabled(self):
        """Return True when the ViTDet hook is inactive for the current batch."""
        return self.apply_prob is None or self.apply_prob <= 0 or not self._enabled

    def _enter_model(self, _, inputs: List[torch.Tensor]):
        if self.training_only and not self.blocks.training:
            self._enabled = False
            return None

        if self._rand.random() > self.apply_prob:
            self._enabled = False
            return None

        patches = inputs[0]
        num_patches = patches.shape[1] - self.num_cls_tokens
        self._num_cols = int(math.sqrt(num_patches))
        self._num_rows = num_patches // self._num_cols
        self._batch_size = patches.shape[0]

        # Only use window sizes that evenly divide the spatial grid
        compatible = [
            ws for ws in self.window_sizes
            if self._num_rows % ws == 0 and self._num_cols % ws == 0 and ws < self._num_rows
        ]
        if not compatible:
            self._enabled = False
            return None

        self._enabled = True
        self._window_size = self._rand.choice(compatible)

        return inputs

    def _to_windows(self, _, inputs: List[torch.Tensor]):
        if not self._enabled:
            return None

        patches = inputs[0]
        cls_tokens = patches[:, :self.num_cls_tokens]
        patches = patches[:, self.num_cls_tokens:]

        r = self._num_rows // self._window_size
        c = self._num_cols // self._window_size

        patches = rearrange(
            patches,
            "b (r wr c wc) d -> b (r c) (wr wc) d",
            r=r, c=c, wr=self._window_size, wc=self._window_size,
        )

        patches = torch.cat(
            [
                cls_tokens.unsqueeze(1).expand(-1, patches.shape[1], -1, -1),
                patches,
            ],
            dim=2,
        )

        patches = patches.flatten(0, 1)
        return (patches,) + inputs[1:]

    def _to_global(self, _, inputs: List[torch.Tensor]):
        if not self._enabled:
            return None

        patches = inputs[0]

        r = self._num_rows // self._window_size
        c = self._num_cols // self._window_size

        patches = rearrange(
            patches,
            "(b r c) w d -> b (r c) w d",
            b=self._batch_size, r=r, c=c,
        )

        repl_cls = patches[:, :, :self.num_cls_tokens]
        patches = patches[:, :, self.num_cls_tokens:]

        patches = rearrange(
            patches,
            "b (r c) (wr wc) d -> b (r wr c wc) d",
            r=r, c=c, wr=self._window_size, wc=self._window_size,
        )

        cls_tokens = repl_cls.mean(dim=1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        return (patches,) + inputs[1:]


def apply_vitdet_to_vit(model: nn.Module, args: VitDetArgs):
    """Apply ViTDet augmentation hooks to a VisionTransformer-based student.

    Supports:
      - timm VisionTransformer (has .blocks)
      - TAO RADIO wrapper (has .radio.radio.model.blocks)

    Returns the ViTDetAugHook if applied, or None if the model is
    unsupported or args indicate no augmentation.
    """
    if not args.enabled:
        logger.info("VitDet augmentation disabled (prob=0 or no window sizes).")
        return None

    blocks, num_skip = _extract_blocks_and_num_skip(model)
    if blocks is None:
        logger.warning(
            f"Cannot apply VitDet: unable to find transformer blocks "
            f"in model of type {type(model).__name__}."
        )
        return None

    logger.info(
        f"Applying VitDet augmentation: prob={args.prob}, "
        f"window_sizes={args.window_sizes}, num_skip={num_skip} "
        f"(cls + registers)"
    )
    hook = ViTDetAugHook(blocks, args, num_skip)
    return hook


def _extract_blocks_and_num_skip(model: nn.Module):
    """Extract the transformer blocks and total number of non-spatial tokens.

    Non-spatial tokens include CLS tokens AND register tokens — both must
    be excluded from windowed attention rearrangement.

    Returns (blocks, num_skip) or (None, 0) if not found.
    """
    # TAO RADIO wrapper: model.radio.radio.model.blocks
    radio_model = _deep_getattr(model, "radio.radio.model")
    if radio_model is not None and hasattr(radio_model, "blocks"):
        num_skip = 0
        if hasattr(radio_model, "patch_generator"):
            # num_skip = num_cls_tokens + num_registers
            num_skip = getattr(radio_model.patch_generator, "num_skip", 0)
        elif hasattr(radio_model, "num_summary_tokens"):
            num_skip = radio_model.num_summary_tokens
        return radio_model.blocks, num_skip

    # Plain timm VisionTransformer
    if hasattr(model, "blocks"):
        num_skip = 0
        if hasattr(model, "patch_generator"):
            num_skip = getattr(model.patch_generator, "num_skip", 0)
        elif model.cls_token is not None:
            num_skip = 1
        return model.blocks, num_skip

    return None, 0


def _deep_getattr(obj, attr_path: str):
    """Traverse a dotted attribute path, returning None if any step fails."""
    for attr in attr_path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


@contextmanager
def vitdet_inference_ctx(hook: Optional[ViTDetAugHook], window_size: Optional[int] = None):
    """Context manager to enable ViTDet during inference with a fixed window size.

    If hook is None, this is a no-op.
    """
    if hook is None:
        yield
        return

    if window_size is None:
        ws = sorted(hook.window_sizes)
        window_size = ws[len(ws) // 2]
    elif window_size == 0:
        yield
        return

    prev_prob = hook.apply_prob
    prev_training_only = hook.training_only
    prev_window_sizes = hook.window_sizes
    try:
        hook.apply_prob = 1.0
        hook.training_only = False
        hook.window_sizes = [window_size]
        yield
    finally:
        hook.apply_prob = prev_prob
        hook.training_only = prev_training_only
        hook.window_sizes = prev_window_sizes
