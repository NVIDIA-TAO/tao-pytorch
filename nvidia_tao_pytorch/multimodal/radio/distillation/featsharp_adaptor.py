# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FeatSharp adaptor for teacher models in the distillation pipeline.

Wraps a teacher backbone so that its spatial features are upsampled via a
pre-trained FeatSharp model before being used as distillation targets. This
gives the student higher-resolution spatial targets to learn from.

Requires the ``featsharp`` library to be importable (either installed as a
package or available on ``sys.path``).  A FeatSharp checkpoint trained for
the specific teacher architecture is also required.

Adapted from EVFM's ``teacher_adaptors.py`` (``UpsampleFeatSharp``).
"""

import importlib.util
import logging
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

_FEATSHARP_STATE = {"available": False}


def _ensure_featsharp_importable(lib_path: Optional[str] = None):
    """Make the ``featsharp`` package importable.

    Tries a regular import first.  If that fails and *lib_path* is given
    (e.g. ``".../evfm/libs/FeatUp"``), inserts it into ``sys.path`` and
    retries.
    """
    if _FEATSHARP_STATE["available"]:
        return

    if importlib.util.find_spec("featsharp") is not None:
        _FEATSHARP_STATE["available"] = True
        return

    if lib_path and os.path.isdir(lib_path):
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        if importlib.util.find_spec("featsharp") is not None:
            _FEATSHARP_STATE["available"] = True
            logger.info(f"Loaded featsharp from {lib_path}")
            return

    raise ImportError(
        "Cannot import 'featsharp'. Either install it as a package or set "
        "'featsharp_lib_path' in the teacher config to the directory "
        "containing the featsharp package (e.g. '.../evfm/libs/FeatUp')."
    )


class _TeacherFeaturizer(nn.Module):
    """Thin wrapper that adapts a TAO teacher model's forward to the
    featurizer interface expected by FeatSharp's ``load_from_file``.

    FeatSharp expects:
      - ``.input_size``: the spatial resolution of the input image (e.g. 224)
      - ``.patch_size``: the ViT patch size (e.g. 16)
      - ``.embed_dim``: the feature channel dimension
      - ``__call__(x, return_summary=True)`` → ``(summary, features_BCHW)``
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        input_size: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.model = teacher_model
        self.input_size = input_size
        self._patch_size = patch_size
        self._embed_dim = embed_dim

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def input_conditioner(self):
        return None

    @input_conditioner.setter
    def input_conditioner(self, v):
        pass

    def forward(self, x: torch.Tensor, return_summary: bool = False):
        summary, features = self.model(x, return_features=True)
        if features.dim() == 3:
            h = x.shape[-2] // self._patch_size
            w = x.shape[-1] // self._patch_size
            features = rearrange(features, "b (h w) c -> b c h w", h=h, w=w)
        if return_summary:
            return summary, features
        return features


class FeatSharpTeacher(nn.Module):
    """Teacher model wrapped with FeatSharp spatial upsampling.

    Drop-in replacement for a teacher backbone: ``forward(x,
    return_features=True)`` returns ``(summary, high_res_BCHW)`` where
    the spatial features have been upsampled by FeatSharp.

    Args:
        teacher_model: Original teacher backbone (SigLIP2, DINOv3, RADIO, etc.).
        checkpoint_path: Path to a pre-trained FeatSharp checkpoint for this teacher.
        input_size: Spatial resolution of the input image (e.g. 224, 384).
            Required so FeatSharp can build the correct positional grids.
        do_upsample: If True, load the learned FeatSharp upsampler. If False,
            use identity (only normalizer/bias from checkpoint).
        featsharp_lib_path: Optional path to the directory containing the
            ``featsharp`` package.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        checkpoint_path: str,
        input_size: int = 224,
        do_upsample: bool = True,
        featsharp_lib_path: Optional[str] = None,
    ):
        super().__init__()

        _ensure_featsharp_importable(featsharp_lib_path)
        from featsharp.builder import load_from_file

        self._teacher = teacher_model
        patch_size = teacher_model.patch_size
        embed_dim = _get_embed_dim(teacher_model)

        featurizer = _TeacherFeaturizer(
            teacher_model, input_size=input_size,
            patch_size=patch_size, embed_dim=embed_dim,
        )

        self.upsampler = load_from_file(
            checkpoint_path, featurizer, do_upsample=do_upsample
        )

        logger.info(
            f"FeatSharp adaptor applied: input_size={input_size}, "
            f"patch_size={patch_size}, embed_dim={embed_dim}, "
            f"do_upsample={do_upsample}, checkpoint={checkpoint_path}"
        )

    # ---- Proxy properties so the loss module can query dims ----

    @property
    def num_features(self):
        """Return the per-patch feature dimension of the wrapped teacher."""
        return self._teacher.num_features

    @property
    def patch_size(self):
        """Return the ViT patch size of the wrapped teacher."""
        return self._teacher.patch_size

    @property
    def summary_idxs(self):
        """Return the teacher's summary feature indices if available."""
        return getattr(self._teacher, "summary_idxs", None)

    def get_classifier(self):
        """Delegate ``get_classifier`` to the wrapped teacher model."""
        return self._teacher.get_classifier()

    def reset_classifier(self, num_classes, global_pool=""):
        """Delegate ``reset_classifier`` to the wrapped teacher model."""
        self._teacher.reset_classifier(num_classes, global_pool)

    # ---- Forward ----

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run the teacher forward pass, optionally returning upsampled features."""
        if return_features:
            _, hr_y, summary = self.upsampler(
                x, denormalize=True, return_summary=True
            )
            return summary, hr_y
        summary, _ = self._teacher(x, return_features=True)
        if return_logits:
            return summary
        head = self._teacher.get_classifier()
        return head(summary)

    def forward_pre_logits(self, x: torch.Tensor):
        """Return the teacher's pre-logit summary feature."""
        summary, _ = self._teacher(x, return_features=True)
        return summary

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Return the upsampled high-resolution feature pyramid."""
        _, hr_y, _ = self.upsampler(x, denormalize=True, return_summary=True)
        return hr_y


def _get_embed_dim(model: nn.Module) -> int:
    """Resolve the per-patch embedding dimension of a teacher model."""
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    if hasattr(model, "num_features"):
        # For RADIO, num_features = embed_dim * len(summary_idxs)
        summary_idxs = getattr(model, "summary_idxs", None)
        if summary_idxs is not None:
            return model.num_features // len(summary_idxs)
        return model.num_features
    raise AttributeError(
        f"Cannot determine embed_dim for teacher model of type {type(model).__name__}"
    )


def wrap_teacher_with_featsharp(
    teacher_model: nn.Module,
    checkpoint_path: str,
    input_size: int = 224,
    do_upsample: bool = True,
    featsharp_lib_path: Optional[str] = None,
) -> FeatSharpTeacher:
    """Convenience function to wrap a teacher model with FeatSharp."""
    return FeatSharpTeacher(
        teacher_model,
        checkpoint_path=checkpoint_path,
        input_size=input_size,
        do_upsample=do_upsample,
        featsharp_lib_path=featsharp_lib_path,
    )
