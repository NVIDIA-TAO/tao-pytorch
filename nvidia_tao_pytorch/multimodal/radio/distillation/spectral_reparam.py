# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learnable-scale spectral reparametrization.

The parametrization learns a trainable gain on top of spectral
normalization: ``weight * (softplus(scale) + alpha) / sigma``. It supports
individual projection heads and a whole-backbone walker, including fused QKV
and SwiGLU weights.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

logger = logging.getLogger(__name__)

_EPS = 1e-5

try:
    from torch.nn.utils.parametrizations import _SpectralNorm
except ImportError:  # pragma: no cover - depends on torch version
    _SpectralNorm = None


def _snr_inner_forward(weight, sigma, scale, alpha, eps):
    """Return weight scaled by a learnable gain and inverse spectral norm."""
    dtype = weight.dtype
    effective_scale = F.softplus(scale) + alpha
    effective_scale = effective_scale.float() / (sigma.float() + eps)
    y = weight * effective_scale
    if dtype in (torch.float16, torch.bfloat16):
        y = y.to(dtype)
    return y


class LearnableScaleSpectralNorm(_SpectralNorm if _SpectralNorm is not None else nn.Module):
    """Spectral norm with a learnable softplus-parametrized scale."""

    def __init__(self, weight, n_power_iterations=1, dim=0, eps=1e-12, alpha=0.05,
                 init_norm_to_current=False):
        if _SpectralNorm is None:
            raise ImportError(
                "torch.nn.utils.parametrizations._SpectralNorm is unavailable in this torch "
                "version; learnable-scale spectral reparam requires it."
            )
        super().__init__(weight, n_power_iterations=n_power_iterations, dim=dim, eps=eps)
        self.alpha = alpha
        self._initialized = False
        init_value = self._initialize_scale(weight, init_norm_to_current)
        self.scale = nn.Parameter(init_value.to(device=weight.device))

    def _get_sigma(self, weight, n_power_iterations=None):
        if not n_power_iterations:
            n_power_iterations = self.n_power_iterations
        if weight.ndim == 1:
            return weight.norm()
        weight_mat = self._reshape_weight_to_matrix(weight)
        if self.training:
            self._power_method(weight_mat, n_power_iterations)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)
        return torch.dot(u, torch.mv(weight_mat, v))

    def _initialize_scale(self, weight, init_norm_to_current, n_power_iterations=20):
        if init_norm_to_current:
            init_scale = self._get_sigma(weight, n_power_iterations=n_power_iterations) + self.eps
        else:
            init_scale = torch.tensor(1.0, dtype=torch.float32)
        t = init_scale - self.alpha
        if t < _EPS:
            logger.warning(
                "Initialized spectral norm %s is too small for alpha=%s; clamping to %s.",
                init_scale, self.alpha, _EPS,
            )
            t = torch.tensor(_EPS, dtype=torch.float32)
        # Inverse-softplus init so that softplus(init_value) + alpha == init_scale.
        init_value = torch.log(torch.exp(t) - 1)
        self._initialized = True
        return init_value.reshape(1, 1)

    def forward(self, weight, *args, **kwargs):
        """Apply the learnable-scale spectral reparametrization to *weight* and return it."""
        if not self._initialized:
            self.scale.data.copy_(self._initialize_scale(weight, init_norm_to_current=True, n_power_iterations=200))
        sigma = self._get_sigma(weight)
        return _snr_inner_forward(weight, sigma, self.scale, self.alpha, self.eps)


def apply_learnable_spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12,
                                  alpha=0.05, init_norm_to_current=True):
    """Register a LearnableScaleSpectralNorm parametrization on module.<name>."""
    weight = getattr(module, name)
    register_parametrization(
        module, name,
        LearnableScaleSpectralNorm(
            weight, n_power_iterations=n_power_iterations, dim=0, eps=eps,
            alpha=alpha, init_norm_to_current=init_norm_to_current,
        ),
    )


# ---------------------------------------------------------------------------
# Backbone-scope spectral reparametrization.
# ---------------------------------------------------------------------------

try:  # timm is a tao dependency; used only for module-type detection (isinstance).
    from timm.models.vision_transformer import Attention as _TimmAttention
except Exception:  # pragma: no cover - defensive
    _TimmAttention = None


class _ChunkedSNReweight(nn.Module):
    """Split weight into ``num_chunks`` row-blocks and spectrally reparam each.

    This is used for fused SwiGLU ``w12`` weights. ``forward`` re-splits the
    current weight, applies each part's spectral parametrization, and
    concatenates the parts.
    """

    def __init__(self, weight, num_chunks, n_power_iterations=1, eps=1e-6,
                 alpha=0.05, init_norm_to_current=False):
        super().__init__()
        self.num_chunks = num_chunks
        parts = weight.split(weight.shape[0] // num_chunks, dim=0)
        self.parts = nn.ModuleList([
            LearnableScaleSpectralNorm(
                p, n_power_iterations=n_power_iterations, dim=0, eps=eps,
                alpha=alpha, init_norm_to_current=init_norm_to_current,
            )
            for p in parts
        ])

    def forward(self, weight, *args, **kwargs):
        parts = weight.split(weight.shape[0] // self.num_chunks, dim=0)
        return torch.cat([fn(p) for fn, p in zip(self.parts, parts)], dim=0)


class _AttnSNReweight(_ChunkedSNReweight):
    """Fused-qkv (3 chunks) spectral reparam; the values (k=chunk 2) may be left
    un-renormalized.
    """

    def __init__(self, weight, n_power_iterations=1, eps=1e-6, alpha=0.05,
                 init_norm_to_current=False, renorm_values=True):
        super().__init__(weight, 3, n_power_iterations=n_power_iterations, eps=eps,
                         alpha=alpha, init_norm_to_current=init_norm_to_current)
        if not renorm_values:
            self.parts[2] = nn.Identity()


def _parametrize(module, name, param):
    register_parametrization(module, name, param)


def enable_spectral_reparam(model, n_power_iterations=1, eps=1e-6,
                            init_norm_to_current=True, renorm_values=True,
                            renorm_mlp=True, renorm_proj=True, alpha=0.05):
    """Apply learnable spectral reparam across a model's Linear layers.

      - timm ``Attention`` (or any ``*.attn``) with a fused ``qkv`` -> _AttnSNReweight
        (3-chunk; values left un-renormed when renorm_values=False), ``proj`` -> Linear reparam.
      - ``*mlp`` with fused SwiGLU ``w12`` -> _ChunkedSNReweight(2), ``w3`` -> Linear reparam.
      - any other ``nn.Linear`` (except ``patch_generator``) -> LearnableScaleSpectralNorm.
    ``visited_prefixes`` prevents double-wrapping the qkv/proj/w12/w3 Linears via
    the generic branch. Returns the count of parametrized weights.
    """
    if isinstance(model, (list, tuple)):
        total = 0
        for sub in model:
            total += enable_spectral_reparam(
                sub, n_power_iterations=n_power_iterations, eps=eps,
                init_norm_to_current=init_norm_to_current, renorm_values=renorm_values,
                renorm_mlp=renorm_mlp, renorm_proj=renorm_proj, alpha=alpha,
            )
        return total

    args = dict(n_power_iterations=n_power_iterations, eps=eps, alpha=alpha,
                init_norm_to_current=init_norm_to_current)
    visited_prefixes = set()
    count = 0

    def _linear(linear):
        _parametrize(linear, "weight", LearnableScaleSpectralNorm(linear.weight, dim=0, **args))

    for name, mod in model.named_modules():
        pref = ".".join(name.split(".")[:-1])
        if pref in visited_prefixes:
            continue

        is_attn = (_TimmAttention is not None and isinstance(mod, _TimmAttention)) or name.endswith(".attn")
        if is_attn:
            if hasattr(mod, "qkv") and isinstance(mod.qkv, nn.Linear):
                _parametrize(mod.qkv, "weight",
                             _AttnSNReweight(mod.qkv.weight, renorm_values=renorm_values, **args))
                count += 1
            if renorm_proj and hasattr(mod, "proj") and isinstance(mod.proj, nn.Linear):
                _linear(mod.proj)
                count += 1
            visited_prefixes.add(name)
        elif name.endswith("mlp") and renorm_mlp and hasattr(mod, "w12"):
            _parametrize(mod.w12, "weight", _ChunkedSNReweight(mod.w12.weight, num_chunks=2, **args))
            if hasattr(mod, "w3"):
                _linear(mod.w3)
            count += 1
            visited_prefixes.add(name)
        elif isinstance(mod, nn.Linear) and "patch_generator" not in name:
            _linear(mod)
            count += 1

    logger.info("Spectral reparam: parametrized %d weight tensors.", count)
    return count
