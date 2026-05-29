# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-path FFS checkpoint loader.

Why: FFS commercial ckpt is a research ``*_serialize.pth`` that pickles an
entire ``nn.Module`` instance. The pickle stream references the research
source tree as a top-level package ``core`` plus a flat ``Utils`` module.
TAO does not vendor those, so we install a meta-path finder that lazily
synthesises empty ``nn.Module`` subclasses on demand. After unpickling,
only ``state_dict()`` is consulted — that walks ``_parameters`` /
``_buffers`` / ``_modules`` regardless of class identity.

Beyond the pickle stub, three load-time bridges are needed against the
TAO ``FastFoundationStereo`` wrapper. These cover only the slots that
cannot be expressed via the configurable knobs on the wrapper alone:

  - Prefix rename ``cost_agg.atts.4.* -> cost_agg.atts.cost_vol_disp.*``:
    research FFS keys the cost-volume attention by the integer ``'4'``;
    TAO ``HourGlass`` keys the same module by the string ``'cost_vol_disp'``.
    The ``HourGlass`` knobs control divisor and forward order but not
    the dict key name, so we remap at load.
  - Optional missing keys (``dx``): the wrapper registers ``dx`` as an
    int8 buffer (so device moves work via ``.to``); research FFS stores
    it as a plain attribute and the ckpt has no entry. ``strict=False``
    + reporting under ``optional_missing`` keeps the loader honest.
  - ``.bn`` / ``.IN`` -> ``.norm`` rewiring at every TAO ``Conv``: handled
    automatically by the ``_remap_norm_keys`` pre-hook registered on
    each ``Conv`` instance — this loader does not need to touch it.
"""

import importlib.abc
import importlib.machinery
import sys
import types
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


# Research FFS keys cost_agg.atts by the integer '4'; TAO HourGlass keys
# the same module 'cost_vol_disp'. Prefix-based so all four state slots
# (``in_proj_*``, ``out_proj.*``, ``norm*.*``, ``self_attn.*``) remap.
#
# corr_feature_att / classifier in the ckpt are wrapped under ``.layers``;
# the wrapper uses inline ``nn.Sequential`` so the wrapper key prefix is
# ``corr_feature_att.X.*`` / ``classifier.X.*`` (no ``.layers.`` segment).
#
# Feature backbone (timm edgenext_small in ckpt vs TAO internal EdgeNeXt
# in wrapper) needs four prefix rules to map ckpt's ``feature.stem.*`` and
# ``feature.stages.X.{downsample,blocks}.*`` to TAO's
# ``feature.downsample_layers.X.*`` and ``feature.stages.X.*``. Inner block
# attribute names (timm ``conv_dw``/``mlp.fc1``/``mlp.fc2`` vs TAO
# ``dwconv``/``pwconv1``/``pwconv2``) are handled by ``_SUBSTRING_REMAP_RULES``.
_PREFIX_REMAP_RULES: Dict[str, str] = {
    'cost_agg.atts.4.': 'cost_agg.atts.cost_vol_disp.',
    'corr_feature_att.layers.': 'corr_feature_att.',
    'classifier.layers.': 'classifier.',
    'feature.stem.': 'feature.downsample_layers.0.',
    'feature.stages.1.downsample.': 'feature.downsample_layers.1.',
    'feature.stages.2.downsample.': 'feature.downsample_layers.2.',
    'feature.stages.3.downsample.': 'feature.downsample_layers.3.',
    'feature.stages.0.blocks.': 'feature.stages.0.',
    'feature.stages.1.blocks.': 'feature.stages.1.',
    'feature.stages.2.blocks.': 'feature.stages.2.',
    'feature.stages.3.blocks.': 'feature.stages.3.',
}

# Substring rules applied AFTER prefix rules. timm-edgenext-small uses
# ``conv_dw`` / ``mlp.fc1`` / ``mlp.fc2`` for the depth-wise + pointwise
# convs inside an EdgeNeXt block; TAO's internal ``EdgeNextConvEncoder``
# uses ``dwconv`` / ``pwconv1`` / ``pwconv2``. These tokens appear inside
# the key (not at the prefix) so they need ``str.replace`` rather than
# ``startswith``.
_SUBSTRING_REMAP_RULES: Dict[str, str] = {
    '.conv_dw.': '.dwconv.',
    '.mlp.fc1.': '.pwconv1.',
    '.mlp.fc2.': '.pwconv2.',
}

# Buffers/parameters that ``FastFoundationStereo`` registers but the
# research FFS ckpt does not contain. ``strict=False`` would silently let
# any missing key through; this list is the explicit whitelist.
_OPTIONAL_MISSING_KEYS: Tuple[str, ...] = (
    'dx',
)


def _strip_prefix(sd: Dict[str, torch.Tensor],
                  prefix: str = 'module.') -> Dict[str, torch.Tensor]:
    """Drop a leading prefix on every key (e.g. DDP's ``module.``)."""
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        out[k] = v
    return out


def _apply_prefix_remap(sd: Dict[str, torch.Tensor],
                        rules: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Rename keys whose prefix matches an entry in ``rules`` (most-specific wins).

    Rules are tried in descending prefix-length order, so a longer (more
    specific) prefix wins over a shorter one even if both match. Raises
    ValueError on collision — i.e. the remap produces a key that the output
    already holds.
    """
    if not rules:
        return sd
    sorted_rules = sorted(rules.items(), key=lambda kv: -len(kv[0]))
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        for old, new in sorted_rules:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        if k in out:
            raise ValueError(
                f"Prefix remap collision on key {k!r}; check _PREFIX_REMAP_RULES.")
        out[k] = v
    return out


def _apply_substring_remap(sd: Dict[str, torch.Tensor],
                           rules: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Replace every occurrence of each ``rules`` key inside the state-dict keys.

    Used for inner-block attribute renames where the token appears mid-key
    (e.g. timm-style ``.conv_dw.`` -> TAO ``.dwconv.``). Rules apply in
    insertion order; collisions raise ValueError.
    """
    if not rules:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        for old, new in rules.items():
            k = k.replace(old, new)
        if k in out:
            raise ValueError(
                f"Substring remap collision on key {k!r}; check _SUBSTRING_REMAP_RULES.")
        out[k] = v
    return out


class _StubModuleType(types.ModuleType):
    """Module type whose attribute lookup synthesises ``nn.Module`` subclasses.

    Why: pickle's ``find_class`` does ``getattr(sys.modules[mod], name)``;
    for missing FFS classes we want to return a class object, not raise.
    Caching the synthesised class on the module preserves identity, which
    matters for pickle's class-identity bookkeeping.
    """

    def __getattr__(self, attr_name):  # noqa: D401
        if attr_name.startswith('__'):
            # Why: dunder lookups (e.g. ``__path__``, ``__loader__``) must
            # raise AttributeError so import machinery behaves normally.
            raise AttributeError(attr_name)
        cls = type(attr_name, (nn.Module,), {})
        cls.__module__ = self.__name__
        setattr(self, attr_name, cls)
        return cls


# Why: only stub modules whose ROOT name is in this set; avoids accidentally
# stubbing real packages that happen to be missing for unrelated reasons.
_STUB_ROOTS: Iterable[str] = (
    'core',
    'Utils',
    'foundation_stereo_ori',
    'depth_anything',
    'dinov2',
    'dpt',
)


class _StubLoader(importlib.abc.Loader):
    """Loader that produces ``_StubModuleType`` modules — used by ``_StubFinder``."""

    def create_module(self, spec):  # noqa: D401
        return _StubModuleType(spec.name)

    def exec_module(self, module):  # noqa: D401
        return None


class _StubFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that fabricates packages for FFS research roots."""

    def __init__(self, stub_roots: Iterable[str]):
        self._stub_roots = tuple(stub_roots)

    def find_spec(self, fullname, path, target=None):  # noqa: D401
        root = fullname.split('.', 1)[0]
        if root not in self._stub_roots:
            return None
        spec = importlib.machinery.ModuleSpec(fullname, _StubLoader())
        # Why: marking with submodule_search_locations makes Python treat
        # the synthesised module as a package, which is what pickle expects
        # when it walks dotted names like ``core.foundation_stereo``.
        spec.submodule_search_locations = []
        return spec


_FINDER_STATE: Dict[str, bool] = {"installed": False}


def _ensure_stub_modules() -> None:
    """Install ``_StubFinder`` on ``sys.meta_path`` exactly once.

    Why: idempotent so repeated ``load_ffs_pretrained`` calls don't stack
    multiple finders on ``sys.meta_path``.
    """
    if _FINDER_STATE["installed"]:
        return
    sys.meta_path.append(_StubFinder(_STUB_ROOTS))
    _FINDER_STATE["installed"] = True


def _extract_state_dict(loaded) -> Dict[str, torch.Tensor]:
    """Coerce a ``torch.load`` result into a plain state_dict.

    Args:
        loaded: payload from ``torch.load(...)`` — may be an ``nn.Module``
            (research ``*_serialize.pth``), a wrapper dict, or already a
            state_dict.

    Returns:
        plain ``Dict[str, torch.Tensor]``.
    """
    if isinstance(loaded, nn.Module):
        return loaded.state_dict()
    if isinstance(loaded, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            inner = loaded.get(key)
            if isinstance(inner, dict):
                return inner
        return loaded
    raise TypeError(f'Unsupported ckpt payload type: {type(loaded)}')


def load_ffs_pretrained(model: nn.Module,
                        ckpt_path: str) -> Dict[str, List[str]]:
    """Load an FFS commercial checkpoint into a ``FastFoundationStereo``.

    Args:
        model: target ``FastFoundationStereo`` (or any nn.Module with a
            compatible state_dict layout).
        ckpt_path: path to a FFS checkpoint file.

    Returns:
        ``{'missing': [...], 'unexpected': [...], 'optional_missing': [...]}``.
        ``missing`` excludes keys in ``_OPTIONAL_MISSING_KEYS`` (those are
        reported under ``optional_missing`` instead).
    """
    _ensure_stub_modules()
    # weights_only=False is intentional — research FFS ckpt pickles an
    # entire nn.Module instance, not a state_dict.
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = _extract_state_dict(loaded)
    # Lightning saves a LightningModule whose `self.model` holds the actual
    # FastFoundationStereo, so its state_dict keys carry an extra ``model.``
    # prefix. Strip this OUTER wrap before the DDP `module.` wrap so a
    # Lightning+DDP ckpt (`model.module.x`) reduces correctly. Pure research
    # bp2 ckpts have no `model.` prefix, so this strip is a no-op on them.
    if any(k.startswith('model.') for k in sd):
        sd = _strip_prefix(sd, prefix='model.')
    sd = _strip_prefix(sd, prefix='module.')
    sd = _apply_prefix_remap(sd, _PREFIX_REMAP_RULES)
    sd = _apply_substring_remap(sd, _SUBSTRING_REMAP_RULES)
    result = model.load_state_dict(sd, strict=False)

    missing: List[str] = []
    optional_missing: List[str] = []
    for k in result.missing_keys:
        (optional_missing if k in _OPTIONAL_MISSING_KEYS else missing).append(k)
    return {'missing': missing,
            'unexpected': list(result.unexpected_keys),
            'optional_missing': optional_missing}
