# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FFS commercial-checkpoint loader utilities.

Covers the remap pipeline (`_strip_prefix`, `_apply_prefix_remap`,
`_apply_substring_remap`), payload coercion (`_extract_state_dict`), and
the public `load_ffs_pretrained` entrypoint, all using small synthetic
state-dicts so the tests run without touching the real FFS bp2 checkpoint.
"""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.ckpt_utils import (
    _PREFIX_REMAP_RULES,
    _SUBSTRING_REMAP_RULES,
    _OPTIONAL_MISSING_KEYS,
    _strip_prefix,
    _apply_prefix_remap,
    _apply_substring_remap,
    _extract_state_dict,
    _ensure_stub_modules,
    load_ffs_pretrained,
)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestStripPrefix:
    """`_strip_prefix` should drop the leading prefix exactly once per key."""

    def test_strips_ddp_module_prefix(self):
        sd = {'module.layer.weight': torch.zeros(1), 'module.layer.bias': torch.zeros(1)}
        out = _strip_prefix(sd, prefix='module.')
        assert set(out) == {'layer.weight', 'layer.bias'}

    def test_leaves_unprefixed_keys_unchanged(self):
        sd = {'module.a': torch.zeros(1), 'b.c': torch.zeros(1)}
        out = _strip_prefix(sd, prefix='module.')
        assert out['a'] is sd['module.a']
        assert out['b.c'] is sd['b.c']

    def test_empty_prefix_is_passthrough(self):
        sd = {'a': torch.zeros(1)}
        assert _strip_prefix(sd, prefix='') is sd


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestApplyPrefixRemap:
    """`_apply_prefix_remap` must apply longest-prefix-wins, detect collisions."""

    def test_simple_prefix_remap(self):
        sd = {'classifier.layers.0.weight': torch.zeros(1)}
        out = _apply_prefix_remap(sd, {'classifier.layers.': 'classifier.'})
        assert list(out) == ['classifier.0.weight']

    def test_longest_prefix_wins(self):
        # Both rules match 'feature.stages.1.downsample.x'; the longer prefix
        # 'feature.stages.1.downsample.' should win.
        sd = {'feature.stages.1.downsample.0.weight': torch.zeros(1)}
        rules = {
            'feature.stages.1.': 'feature.stages.1_OUT.',
            'feature.stages.1.downsample.': 'feature.downsample_layers.1.',
        }
        out = _apply_prefix_remap(sd, rules)
        assert list(out) == ['feature.downsample_layers.1.0.weight']

    def test_no_match_passthrough(self):
        sd = {'unrelated.x': torch.zeros(1)}
        out = _apply_prefix_remap(sd, {'foo.': 'bar.'})
        assert out == sd

    def test_empty_rules_passthrough(self):
        sd = {'a': torch.zeros(1)}
        assert _apply_prefix_remap(sd, {}) is sd

    def test_collision_raises(self):
        # Two source keys remap onto the same target key.
        sd = {'a.x': torch.zeros(1), 'b.x': torch.zeros(1)}
        with pytest.raises(ValueError, match='Prefix remap collision'):
            _apply_prefix_remap(sd, {'a.': 'c.', 'b.': 'c.'})


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestApplySubstringRemap:
    """`_apply_substring_remap` must replace tokens anywhere in the key."""

    def test_replaces_inner_token(self):
        sd = {'feature.stages.0.0.conv_dw.weight': torch.zeros(1)}
        out = _apply_substring_remap(sd, {'.conv_dw.': '.dwconv.'})
        assert list(out) == ['feature.stages.0.0.dwconv.weight']

    def test_multiple_rules_apply_in_order(self):
        sd = {'block.mlp.fc1.weight': torch.zeros(1),
              'block.mlp.fc2.weight': torch.zeros(1)}
        out = _apply_substring_remap(sd, {'.mlp.fc1.': '.pwconv1.',
                                          '.mlp.fc2.': '.pwconv2.'})
        assert set(out) == {'block.pwconv1.weight', 'block.pwconv2.weight'}

    def test_empty_rules_passthrough(self):
        sd = {'a': torch.zeros(1)}
        assert _apply_substring_remap(sd, {}) is sd

    def test_collision_raises(self):
        sd = {'foo.A.x': torch.zeros(1), 'foo.B.x': torch.zeros(1)}
        with pytest.raises(ValueError, match='Substring remap collision'):
            _apply_substring_remap(sd, {'.A.': '.X.', '.B.': '.X.'})


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestExtractStateDict:
    """`_extract_state_dict` should coerce 3 payload shapes into a plain dict."""

    def test_extracts_from_nn_module(self):
        mod = nn.Linear(2, 3)
        sd = _extract_state_dict(mod)
        assert isinstance(sd, dict)
        assert 'weight' in sd and 'bias' in sd

    def test_extracts_inner_state_dict_key(self):
        inner = {'a': torch.zeros(1)}
        out = _extract_state_dict({'state_dict': inner, 'epoch': 0})
        assert out is inner

    def test_extracts_inner_model_key(self):
        inner = {'a': torch.zeros(1)}
        out = _extract_state_dict({'model': inner})
        assert out is inner

    def test_returns_plain_dict_unchanged(self):
        sd = {'a': torch.zeros(1)}
        out = _extract_state_dict(sd)
        assert out is sd

    def test_raises_on_unsupported_type(self):
        with pytest.raises(TypeError, match='Unsupported ckpt payload type'):
            _extract_state_dict(42)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestEnsureStubModulesIdempotent:
    """`_ensure_stub_modules` must install the meta-path finder exactly once."""

    def test_repeated_calls_install_one_finder(self):
        import sys
        # Run twice; sys.meta_path length should not grow on the second call.
        _ensure_stub_modules()
        n_after_first = len(sys.meta_path)
        _ensure_stub_modules()
        _ensure_stub_modules()
        assert len(sys.meta_path) == n_after_first


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
class TestLoadFFSPretrained:
    """End-to-end `load_ffs_pretrained` against a synthetic state-dict ckpt.

    Builds a tiny `nn.Module` that mirrors a small slice of FFS key naming so
    the remap rules engage without instantiating the full FastFoundationStereo
    network (avoids pulling in the vit-large encoder).
    """

    def _save_synthetic_research_ckpt(self, tmp_path):
        """Synthesise a research-style state_dict (pre-remap key naming)."""
        sd = {
            # `module.` DDP prefix - to be stripped.
            'module.classifier.layers.0.weight': torch.zeros(2, 2),
            'module.classifier.layers.0.bias': torch.zeros(2),
            # Substring remap candidate (.conv_dw. -> .dwconv.).
            'module.feature.stages.0.0.conv_dw.weight': torch.zeros(2, 2, 1, 1),
            # Prefix remap candidate (corr_feature_att.layers. -> corr_feature_att.).
            'module.corr_feature_att.layers.0.weight': torch.zeros(2, 2),
            # Plain key, no remap needed.
            'module.proj_cmb.weight': torch.zeros(2, 2),
            # Source-side key that has no target (will appear as 'unexpected').
            'module.legacy_only.x': torch.zeros(2),
        }
        path = tmp_path / 'ckpt.pth'
        torch.save(sd, path)
        return str(path)

    def _build_target_module(self):
        """Build an nn.Module whose param names match the post-remap target shape."""
        target = nn.Module()
        # Post-remap names (after _strip_prefix + _PREFIX_REMAP_RULES + _SUBSTRING_REMAP_RULES).
        target.classifier = nn.ModuleList([nn.Linear(2, 2)])
        target.feature = nn.Module()
        target.feature.stages = nn.ModuleList([nn.ModuleList([nn.Module()])])
        target.feature.stages[0][0].dwconv = nn.Conv2d(2, 2, 1)
        target.corr_feature_att = nn.ModuleList([nn.Linear(2, 2)])
        target.proj_cmb = nn.Linear(2, 2)
        # `dx` is in _OPTIONAL_MISSING_KEYS — register it so the loader sees it
        # as a target buffer and routes a missing-key report there (not into
        # the hard 'missing' bucket).
        target.register_buffer('dx', torch.zeros(1))
        return target

    def test_remap_pipeline_loads_synthetic_ckpt(self, tmp_path):
        ckpt_path = self._save_synthetic_research_ckpt(tmp_path)
        target = self._build_target_module()

        report = load_ffs_pretrained(target, ckpt_path)

        # 'dx' should appear in optional_missing (registered as buffer, not in ckpt).
        assert 'dx' in report['optional_missing'], (
            f"'dx' should be in optional_missing; got {report}"
        )
        # 'dx' should NOT appear in the hard missing list.
        assert 'dx' not in report['missing']
        # 'legacy_only.x' has no target, so it must surface as 'unexpected'.
        assert 'legacy_only.x' in report['unexpected']

    def _save_synthetic_lightning_ckpt(self, tmp_path):
        """Synthesise a Lightning-style ckpt (outer 'state_dict' + ``model.`` prefix)."""
        # Same content as `_save_synthetic_research_ckpt` but with the extra
        # ``model.`` prefix that pl.LightningModule wrap adds, plus the outer
        # `state_dict` envelope that `Trainer.save_checkpoint` produces.
        inner = {
            'model.module.classifier.layers.0.weight': torch.zeros(2, 2),
            'model.module.classifier.layers.0.bias': torch.zeros(2),
            'model.module.feature.stages.0.0.conv_dw.weight': torch.zeros(2, 2, 1, 1),
            'model.module.corr_feature_att.layers.0.weight': torch.zeros(2, 2),
            'model.module.proj_cmb.weight': torch.zeros(2, 2),
            'model.module.legacy_only.x': torch.zeros(2),
        }
        path = tmp_path / 'lightning_ckpt.pth'
        torch.save({'state_dict': inner, 'epoch': 1, 'global_step': 100}, path)
        return str(path)

    def test_load_lightning_ckpt(self, tmp_path):
        """Lightning-wrapped ckpt with `model.` key prefix should load equivalently."""
        ckpt_path = self._save_synthetic_lightning_ckpt(tmp_path)
        target = self._build_target_module()

        report = load_ffs_pretrained(target, ckpt_path)

        # Same expectations as the research-ckpt test: 'dx' optional-missing,
        # 'legacy_only.x' surfaces as unexpected.
        assert 'dx' in report['optional_missing']
        assert 'dx' not in report['missing']
        assert 'legacy_only.x' in report['unexpected']

    def test_remap_rules_constants_well_formed(self):
        """Smoke check: the constant remap dicts should be non-empty dicts of str→str."""
        assert isinstance(_PREFIX_REMAP_RULES, dict)
        assert isinstance(_SUBSTRING_REMAP_RULES, dict)
        assert _PREFIX_REMAP_RULES, "_PREFIX_REMAP_RULES is empty"
        assert _SUBSTRING_REMAP_RULES, "_SUBSTRING_REMAP_RULES is empty"
        for k, v in _PREFIX_REMAP_RULES.items():
            assert isinstance(k, str) and isinstance(v, str)
        for k, v in _SUBSTRING_REMAP_RULES.items():
            assert isinstance(k, str) and isinstance(v, str)
        assert isinstance(_OPTIONAL_MISSING_KEYS, tuple)
        assert all(isinstance(k, str) for k in _OPTIONAL_MISSING_KEYS)
