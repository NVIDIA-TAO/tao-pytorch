# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP LoRA and preservation-loss primitives."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.clip.model.lora import (
    LoRALinear,
    inject_lora,
    merge_lora,
    resolve_tower_mode,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import build_optimizer


class _AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.other = nn.Linear(4, 4, bias=False)

    def forward(self, values):
        return self.k_proj(self.q_proj(values)) + self.other(values)


class _TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_blocks = nn.ModuleList([_AttentionBlock(), _AttentionBlock()])
        self.text_blocks = nn.ModuleList([_AttentionBlock(), _AttentionBlock()])
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def get_encoder_blocks(self, tower):
        return self.vision_blocks if tower == 'vision' else self.text_blocks

    def named_logit_parameters(self):
        yield 'logit_scale', self.logit_scale

    def vision_named_parameters(self):
        yield from (
            (f'vision_blocks.{name}', parameter)
            for name, parameter in self.vision_blocks.named_parameters()
        )

    def text_named_parameters(self):
        yield from (
            (f'text_blocks.{name}', parameter)
            for name, parameter in self.text_blocks.named_parameters()
        )

    def forward(self, image=None, text=None):
        image_features = F.normalize(self.vision_blocks[-1](image), dim=-1)
        text_features = F.normalize(self.text_blocks[-1](text), dim=-1)
        return image_features, text_features, self.logit_scale.exp()


def _peft_config():
    tower = SimpleNamespace(
        mode='lora',
        target_modules=['q_proj', 'k_proj'],
        num_last_blocks=1,
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    return SimpleNamespace(enabled=True, vision=tower, text=tower)


def _tower(mode, targets=None):
    return SimpleNamespace(
        mode=mode,
        target_modules=targets or ['q_proj', 'k_proj'],
        num_last_blocks=1,
        rank=2,
        alpha=4,
        dropout=0.0,
    )


def _legacy_tower(enabled):
    return SimpleNamespace(
        enabled=enabled,
        target_modules=['q_proj', 'k_proj'],
        num_last_blocks=1,
        rank=2,
        alpha=4,
        dropout=0.0,
    )


def _hybrid_config(vision_mode, text_mode, calibration=False):
    return SimpleNamespace(
        enabled=True,
        train_logit_calibration=calibration,
        vision=_tower(vision_mode),
        text=_tower(text_mode),
    )


def _optimizer_config(vision_lr=1e-6, text_lr=1e-4):
    return SimpleNamespace(
        optim=SimpleNamespace(
            optimizer_type='adamw', vision_lr=vision_lr, text_lr=text_lr,
            weight_decay=0.1, betas=[0.9, 0.98], eps=1e-8,
            warmup_steps=0, scheduler='constant',
        )
    )


def test_inject_lora_limits_adaptation_to_requested_final_blocks():
    """Only configured final block projections and calibration stay trainable."""
    model = _TinyCLIP()
    stats = inject_lora(model, _peft_config())

    assert isinstance(model.vision_blocks[1].q_proj, LoRALinear)
    assert isinstance(model.text_blocks[1].k_proj, LoRALinear)
    assert isinstance(model.vision_blocks[0].q_proj, nn.Linear)
    assert isinstance(model.vision_blocks[1].other, nn.Linear)
    assert len(stats['injected_modules']) == 4
    assert model.logit_scale.requires_grad
    assert all(
        parameter.requires_grad == ('lora_' in name or name == 'logit_scale')
        for name, parameter in model.named_parameters()
    )


def test_inject_lora_is_idempotent():
    """A repeated injection keeps existing adapters in the optimizer."""
    model = _TinyCLIP()
    config = _peft_config()
    first = inject_lora(model, config)
    second = inject_lora(model, config)

    assert len(first['injected_modules']) == 4
    assert second['injected_modules'] == []
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if 'lora_' in name
    )


@pytest.mark.parametrize(
    ('vision_mode', 'text_mode', 'vision_trainable', 'text_trainable'),
    [
        ('full', 'lora', True, False),
        ('lora', 'full', False, True),
        ('lora', 'lora', False, False),
        ('frozen', 'lora', False, False),
        ('full', 'frozen', True, False),
    ],
)
def test_per_tower_modes_enable_only_the_requested_parameters(
    vision_mode, text_mode, vision_trainable, text_trainable,
):
    """Hybrid modes give each tower exactly its requested trainability."""
    model = _TinyCLIP()
    stats = inject_lora(model, _hybrid_config(vision_mode, text_mode))

    assert stats['requested_modes'] == {
        'vision': vision_mode, 'text': text_mode,
    }
    vision = dict(model.vision_named_parameters())
    text = dict(model.text_named_parameters())
    if vision_mode == 'lora':
        assert any('lora_A' in name and value.requires_grad for name, value in vision.items())
        assert all(
            not value.requires_grad for name, value in vision.items()
            if 'lora_' not in name
        )
    else:
        assert all(value.requires_grad == vision_trainable for value in vision.values())
    if text_mode == 'lora':
        assert any('lora_A' in name and value.requires_grad for name, value in text.items())
        assert all(
            not value.requires_grad for name, value in text.items()
            if 'lora_' not in name
        )
    else:
        assert all(value.requires_grad == text_trainable for value in text.values())
    assert not model.logit_scale.requires_grad


def test_legacy_enabled_only_tower_config_preserves_lora_behavior():
    """Existing enabled-only YAMLs resolve to the equivalent tower modes."""
    model = _TinyCLIP()
    stats = inject_lora(model, SimpleNamespace(
        enabled=True,
        train_logit_calibration=False,
        vision=_legacy_tower(True),
        text=_legacy_tower(False),
    ))

    assert stats['requested_modes'] == {'vision': 'lora', 'text': 'frozen'}
    assert any('lora_A' in name for name, _ in model.vision_named_parameters())
    assert not any('lora_A' in name for name, _ in model.text_named_parameters())


@pytest.mark.parametrize(
    ('mode', 'enabled'),
    [('lora', False), ('frozen', True), ('full', False), ('full', True)],
)
def test_explicit_mode_rejects_conflicting_legacy_enabled(mode, enabled):
    """A supplied legacy switch cannot silently contradict an explicit mode."""
    with pytest.raises(ValueError, match='conflicts with legacy'):
        resolve_tower_mode(SimpleNamespace(mode=mode, enabled=enabled), 'vision')


@pytest.mark.parametrize(('mode', 'enabled'), [('lora', True), ('frozen', False)])
def test_explicit_mode_accepts_equivalent_legacy_enabled(mode, enabled):
    assert resolve_tower_mode(SimpleNamespace(mode=mode, enabled=enabled), 'vision') == mode


def test_lora_with_no_matching_projection_is_rejected():
    """A typo in target modules must not create a silently frozen PEFT run."""
    with pytest.raises(ValueError, match='injected zero modules'):
        inject_lora(_TinyCLIP(), SimpleNamespace(
            enabled=True, train_logit_calibration=False,
            vision=_tower('lora', targets=['does_not_exist']),
            text=_tower('frozen'),
        ))


@pytest.mark.parametrize('num_last_blocks', [-1, 3])
def test_lora_rejects_block_counts_outside_the_tower_depth(num_last_blocks):
    """LoRA depth must be zero/all or within the encoder block count."""
    config = _hybrid_config('lora', 'frozen', calibration=False)
    config.vision.num_last_blocks = num_last_blocks
    with pytest.raises(ValueError, match='num_last_blocks must be between'):
        inject_lora(_TinyCLIP(), config)


@pytest.mark.parametrize('calibration', [False, True])
def test_logit_calibration_is_controlled_independently(calibration):
    """Logit parameters follow train_logit_calibration, not tower mode."""
    model = _TinyCLIP()
    inject_lora(model, _hybrid_config('frozen', 'lora', calibration))
    assert model.logit_scale.requires_grad is calibration


def test_peft_rejects_no_trainable_tower_or_calibration():
    """An enabled PEFT config cannot silently produce an empty optimizer."""
    with pytest.raises(ValueError, match='no tower parameters are trainable'):
        inject_lora(_TinyCLIP(), _hybrid_config('frozen', 'frozen', False))


def test_optimizer_groups_keep_towers_disjoint_and_use_per_tower_lrs():
    """Hybrid LoRA keeps full vision and text adapter learning rates separate."""
    model = _TinyCLIP()
    inject_lora(model, _hybrid_config('full', 'lora', calibration=True))
    optimizer = build_optimizer(model, _optimizer_config())
    parameter_towers = {
        id(parameter): 'vision' for _, parameter in model.vision_named_parameters()
        if parameter.requires_grad
    }
    parameter_towers.update({
        id(parameter): 'text' for _, parameter in model.text_named_parameters()
        if parameter.requires_grad
    })
    parameter_towers[id(model.logit_scale)] = 'logit'
    grouped_ids = set()
    for group in optimizer.param_groups:
        assert group['lr'] == pytest.approx(
            1e-6 if group['_tower'] == 'vision' else 1e-4
        )
        for parameter in group['params']:
            assert parameter_towers[id(parameter)] == group['_tower']
            assert id(parameter) not in grouped_ids
            grouped_ids.add(id(parameter))
    assert grouped_ids == set(parameter_towers)


def test_frozen_base_parameters_remain_identical_after_optimizer_step():
    """A LoRA tower's frozen base weights cannot change during optimization."""
    model = _TinyCLIP()
    inject_lora(model, _hybrid_config('lora', 'frozen', calibration=False))
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.vision_named_parameters()
        if 'lora_' not in name
    }
    optimizer = build_optimizer(model, _optimizer_config())
    image, text = torch.randn(2, 4), torch.randn(2, 4)
    loss = model(image=image, text=text)[0].sum()
    loss.backward()
    optimizer.step()
    for name, parameter in model.vision_named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(parameter, frozen_before[name])


def test_merge_lora_preserves_adapted_output_and_removes_wrappers():
    """Merged export weights produce the same output without LoRA modules."""
    torch.manual_seed(1)
    model = _TinyCLIP().eval()
    inject_lora(model, _peft_config())
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B)
    image, text = torch.randn(2, 4), torch.randn(2, 4)
    before = model(image=image, text=text)

    assert merge_lora(model) == 4
    after = model(image=image, text=text)

    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    for before_value, after_value in zip(before, after):
        torch.testing.assert_close(before_value, after_value)


def test_lora_zero_initialization_preserves_base_output():
    """B=0 makes LoRA injection an exact no-op before the first optimizer step."""
    torch.manual_seed(7)
    model = _TinyCLIP().eval()
    image, text = torch.randn(2, 4), torch.randn(2, 4)
    before = model(image=image, text=text)
    inject_lora(model, _hybrid_config('lora', 'lora'))
    after = model(image=image, text=text)
    for before_value, after_value in zip(before, after):
        torch.testing.assert_close(before_value, after_value)


def test_lora_checkpoint_state_restores_exactly(tmp_path):
    """LoRA adapter state survives checkpoint-style save and resume loading."""
    torch.manual_seed(11)
    model = _TinyCLIP().eval()
    config = _hybrid_config('frozen', 'lora')
    inject_lora(model, config)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B)
    checkpoint = tmp_path / 'lora.ckpt'
    torch.save(model.state_dict(), checkpoint)

    resumed = _TinyCLIP().eval()
    inject_lora(resumed, config)
    resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    image, text = torch.randn(2, 4), torch.randn(2, 4)
    for actual, expected in zip(
        resumed(image=image, text=text), model(image=image, text=text)
    ):
        torch.testing.assert_close(actual, expected)
