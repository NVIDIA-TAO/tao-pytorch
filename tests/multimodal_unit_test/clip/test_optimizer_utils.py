# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP optimizer utilities."""

import math

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    _is_bias_or_norm,
    _make_tower_groups,
    _TowerCfg,
    compute_lr,
    build_optimizer,
    VALID_OPTIMIZER_TYPES,
    VALID_SCHEDULERS,
)


@pytest.mark.multimodal_unit
class TestIsBiasOrNorm:
    """Test _is_bias_or_norm helper function."""

    def test_bias_parameter(self):
        """Test that bias parameters are identified."""
        param = torch.zeros(512)  # 1D tensor
        assert _is_bias_or_norm("layer.bias", param) is True

    def test_weight_parameter(self):
        """Test that weight parameters are not identified as bias/norm."""
        param = torch.zeros(512, 512)  # 2D tensor
        assert _is_bias_or_norm("layer.weight", param) is False

    def test_ln_parameter(self):
        """Test that layer norm parameters are identified."""
        param = torch.zeros(512, 512)  # 2D tensor
        assert _is_bias_or_norm("ln_1.weight", param) is True

    def test_bn_parameter(self):
        """Test that batch norm parameters are identified."""
        param = torch.zeros(512, 512)
        assert _is_bias_or_norm("bn1.weight", param) is True

    def test_logit_scale_parameter(self):
        """Test that logit_scale is identified."""
        param = torch.zeros(1)
        assert _is_bias_or_norm("logit_scale", param) is True

    def test_1d_parameter(self):
        """Test that 1D parameters are identified as bias/norm."""
        param = torch.zeros(512)
        assert _is_bias_or_norm("some_layer.weight", param) is True

    def test_scalar_parameter(self):
        """Test that scalar parameters are identified as bias/norm."""
        param = torch.zeros([])
        assert _is_bias_or_norm("scalar_param", param) is True


@pytest.mark.multimodal_unit
class TestTowerCfg:
    """Test _TowerCfg config holder."""

    def test_initialization(self):
        """Test TowerCfg initialization."""
        cfg = _TowerCfg(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8)
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.betas == [0.9, 0.999]
        assert cfg.eps == 1e-8

    def test_slots(self):
        """Test that TowerCfg uses __slots__."""
        cfg = _TowerCfg(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8)
        with pytest.raises(AttributeError):
            cfg.unknown_attr = "test"


@pytest.mark.multimodal_unit
class TestMakeTowerGroups:
    """Test _make_tower_groups function."""

    def test_separates_bias_and_rest(self):
        """Test that bias/norm and rest parameters are separated."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
        )
        named_params = list(model.named_parameters())
        cfg = _TowerCfg(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8)

        groups = _make_tower_groups(named_params, cfg, 'test')

        # Should have two groups: bias_norm and rest
        assert len(groups) == 2

        # Find bias_norm group
        bias_norm_group = next(g for g in groups if g['_is_bias_norm'])
        rest_group = next(g for g in groups if not g['_is_bias_norm'])

        # Bias/norm group should have no weight decay
        assert bias_norm_group['weight_decay'] == 0.0
        # Rest group should have weight decay
        assert rest_group['weight_decay'] == 0.01

    def test_tower_label(self):
        """Test that tower label is correctly assigned."""
        model = nn.Linear(10, 10)
        named_params = list(model.named_parameters())
        cfg = _TowerCfg(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8)

        groups = _make_tower_groups(named_params, cfg, 'vision')

        for group in groups:
            assert group['_tower'] == 'vision'

    def test_skips_non_trainable(self):
        """Test that non-trainable parameters are skipped."""
        model = nn.Linear(10, 10)
        for param in model.parameters():
            param.requires_grad = False

        named_params = list(model.named_parameters())
        cfg = _TowerCfg(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8)

        groups = _make_tower_groups(named_params, cfg, 'test')

        # No groups should be created
        assert len(groups) == 0

    def test_lr_and_betas_copied(self):
        """Test that LR and betas are copied to groups."""
        model = nn.Linear(10, 10)
        named_params = list(model.named_parameters())
        cfg = _TowerCfg(lr=5e-5, weight_decay=0.01, betas=[0.9, 0.95], eps=1e-6)

        groups = _make_tower_groups(named_params, cfg, 'test')

        for group in groups:
            assert group['lr'] == 5e-5
            assert group['betas'] == (0.9, 0.95)
            assert group['eps'] == 1e-6


@pytest.mark.multimodal_unit
class TestComputeLR:
    """Test compute_lr function."""

    def test_warmup_phase(self):
        """Test LR during warmup phase."""
        base_lr = 1e-4
        warmup_steps = 100
        max_steps = 1000

        # At step 0, LR should be base_lr * 1/100
        lr = compute_lr(0, base_lr, warmup_steps, max_steps)
        assert pytest.approx(lr, rel=1e-5) == base_lr * 1 / 100

        # At step 49, LR should be base_lr * 50/100
        lr = compute_lr(49, base_lr, warmup_steps, max_steps)
        assert pytest.approx(lr, rel=1e-5) == base_lr * 50 / 100

        # At step 99, LR should be base_lr * 100/100 = base_lr
        lr = compute_lr(99, base_lr, warmup_steps, max_steps)
        assert pytest.approx(lr, rel=1e-5) == base_lr

    def test_constant_scheduler(self):
        """Test constant LR scheduler after warmup."""
        base_lr = 1e-4
        warmup_steps = 100
        max_steps = 1000

        # After warmup, LR should stay at base_lr
        lr = compute_lr(100, base_lr, warmup_steps, max_steps, scheduler='constant')
        assert lr == base_lr

        lr = compute_lr(500, base_lr, warmup_steps, max_steps, scheduler='constant')
        assert lr == base_lr

        lr = compute_lr(999, base_lr, warmup_steps, max_steps, scheduler='constant')
        assert lr == base_lr

    def test_linear_scheduler(self):
        """Test linear LR decay scheduler."""
        base_lr = 1e-4
        warmup_steps = 100
        max_steps = 1000

        # Just after warmup, LR should be close to base_lr
        lr = compute_lr(100, base_lr, warmup_steps, max_steps, scheduler='linear')
        assert pytest.approx(lr, rel=1e-3) == base_lr

        # At midpoint, LR should be ~half
        lr = compute_lr(550, base_lr, warmup_steps, max_steps, scheduler='linear')
        assert pytest.approx(lr, rel=1e-2) == base_lr * 0.5

        # At end, LR should be ~0
        lr = compute_lr(999, base_lr, warmup_steps, max_steps, scheduler='linear')
        assert lr < base_lr * 0.01

    def test_cosine_scheduler(self):
        """Test cosine LR decay scheduler (default)."""
        base_lr = 1e-4
        warmup_steps = 100
        max_steps = 1000

        # Just after warmup, LR should be close to base_lr
        lr = compute_lr(100, base_lr, warmup_steps, max_steps, scheduler='cosine')
        assert pytest.approx(lr, rel=1e-3) == base_lr

        # At midpoint, LR should be ~half (cosine at pi/2)
        lr = compute_lr(550, base_lr, warmup_steps, max_steps, scheduler='cosine')
        assert pytest.approx(lr, rel=1e-2) == base_lr * 0.5

        # At end, LR should be ~0 (cosine at pi)
        lr = compute_lr(999, base_lr, warmup_steps, max_steps, scheduler='cosine')
        assert lr < base_lr * 0.01

    def test_cosine_scheduler_checkpoint_is_weights_only_safe(self, tmp_path):
        """Test cosine LR optimizer state uses checkpoint-safe primitives."""
        lr = compute_lr(500, 1e-4, 100, 1000, scheduler='cosine')

        assert type(lr) is float

        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        optimizer.param_groups[0]['lr'] = lr
        checkpoint_path = tmp_path / 'cosine_lr.ckpt'
        torch.save(
            {'optimizer_states': [optimizer.state_dict()]},
            checkpoint_path,
        )

        original_safe_globals = torch.serialization.get_safe_globals()
        try:
            torch.serialization.clear_safe_globals()
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=True,
            )
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(original_safe_globals)

        restored_lr = checkpoint['optimizer_states'][0]['param_groups'][0]['lr']
        assert type(restored_lr) is float

    def test_zero_warmup(self):
        """Test behavior with zero warmup steps."""
        base_lr = 1e-4
        warmup_steps = 0
        max_steps = 1000

        # At step 0, should start at base_lr (no warmup)
        lr = compute_lr(0, base_lr, warmup_steps, max_steps, scheduler='constant')
        assert lr == base_lr

    def test_default_scheduler_is_cosine(self):
        """Test that default scheduler is cosine."""
        base_lr = 1e-4
        warmup_steps = 0
        max_steps = 1000

        lr_default = compute_lr(500, base_lr, warmup_steps, max_steps)
        lr_cosine = compute_lr(500, base_lr, warmup_steps, max_steps, scheduler='cosine')

        assert lr_default == lr_cosine


@pytest.mark.multimodal_unit
class TestBuildOptimizer:
    """Test build_optimizer function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with vision and text named parameters."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 10)
                self.text_encoder = nn.Linear(10, 10)
                self.logit_scale = nn.Parameter(torch.ones([]))
                self.logit_bias = nn.Parameter(torch.zeros([]))

            def vision_named_parameters(self):
                for name, param in self.vision_encoder.named_parameters():
                    yield f'vision_encoder.{name}', param

            def text_named_parameters(self):
                for name, param in self.text_encoder.named_parameters():
                    yield f'text_encoder.{name}', param

            def other_named_parameters(self):
                yield 'logit_scale', self.logit_scale
                yield 'logit_bias', self.logit_bias

        return MockModel()

    @pytest.fixture
    def mock_train_cfg(self):
        """Create a mock training config."""
        class MockOptimCfg:
            optimizer_type = 'adamw'
            vision_lr = 1e-4
            text_lr = 5e-5
            weight_decay = 0.01
            betas = [0.9, 0.999]
            eps = 1e-8
            warmup_steps = 100
            scheduler = 'cosine'

        class MockTrainCfg:
            optim = MockOptimCfg()

        return MockTrainCfg()

    def test_creates_adamw_optimizer(self, mock_model, mock_train_cfg):
        """Test that AdamW optimizer is created."""
        from torch.optim import AdamW

        optimizer = build_optimizer(mock_model, mock_train_cfg)
        assert isinstance(optimizer, AdamW)

    def test_creates_lamb_optimizer(self, mock_model, mock_train_cfg):
        """Test that LAMB optimizer is created."""
        from apex.optimizers import FusedLAMB

        mock_train_cfg.optim.optimizer_type = 'lamb'
        optimizer = build_optimizer(mock_model, mock_train_cfg)
        assert isinstance(optimizer, FusedLAMB)

    def test_creates_per_tower_groups(self, mock_model, mock_train_cfg):
        """Test that optimizer has per-tower parameter groups."""
        optimizer = build_optimizer(mock_model, mock_train_cfg)

        towers = set()
        for group in optimizer.param_groups:
            if '_tower' in group:
                towers.add(group['_tower'])

        assert 'vision' in towers
        assert 'text' in towers
        assert 'logit' in towers

    def test_vision_lr_applied(self, mock_model, mock_train_cfg):
        """Test that vision LR is correctly applied."""
        mock_train_cfg.optim.vision_lr = 1e-3
        mock_train_cfg.optim.text_lr = 1e-5

        optimizer = build_optimizer(mock_model, mock_train_cfg)

        vision_groups = [g for g in optimizer.param_groups if g.get('_tower') == 'vision']
        for group in vision_groups:
            assert group['lr'] == 1e-3

    def test_text_lr_applied(self, mock_model, mock_train_cfg):
        """Test that text LR is correctly applied."""
        mock_train_cfg.optim.vision_lr = 1e-3
        mock_train_cfg.optim.text_lr = 1e-5

        optimizer = build_optimizer(mock_model, mock_train_cfg)

        text_groups = [g for g in optimizer.param_groups if g.get('_tower') == 'text']
        for group in text_groups:
            assert group['lr'] == 1e-5


@pytest.mark.multimodal_unit
class TestValidConstants:
    """Test validation constants."""

    def test_valid_optimizer_types(self):
        """Test valid optimizer types constant."""
        assert VALID_OPTIMIZER_TYPES == {'adamw', 'lamb'}

    def test_valid_schedulers(self):
        """Test valid schedulers constant."""
        assert VALID_SCHEDULERS == {'cosine', 'constant', 'linear'}
