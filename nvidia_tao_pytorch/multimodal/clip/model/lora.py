# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA injection and merge utilities for CLIP encoder projections."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank residual."""

    def __init__(self, original, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(
            rank, original.in_features,
            device=original.weight.device, dtype=original.weight.dtype,
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            original.out_features, rank,
            device=original.weight.device, dtype=original.weight.dtype,
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        """Apply the frozen base layer and its scaled LoRA residual."""
        residual = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return self.original(x) + residual * self.scaling

    def merge(self):
        """Fold the LoRA residual into and return the wrapped linear layer."""
        with torch.no_grad():
            self.original.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
        return self.original


def _match_target(module_name, target_modules):
    """Match a configured projection name against a module leaf name."""
    return module_name.rsplit('.', 1)[-1] in target_modules


VALID_TOWER_MODES = {'frozen', 'full', 'lora'}


def _config_value(config, name, default=None):
    """Read a config value from a mapping or dataclass-like object."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_tower_mode(tower_config, tower_name):
    """Validate and return a tower's explicit adaptation mode."""
    mode = _config_value(tower_config, 'mode')
    if mode in (None, '???'):
        raise ValueError(
            f"PEFT {tower_name}.mode is required when peft.enabled=true. "
            "Choose one of: frozen, full, lora."
        )
    if mode not in VALID_TOWER_MODES:
        raise ValueError(
            f"Invalid {tower_name} PEFT mode {mode!r}. Expected one of "
            f"{sorted(VALID_TOWER_MODES)}."
        )
    return mode


def _tower_named_parameters(model, tower_name):
    """Return the canonical parameter iterator for one encoder tower."""
    getter = getattr(model, f'{tower_name}_named_parameters', None)
    if not callable(getter):
        raise TypeError(f"Model must expose {tower_name}_named_parameters() for PEFT.")
    return list(getter())


def _enable_lora_parameters(module):
    """Enable only the trainable adapter weights of one wrapped projection."""
    module.lora_A.requires_grad = True
    module.lora_B.requires_grad = True


def inject_lora(model, peft_config):
    """Apply explicit per-tower PEFT modes and inject LoRA where requested."""
    if not _config_value(peft_config, 'enabled', False):
        return None
    towers = (
        ('vision', _config_value(peft_config, 'vision')),
        ('text', _config_value(peft_config, 'text')),
    )
    resolved_modes = {name: resolve_tower_mode(config, name) for name, config in towers}
    for parameter in model.parameters():
        parameter.requires_grad = False

    injected, lora_param_count = [], 0
    tower_trainable_params = {'vision': 0, 'text': 0}
    for tower_name, tower_config in towers:
        mode = resolved_modes[tower_name]
        if mode == 'frozen':
            continue
        if mode == 'full':
            for _, parameter in _tower_named_parameters(model, tower_name):
                parameter.requires_grad = True
        else:
            blocks = list(model.get_encoder_blocks(tower_name))
            num_last_blocks = tower_config.num_last_blocks
            if not 0 <= num_last_blocks <= len(blocks):
                raise ValueError(
                    f"PEFT {tower_name}.num_last_blocks must be between 0 "
                    f"(all blocks) and {len(blocks)}, got {num_last_blocks}."
                )
            target_blocks = blocks if num_last_blocks == 0 else blocks[-num_last_blocks:]
            start_index = len(blocks) - len(target_blocks)
            available_linear_leaves = set()
            for offset, block in enumerate(target_blocks):
                for name, module in list(block.named_modules()):
                    if isinstance(module, nn.Linear):
                        available_linear_leaves.add(name.rsplit('.', 1)[-1])
                    if not _match_target(name, tower_config.target_modules):
                        continue
                    parent_path, _, attribute = name.rpartition('.')
                    parent = block.get_submodule(parent_path) if parent_path else block
                    if isinstance(module, nn.Linear):
                        module = LoRALinear(module, tower_config.rank, tower_config.alpha, tower_config.dropout)
                        setattr(parent, attribute, module)
                        injected.append((tower_name, f'block[{start_index + offset}].{name}'))
                    if isinstance(module, LoRALinear):
                        _enable_lora_parameters(module)
                        lora_param_count += module.lora_A.numel() + module.lora_B.numel()
        tower_trainable_params[tower_name] = sum(
            parameter.numel() for _, parameter in _tower_named_parameters(model, tower_name)
            if parameter.requires_grad
        )
        if mode == 'lora' and tower_trainable_params[tower_name] == 0:
            raise ValueError(
                f"PEFT {tower_name} mode='lora' injected zero modules (no LoRA adapters): "
                f"target_modules {list(tower_config.target_modules)} matched no "
                f"nn.Linear modules. Available leaves: "
                f"{sorted(available_linear_leaves)}. Check target_modules and "
                "num_last_blocks for this backbone."
            )

    if _config_value(peft_config, 'train_logit_calibration', True):
        for _, parameter in model.named_logit_parameters():
            parameter.requires_grad = True
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    stats = {
        'total_params': total_params, 'trainable_params': trainable_params,
        'lora_params': lora_param_count, 'injected_modules': injected,
        'requested_modes': resolved_modes, 'tower_trainable_params': tower_trainable_params,
        'logit_trainable_params': sum(parameter.numel() for _, parameter in model.named_logit_parameters() if parameter.requires_grad),
    }
    if trainable_params == 0:
        raise ValueError(
            "PEFT is enabled but no tower parameters are trainable and "
            "train_logit_calibration is disabled. Select a full or lora tower, "
            "or enable calibration."
        )
    logging.info(
        'PEFT modes requested: vision=%s, text=%s; trainable parameters: vision=%s, text=%s, logit=%s. '
        '%s LoRA modules adapted, %s / %s trainable (%.2f%%).',
        resolved_modes['vision'], resolved_modes['text'], tower_trainable_params['vision'],
        tower_trainable_params['text'], stats['logit_trainable_params'], len(injected), trainable_params,
        total_params, 100.0 * trainable_params / total_params if total_params else 0.0,
    )
    return stats


def merge_lora(model):
    """Replace every ``LoRALinear`` below ``model`` with its merged linear."""
    merged_count = 0
    for parent in model.modules():
        for attribute, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, attribute, child.merge())
                merged_count += 1
    if merged_count:
        logging.info('Merged %s LoRA modules into base weights.', merged_count)
    return merged_count
