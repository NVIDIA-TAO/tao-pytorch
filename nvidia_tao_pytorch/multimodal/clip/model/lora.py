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

"""LoRA injection and merging utilities for CLIP models.

Supports the ``nn.Linear`` attention projections used by SigLIP2, RADIO, and
VideoCLIP backbones. OpenCLIP's ``nn.MultiheadAttention`` projections are not
currently supported in LoRA mode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    Wraps an existing nn.Linear, freezing its weight and bias, and adds
    trainable low-rank matrices A and B such that:

        output = original_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    Args:
        original: The nn.Linear module to wrap.
        rank: Low-rank dimension.
        alpha: Scaling factor. Effective scale = alpha / rank.
        dropout: Dropout probability applied to input before LoRA path.
    """

    def __init__(self, original, rank=8, alpha=16, dropout=0.05):
        """Initialize LoRALinear from an existing nn.Linear."""
        super().__init__()
        self.original = original
        self.rank = rank
        self.register_buffer(
            'scaling',
            torch.tensor(
                alpha / rank,
                dtype=torch.float32,
                device=original.weight.device,
            ),
        )

        in_features = original.in_features
        out_features = original.out_features

        # Freeze the original parameters.
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # A projects down to the configured rank; B projects back to the
        # original output width. Match the wrapped layer's device and dtype.
        device = original.weight.device
        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """Forward pass: original output + scaled LoRA residual."""
        base_out = self.original(x)
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        )
        return base_out + lora_out * self.scaling

    def merge(self):
        """Fold LoRA weights into the original linear and return it.

        After merging, the original nn.Linear produces the adapted output
        directly, with no LoRA overhead. Used before ONNX export.

        Returns:
            The original nn.Linear with merged weights.
        """
        with torch.no_grad():
            # delta_W has shape (out_features, in_features).
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original.weight.add_(delta)
        return self.original


def _match_target(module_name, target_modules):
    """Match a configured projection name against a module leaf name."""
    leaf = module_name.rsplit('.', 1)[-1]
    return leaf in target_modules


VALID_TOWER_MODES = {'frozen', 'full', 'lora'}


def _config_value(config, name, default=None):
    """Read a config value from a mapping or dataclass-like object."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_tower_mode(tower_config, tower_name):
    """Validate and return the configured adaptation mode for one tower.

    Args:
        tower_config: Mapping or dataclass-like tower configuration.
        tower_name: Tower name used in validation errors.

    Returns:
        One of ``frozen``, ``full``, or ``lora``.

    Raises:
        ValueError: If the configured mode is not supported.
    """
    mode = _config_value(tower_config, 'mode')
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


def _named_logit_parameters(model):
    """Return logit parameters through the canonical API or adapter fallback."""
    getter = getattr(model, 'named_logit_parameters', None)
    if callable(getter):
        return list(getter())

    named_parameters = []
    for name in ('logit_scale', 'logit_bias'):
        getter = getattr(model, f'get_{name}_parameter', None)
        parameter = getter() if callable(getter) else getattr(model, name, None)
        if isinstance(parameter, nn.Parameter):
            named_parameters.append((name, parameter))
    return named_parameters


def _preflight_lora_towers(model, towers, resolved_modes):
    """Reject unsupported LoRA architectures before changing model state."""
    for tower_name, tower_config in towers:
        if resolved_modes[tower_name] != 'lora':
            continue
        blocks = list(model.get_encoder_blocks(tower_name))
        num_last_blocks = tower_config.num_last_blocks
        if not 0 <= num_last_blocks <= len(blocks):
            raise ValueError(
                f"PEFT {tower_name}.num_last_blocks must be between 0 "
                f"(all blocks) and {len(blocks)}, got {num_last_blocks}."
            )
        target_blocks = blocks if num_last_blocks == 0 else blocks[-num_last_blocks:]
        if any(
            isinstance(module, nn.MultiheadAttention)
            for block in target_blocks for module in block.modules()
        ):
            raise NotImplementedError(
                f"PEFT {tower_name} mode='lora' is not supported for "
                "nn.MultiheadAttention-based towers (OpenCLIP). Use "
                "mode='full' or mode='frozen'."
            )


def _restore_legacy_lora_scaling(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Supply configured scaling when loading a legacy LoRA checkpoint."""
    del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    expected_scaling = []
    for name, child in module.named_modules():
        if not isinstance(child, LoRALinear):
            continue
        local_key = f'{name}.scaling' if name else 'scaling'
        checkpoint_key = prefix + local_key
        expected_scaling.append((checkpoint_key, child.scaling))
    missing_scaling = [
        (key, scaling)
        for key, scaling in expected_scaling
        if key not in state_dict
    ]
    if missing_scaling and len(missing_scaling) == len(expected_scaling):
        for key, scaling in missing_scaling:
            state_dict[key] = scaling.detach().clone()
        logging.warning(
            'Loading a legacy LoRA checkpoint without persisted scaling for '
            '%s adapter(s); using scaling derived from the current PEFT rank '
            'and alpha. These values must match the original training config.',
            len(missing_scaling),
        )


def _register_lora_checkpoint_compatibility(model):
    """Register one model-level hook for legacy LoRA scaling fallback."""
    if getattr(model, '_lora_checkpoint_compatibility_registered', False):
        return
    model.register_load_state_dict_pre_hook(_restore_legacy_lora_scaling)
    model._lora_checkpoint_compatibility_registered = True


def inject_lora(model, peft_config):
    """Apply the configured adaptation mode to each CLIP encoder tower.

    When PEFT is enabled, all model parameters are frozen first. Each tower is
    then kept frozen, fully unfrozen, or given ``LoRALinear`` wrappers in its
    configured final transformer blocks. Logit calibration parameters are
    controlled independently by ``train_logit_calibration``.

    Args:
        model: CLIP or VideoCLIP adapter exposing tower blocks and parameters.
        peft_config: PEFT configuration with ``vision`` and ``text`` targets.

    Returns:
        Injection statistics, including trainable parameter counts, requested
        modes, and injected module names. Returns ``None`` when PEFT is disabled.

    Raises:
        RuntimeError: If the model already contains LoRA adapters.
        TypeError: If the model does not expose a required tower API.
        ValueError: If a mode, block depth, or target selection is invalid.
        NotImplementedError: If LoRA is requested for an unsupported tower.
    """
    if not _config_value(peft_config, 'enabled', False):
        return None
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError(
            "inject_lora() does not support re-injection on an adapted model."
        )
    towers = (
        ('vision', _config_value(peft_config, 'vision')),
        ('text', _config_value(peft_config, 'text')),
    )
    resolved_modes = {name: resolve_tower_mode(config, name) for name, config in towers}
    _preflight_lora_towers(model, towers, resolved_modes)
    logit_parameters = _named_logit_parameters(model)

    # Establish a frozen baseline before applying each tower's requested mode.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze full towers or inject trainable low-rank adapters into LoRA towers.
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
        for _, parameter in logit_parameters:
            parameter.requires_grad = True
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    stats = {
        'total_params': total_params, 'trainable_params': trainable_params,
        'lora_params': lora_param_count, 'injected_modules': injected,
        'requested_modes': resolved_modes, 'tower_trainable_params': tower_trainable_params,
        'logit_trainable_params': sum(
            parameter.numel() for _, parameter in logit_parameters if parameter.requires_grad
        ),
    }
    if trainable_params == 0:
        raise ValueError(
            "PEFT is enabled but no tower parameters are trainable and "
            "train_logit_calibration is disabled. Select a full or lora tower, "
            "or enable calibration."
        )
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        _register_lora_checkpoint_compatibility(model)
    logging.info(
        'PEFT modes requested: vision=%s, text=%s; trainable parameters: vision=%s, text=%s, logit=%s. '
        '%s LoRA modules adapted, %s / %s trainable (%.2f%%).',
        resolved_modes['vision'], resolved_modes['text'], tower_trainable_params['vision'],
        tower_trainable_params['text'], stats['logit_trainable_params'], len(injected), trainable_params,
        total_params, 100.0 * trainable_params / total_params if total_params else 0.0,
    )
    return stats


def merge_lora(model):
    """Merge all LoRA weights into the base model for export.

    Walks the model and replaces every LoRALinear with its merged
    nn.Linear. After this call the model has no LoRA modules and
    is architecturally identical to a non-LoRA model.

    Args:
        model: A model potentially containing LoRALinear modules.

    Returns:
        Number of modules merged.
    """
    merged_count = 0
    for _, parent_module in model.named_modules():
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, LoRALinear):
                merged_linear = child.merge()
                setattr(parent_module, attr_name, merged_linear)
                merged_count += 1

    if merged_count > 0:
        logging.info(f"Merged {merged_count} LoRA modules into base weights.")
    return merged_count
