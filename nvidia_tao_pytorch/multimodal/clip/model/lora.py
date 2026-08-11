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

"""LoRA (Low-Rank Adaptation) utilities for CLIP models.

Provides lightweight LoRA injection and merging for CLIP adapters.
Works with both separate (q_proj, k_proj, ...) and fused (qkv) attention
projections across SigLIP2, OpenCLIP, and RADIO backbones.
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
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze the original parameters
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # LoRA matrices: A projects down, B projects back up
        # Create on the same device/dtype as the original weight so
        # inject_lora works correctly when the model is already on GPU.
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
            # delta_W = B @ A * scaling, shape (out_features, in_features)
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original.weight.add_(delta)
        return self.original


def _match_target(module_name, target_modules):
    """Check if a module name matches any target substring."""
    # Match the final component of the module name
    leaf = module_name.rsplit('.', 1)[-1]
    return leaf in target_modules


def inject_lora(model, peft_config):
    """Inject LoRA adapters into a CLIP model based on config.

    Freezes all backbone parameters, then inserts LoRALinear wrappers
    into the target attention modules in the last N blocks of each
    enabled encoder tower.

    Args:
        model: A BaseCLIPAdapter instance.
        peft_config: CLIPPEFTConfig with vision/text sub-configs.

    Returns:
        Dict with injection statistics:
            - 'total_params': Total model parameters.
            - 'trainable_params': Trainable parameters after injection.
            - 'lora_params': Parameters in LoRA modules only.
            - 'injected_modules': List of (tower, module_name) tuples.
    """
    if not peft_config.enabled:
        return None

    # Phase 1: Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Phase 2: Inject LoRA into enabled towers
    injected = []
    lora_param_count = 0

    for tower_name, tower_cfg in [('vision', peft_config.vision),
                                  ('text', peft_config.text)]:
        if not tower_cfg.enabled:
            continue

        blocks = model.get_encoder_blocks(tower_name)
        num_blocks = len(blocks)

        if tower_cfg.num_last_blocks == 0 or tower_cfg.num_last_blocks >= num_blocks:
            target_blocks = blocks
        else:
            target_blocks = blocks[-tower_cfg.num_last_blocks:]

        start_idx = num_blocks - len(target_blocks)
        tower_injected = 0
        seen_linear_leaves = set()

        for block_offset, block in enumerate(target_blocks):
            block_idx = start_idx + block_offset
            for name, module in block.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                seen_linear_leaves.add(name.rsplit('.', 1)[-1])
                if not _match_target(name, tower_cfg.target_modules):
                    continue

                # Replace the Linear with a LoRALinear
                lora_module = LoRALinear(
                    module,
                    rank=tower_cfg.rank,
                    alpha=tower_cfg.alpha,
                    dropout=tower_cfg.dropout,
                )

                # Set the LoRALinear on the parent module
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    setattr(block, name, lora_module)
                else:
                    parent = block.get_submodule(parts[0])
                    setattr(parent, parts[1], lora_module)

                injected.append((tower_name, f"block[{block_idx}].{name}"))
                tower_injected += 1
                lora_param_count += (
                    lora_module.lora_A.numel() + lora_module.lora_B.numel()
                )

        if tower_injected == 0:
            raise ValueError(
                f"PEFT is enabled for the {tower_name} tower but no LoRA "
                f"adapters were injected: target_modules "
                f"{list(tower_cfg.target_modules)} matched no nn.Linear in the "
                f"last {len(target_blocks)} of {num_blocks} blocks. The "
                f"backbone is already frozen at this point, so training would "
                f"proceed with nothing trainable. nn.Linear leaf names "
                f"available in those blocks: {sorted(seen_linear_leaves)}. "
                f"(Fused attention, e.g. nn.MultiheadAttention's in_proj, is a "
                f"Parameter rather than a Linear submodule and cannot be "
                f"targeted by name.)"
            )

    # Phase 3: Unfreeze logit_scale and logit_bias
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True
    if hasattr(model, 'logit_bias'):
        model.logit_bias.requires_grad = True

    # Compute stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_param_count,
        'injected_modules': injected,
    }

    # Log summary
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0
    logging.info(
        f"LoRA injection complete: "
        f"{len(injected)} modules adapted, "
        f"{lora_param_count:,} LoRA params, "
        f"{trainable_params:,} / {total_params:,} trainable ({pct:.2f}%)"
    )
    for tower, mod_name in injected:
        logging.info(f"  [{tower}] {mod_name}")

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
