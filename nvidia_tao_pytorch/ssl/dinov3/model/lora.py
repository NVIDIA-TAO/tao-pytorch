# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key-preserving LoRA for the DINOv3 SSL backbone.

Parameter-efficient continual pre-training: instead of updating all 86 M ViT-B backbone
weights, freeze them and learn a rank-``r`` update ``dW = (alpha/r) * B @ A`` per targeted
projection. See ``docs/plans/dinov3_lora_design.md``.

**Key-preserving by design.** :class:`LoRALinear` subclasses ``nn.Linear`` and adopts the
wrapped layer's existing ``weight``/``bias`` *Parameter objects*, so the backbone state-dict
keys are exactly the stock keys plus ``lora_A``/``lora_B``. This is deliberately *not* the
CLIP branch's ``.original``-nesting wrapper, which renames every wrapped weight to
``...qkv.original.weight``. Four DINOv3 subsystems depend on stable backbone key names:

* ``checkpoint_remap.timm_to_tao`` / ``convert_ssl_to_timm`` (timm <-> TAO translation),
* ``CustomModelCheckpoint`` (strips ``student.backbone.``/``teacher.backbone.`` prefixes),
* ``DinoV3PlModel._sync_gram_teacher`` (loads the teacher backbone into the plain anchor teacher),
* ``restore_pretrained_weights`` (non-strict load against reference keys).

**EMA safety.** ``DinoV2PlModel.update_teacher`` zips ``student.parameters()`` with
``teacher.parameters()`` element-wise, so LoRA must be injected into the student *and* the
EMA teacher with identical structure. Injection order here is deterministic
(``nn.Module`` registration order), so the two zips stay aligned. The frozen anchor (Gram)
teacher deliberately gets **no** LoRA -- it anchors to the pretrained weights, which under
LoRA are precisely the frozen base weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging

# Projections that may be targeted, mapped to their parent module inside a ViT block.
# ``qkv``/``proj`` live on ``block.attn``; ``fc1``/``fc2`` on ``block.mlp``. SwiGLU
# backbones (ViT-S+/H+/7B) expose a *fused* ``fc1`` -- targeting it is allowed but is
# deferred in v1 (see the design doc's "Deferred" section).
_TARGET_PARENT = {
    "qkv": "attn",
    "proj": "attn",
    "fc1": "mlp",
    "fc2": "mlp",
}

DEFAULT_TARGET_MODULES = ("qkv", "proj")


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` that additionally learns a low-rank update, preserving its keys.

    The frozen base path is the stock linear ``x @ W.T + b``; the LoRA path adds
    ``(alpha/r) * dropout(x) @ A.T @ B.T``. ``lora_B`` is zero-initialized so a freshly
    injected module is **exactly** the identity of the layer it replaced -- step-0 behavior
    of the SSL run is unchanged (gate G1.1).

    Attributes:
        lora_A (nn.Parameter): Down-projection ``[rank, in_features]``, Kaiming-uniform init.
        lora_B (nn.Parameter): Up-projection ``[out_features, rank]``, zero init.
        scaling (float): ``alpha / rank``.
        merged (bool): Whether :meth:`merge` has folded the delta into ``weight``.
    """

    def __init__(self, in_features, out_features, bias=True, rank=8, alpha=16.0,
                 dropout=0.0, device=None, dtype=None):
        """Build a LoRA-augmented linear layer.

        Args:
            in_features (int): Input dimension.
            out_features (int): Output dimension.
            bias (bool): Whether the base linear has a bias.
            rank (int): LoRA rank ``r`` (must be >= 1).
            alpha (float): LoRA scaling numerator; the applied scale is ``alpha / rank``.
            dropout (float): Dropout applied to the LoRA branch input only.
            device: Passed to ``nn.Linear``.
            dtype: Passed to ``nn.Linear``.
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert rank >= 1, f"LoRA rank must be >= 1, got {rank}"
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.merged = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # Persist alpha/rank as the single scalar that matters, so a checkpoint carries
        # everything needed to fold the adapter in later. Without it ``convert``/``export``
        # would have to re-derive the scale from whichever spec happened to be passed at
        # merge time, and a mismatched alpha would silently produce a wrongly-scaled backbone.
        self.register_buffer("lora_scaling", torch.tensor(self.scaling), persistent=True)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """(Re)initialize the adapter: Kaiming-uniform ``A``, zero ``B`` (identity at init)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear, rank=8, alpha=16.0, dropout=0.0):
        """Wrap an existing ``nn.Linear``, adopting its weight/bias Parameter objects.

        The returned module shares (does not copy) ``linear.weight`` and ``linear.bias``, so
        pretrained values carry over and the state-dict keys are unchanged.

        Args:
            linear (nn.Linear): The layer to augment.
            rank (int): LoRA rank.
            alpha (float): LoRA scaling numerator.
            dropout (float): LoRA-branch dropout.

        Returns:
            LoRALinear: A layer numerically identical to ``linear`` at init.
        """
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Adopt the original Parameter objects so pretrained weights and any existing
        # optimizer/parameter identity are preserved.
        module.weight = linear.weight
        if linear.bias is not None:
            module.bias = linear.bias
        return module

    def delta_weight(self):
        """Return the low-rank weight update ``(alpha/r) * B @ A``, shaped like ``weight``.

        Returns:
            torch.Tensor: Delta shaped ``[out_features, in_features]``.
        """
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self):
        """Fold the LoRA delta into the base ``weight`` and neutralize the adapter.

        After merging, the module computes the stock linear function of the *adapted*
        weights, so ``forward`` is unchanged (gate G1.2) while the adapter contributes
        nothing further. Idempotent.
        """
        if self.merged:
            return self
        self.weight.data += self.delta_weight().to(self.weight.dtype)
        # Zero the adapter so the merged module stays numerically equivalent whether or not
        # a caller later re-runs the LoRA branch.
        self.lora_B.data.zero_()
        self.merged = True
        return self

    def forward(self, x):
        """Apply the frozen base linear plus the low-rank update.

        Args:
            x (torch.Tensor): Input shaped ``[..., in_features]``.

        Returns:
            torch.Tensor: Output shaped ``[..., out_features]``.
        """
        out = F.linear(x, self.weight, self.bias)
        if self.merged:
            return out
        lora_x = self.lora_dropout(x)
        # Cast the adapter to the activation dtype so autocast/fp16 runs stay consistent.
        lora_out = F.linear(F.linear(lora_x, self.lora_A.to(lora_x.dtype)),
                            self.lora_B.to(lora_x.dtype))
        return out + lora_out * self.scaling

    def extra_repr(self):
        """Include the LoRA hyper-parameters in the module repr."""
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha}, merged={self.merged}"


def _resolve_target_blocks(backbone, num_last_blocks):
    """Return the ``(index, block)`` pairs LoRA should be injected into.

    Args:
        backbone (nn.Module): A ``DinoV3VisionTransformer``.
        num_last_blocks (int): Inject into the last ``n`` blocks; ``0`` means all blocks.

    Returns:
        list: ``[(block_index, block_module), ...]`` in ascending index order.
    """
    blocks = list(backbone.blocks)
    if not num_last_blocks or num_last_blocks <= 0 or num_last_blocks >= len(blocks):
        selected = list(range(len(blocks)))
    else:
        selected = list(range(len(blocks) - num_last_blocks, len(blocks)))
    return [(i, blocks[i]) for i in selected]


def inject_lora(backbone, rank=8, alpha=16.0, dropout=0.0,
                target_modules=DEFAULT_TARGET_MODULES, num_last_blocks=0,
                freeze_base=True, keep_mask_token_trainable=True):
    """Replace targeted projections in a DINOv3 backbone with :class:`LoRALinear` in place.

    Injection is idempotent per-module (an already-injected layer is skipped) and preserves
    every base state-dict key. Call this for the student *and* the EMA teacher with the same
    arguments so ``update_teacher``'s parameter zip stays aligned (gate G1.3).

    Args:
        backbone (nn.Module): A ``DinoV3VisionTransformer`` to modify in place.
        rank (int): LoRA rank.
        alpha (float): LoRA scaling numerator.
        dropout (float): LoRA-branch dropout.
        target_modules (Sequence[str]): Projection names to target, from
            ``{'qkv', 'proj', 'fc1', 'fc2'}``.
        num_last_blocks (int): Inject into the last ``n`` blocks; ``0`` = all blocks.
        freeze_base (bool): Freeze every non-LoRA backbone parameter.
        keep_mask_token_trainable (bool): Keep ``mask_token`` trainable. It is absent from the
            (inference) DINOv3 checkpoint and drives iBOT masking, so freezing its random
            init would break the SSL objective.

    Returns:
        list: Fully-qualified names of the injected modules (e.g. ``blocks.0.attn.qkv``).

    Raises:
        ValueError: If ``target_modules`` names an unknown projection.
    """
    unknown = [t for t in target_modules if t not in _TARGET_PARENT]
    if unknown:
        raise ValueError(
            f"Unknown LoRA target_modules {unknown}; supported: {sorted(_TARGET_PARENT)}"
        )

    if freeze_base:
        # Freeze the *base* only. Freezing indiscriminately would also switch off adapters
        # added by an earlier call -- and since already-injected modules are skipped below,
        # nothing would turn them back on. That made injection non-idempotent: a second call,
        # or a staged ``qkv`` then ``proj`` injection, silently left the first batch of
        # adapters frozen and training only the heads.
        for name, param in backbone.named_parameters():
            if not is_lora_key(name):
                param.requires_grad = False

    injected = []
    for block_index, block in _resolve_target_blocks(backbone, num_last_blocks):
        for target in target_modules:
            parent = getattr(block, _TARGET_PARENT[target], None)
            if parent is None:
                continue
            child = getattr(parent, target, None)
            if child is None:
                continue
            if isinstance(child, LoRALinear):
                continue  # already injected
            if not isinstance(child, nn.Linear):
                logging.warning(
                    f"Skipping LoRA target blocks.{block_index}.{_TARGET_PARENT[target]}.{target}: "
                    f"expected nn.Linear, found {type(child).__name__}."
                )
                continue
            setattr(
                parent, target,
                LoRALinear.from_linear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            injected.append(f"blocks.{block_index}.{_TARGET_PARENT[target]}.{target}")

    # Every adapter present must be trainable, including any added by an earlier call.
    for name, param in backbone.named_parameters():
        if is_lora_key(name):
            param.requires_grad = True

    if keep_mask_token_trainable and getattr(backbone, "mask_token", None) is not None:
        backbone.mask_token.requires_grad = True

    return injected


def merge_lora(module):
    """Merge every :class:`LoRALinear` under ``module`` into its base weights, in place.

    After this, the module's forward is a stock DINOv3 forward of adapted weights, so a
    traced/exported graph contains no LoRA ops (gate G3.4).

    Args:
        module (nn.Module): Any module containing LoRA layers (typically a backbone).

    Returns:
        int: Number of layers merged.
    """
    count = 0
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear) and not submodule.merged:
            submodule.merge()
            count += 1
    return count


def has_lora(module):
    """Whether ``module`` contains any :class:`LoRALinear` layer.

    Args:
        module (nn.Module): Module to inspect.

    Returns:
        bool: True if at least one LoRA layer is present.
    """
    return any(isinstance(m, LoRALinear) for m in module.modules())


def is_lora_key(key):
    """Whether a state-dict key belongs to a LoRA adapter.

    Used to filter adapter keys out before loading a LoRA-injected teacher backbone into the
    plain (un-injected) anchor teacher (gate G1.6), and before timm translation on convert.

    Args:
        key (str): State-dict key.

    Returns:
        bool: True for ``lora_A`` / ``lora_B`` keys.
    """
    return key.endswith("lora_A") or key.endswith("lora_B") or ".lora_" in key


def strip_lora_keys(state_dict):
    """Return a copy of ``state_dict`` with all LoRA adapter keys removed.

    Args:
        state_dict (Mapping): Source state dict.

    Returns:
        dict: State dict containing only the base (stock-topology) keys.
    """
    return {k: v for k, v in state_dict.items() if not is_lora_key(k)}


def lora_parameter_report(model, name="model"):
    """Summarize the trainable-parameter split of a (possibly LoRA-injected) model.

    Args:
        model (nn.Module): The model to report on (e.g. the student ``ModuleDict``).
        name (str): Label used in the log line.

    Returns:
        dict: ``{'total', 'trainable', 'lora', 'trainable_fraction'}`` parameter counts.
    """
    total = trainable = lora = 0
    for param_name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            if is_lora_key(param_name):
                lora += n

    stats = {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "trainable_fraction": (trainable / total) if total else 0.0,
    }
    logging.info(
        f"[LoRA] {name}: {trainable:,} trainable / {total:,} total params "
        f"({100.0 * stats['trainable_fraction']:.3f}%); of which LoRA adapters: {lora:,}."
    )
    return stats
