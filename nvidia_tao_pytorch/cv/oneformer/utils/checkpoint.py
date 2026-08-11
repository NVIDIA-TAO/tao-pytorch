# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-dictionary matching helpers for OneFormer transfer checkpoints."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass


_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "model.")


@dataclass(frozen=True)
class PretrainedStateDictReport:
    """Describe how a pretrained state dictionary matched the target model."""

    source_key_count: int
    target_key_count: int
    loaded_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    incompatible_shape_keys: tuple[str, ...]

    def to_dict(self):
        """Return a JSON-serializable report."""
        return asdict(self)


def _shape(value):
    """Return a stable tuple shape for tensor-like values."""
    return tuple(getattr(value, "shape", ()))


def _candidate_target_keys(source_key):
    """Yield deterministic target-key candidates for a wrapped source key."""
    candidate = source_key
    yielded = set()
    while candidate not in yielded:
        yielded.add(candidate)
        yield candidate
        for prefix in _WRAPPER_PREFIXES:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):]
                break
        else:
            break


def match_pretrained_state_dict(source_state, target_state):
    """Match a checkpoint state dictionary to ``OneFormerModel.state_dict``.

    Lightning, DDP, and compiled checkpoints can add one or more wrapper
    prefixes. This function strips only leading wrapper prefixes, retains only
    keys present in the target model, and excludes shape-incompatible tensors.
    A collision after prefix normalization is rejected instead of depending on
    checkpoint enumeration order.

    Args:
        source_state: Mapping of checkpoint parameter names to tensor-like values.
        target_state: Mapping returned by the target model's ``state_dict``.

    Returns:
        A tuple ``(compatible_state, report)``.
    """
    if not isinstance(source_state, Mapping):
        raise TypeError("OneFormer pretrained weights must be a state-dictionary mapping.")
    if not isinstance(target_state, Mapping):
        raise TypeError("OneFormer target state must be a state-dictionary mapping.")

    compatible = OrderedDict()
    source_for_target = {}
    unexpected = []
    incompatible = []

    for source_key in sorted(source_state):
        if not isinstance(source_key, str):
            raise TypeError("OneFormer state-dictionary keys must be strings.")

        target_key = next(
            (
                candidate
                for candidate in _candidate_target_keys(source_key)
                if candidate in target_state
            ),
            None,
        )
        if target_key is None:
            unexpected.append(source_key)
            continue

        if _shape(source_state[source_key]) != _shape(target_state[target_key]):
            incompatible.append(source_key)
            continue

        if target_key in compatible:
            previous = source_for_target[target_key]
            raise ValueError(
                "OneFormer checkpoint keys normalize to the same target key: "
                f"{previous!r} and {source_key!r} -> {target_key!r}."
            )
        compatible[target_key] = source_state[source_key]
        source_for_target[target_key] = source_key

    missing = tuple(sorted(set(target_state) - set(compatible)))
    report = PretrainedStateDictReport(
        source_key_count=len(source_state),
        target_key_count=len(target_state),
        loaded_key_count=len(compatible),
        missing_keys=missing,
        unexpected_keys=tuple(unexpected),
        incompatible_shape_keys=tuple(incompatible),
    )
    return compatible, report
