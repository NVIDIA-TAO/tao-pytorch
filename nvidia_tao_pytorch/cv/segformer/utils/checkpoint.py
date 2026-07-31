# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint helpers for SegFormer pretrained-weight initialization."""

from dataclasses import dataclass
from os import PathLike
from typing import Mapping

import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights


@dataclass(frozen=True)
class PretrainedWeightLoadReport:
    """Deterministic summary of a pretrained-weight initialization."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatched_keys: tuple[str, ...]
    unmatched_keys: tuple[str, ...]
    non_tensor_keys: tuple[str, ...]


def _checkpoint_label(path_or_checkpoint):
    """Return a concise checkpoint label suitable for logs and errors."""
    if isinstance(path_or_checkpoint, (str, PathLike)):
        return str(path_or_checkpoint)
    return "<in-memory checkpoint>"


def _translate_segformer_key(key):
    """Translate an MMSEG decoder namespace to the TAO SegFormer namespace."""
    if key.startswith("decode_head."):
        key = "decoder." + key[len("decode_head."):]
    elif key.startswith("model.decode_head."):
        key = "model.decoder." + key[len("model.decode_head."):]

    key = key.replace(".linear_fuse.conv.", ".linear_fuse.0.")
    key = key.replace(".linear_fuse.bn.", ".linear_fuse.1.")
    return key


def _candidate_model_keys(key):
    """Return ordered, de-duplicated key variants for supported wrappers."""
    variants = []

    def add(candidate):
        if candidate not in variants:
            variants.append(candidate)
        translated = _translate_segformer_key(candidate)
        if translated not in variants:
            variants.append(translated)

    candidate = key
    add(candidate)

    # DDP and torch.compile add these outer wrappers. Only leading wrappers
    # are removed so similarly named components inside a backbone are intact.
    while candidate.startswith(("module.", "_orig_mod.")):
        prefix = "module." if candidate.startswith("module.") else "_orig_mod."
        candidate = candidate[len(prefix):]
        add(candidate)

    # A Lightning state_dict contains ``model.*`` keys, while this helper
    # loads the wrapped torch model whose state_dict starts at ``backbone.*``
    # and ``decoder.*``. Repeated wrappers cover nested Lightning modules.
    while candidate.startswith("model."):
        candidate = candidate[len("model."):]
        add(candidate)

    return tuple(variants)


def _log_report(label, report, shape_mismatches):
    """Log complete sorted missing/skipped information."""
    logging.info(
        "Loaded %d compatible SegFormer pretrained tensors from %s: %s",
        len(report.loaded_keys),
        label,
        list(report.loaded_keys),
    )
    if report.missing_keys:
        logging.warning(
            "SegFormer tensors kept at initialization (%d): %s",
            len(report.missing_keys),
            list(report.missing_keys),
        )
    if shape_mismatches:
        logging.warning(
            "Skipped SegFormer pretrained tensors with incompatible shapes (%d): %s",
            len(shape_mismatches),
            list(shape_mismatches),
        )
    if report.unmatched_keys:
        logging.warning(
            "Skipped SegFormer pretrained tensors without model matches (%d): %s",
            len(report.unmatched_keys),
            list(report.unmatched_keys),
        )
    if report.non_tensor_keys:
        logging.warning(
            "Skipped SegFormer pretrained non-tensor values (%d): %s",
            len(report.non_tensor_keys),
            list(report.non_tensor_keys),
        )


def initialize_pretrained_weights(pl_model, path_or_checkpoint):
    """Initialize a SegFormer Lightning module from generic pretrained weights.

    This deliberately does not restore Lightning training state. The core PTM
    loader unwraps supported checkpoint containers, then only tensors that map
    to existing, shape-compatible model entries are loaded. Decoder tensors
    with a different class count remain at their fresh initialization.

    Args:
        pl_model: A ``SegFormerPlModel``-compatible object with a ``model``
            attribute.
        path_or_checkpoint: A checkpoint path or in-memory checkpoint mapping.

    Returns:
        PretrainedWeightLoadReport: Sorted details of loaded and skipped keys.

    Raises:
        TypeError: If the loader does not return a string-keyed state mapping.
        RuntimeError: If keys are ambiguous or no compatible tensor exists.
    """
    label = _checkpoint_label(path_or_checkpoint)
    state_dict = load_pretrained_weights(path_or_checkpoint, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"SegFormer pretrained checkpoint {label} did not contain a state dictionary."
        )
    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError(
            f"SegFormer pretrained checkpoint {label} contains non-string state-dict keys."
        )

    model = pl_model.model
    model_state = model.state_dict()
    compatible = {}
    source_for_target = {}
    shape_mismatched_keys = []
    shape_mismatches = []
    unmatched_keys = []
    non_tensor_keys = []

    for source_key in sorted(state_dict):
        value = state_dict[source_key]
        target_key = next(
            (
                candidate
                for candidate in _candidate_model_keys(source_key)
                if candidate in model_state
            ),
            None,
        )
        if target_key is None:
            unmatched_keys.append(source_key)
            continue
        if not torch.is_tensor(value):
            non_tensor_keys.append(source_key)
            continue

        target = model_state[target_key]
        checkpoint_shape = tuple(value.shape)
        model_shape = tuple(target.shape)
        if checkpoint_shape != model_shape:
            shape_mismatched_keys.append(target_key)
            shape_mismatches.append(
                f"{source_key} -> {target_key}: "
                f"checkpoint={checkpoint_shape}, model={model_shape}"
            )
            continue
        if target_key in compatible:
            previous = source_for_target[target_key]
            raise RuntimeError(
                "Ambiguous SegFormer pretrained checkpoint keys "
                f"{previous!r} and {source_key!r} both map to {target_key!r}."
            )
        compatible[target_key] = value
        source_for_target[target_key] = source_key

    loaded_keys = tuple(sorted(compatible))
    report = PretrainedWeightLoadReport(
        loaded_keys=loaded_keys,
        missing_keys=tuple(sorted(set(model_state) - set(compatible))),
        shape_mismatched_keys=tuple(sorted(shape_mismatched_keys)),
        unmatched_keys=tuple(sorted(unmatched_keys)),
        non_tensor_keys=tuple(sorted(non_tensor_keys)),
    )
    _log_report(label, report, tuple(sorted(shape_mismatches)))

    if not compatible:
        raise RuntimeError(
            "SegFormer pretrained checkpoint "
            f"{label} contains no tensors compatible with the configured model."
        )

    model.load_state_dict(compatible, strict=False)
    return report
