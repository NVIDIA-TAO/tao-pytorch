# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint helpers for SegFormer pretrained-weight initialization."""

from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from typing import Mapping

import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights


PRETRAINED_LOAD_REPORT_PREFIX = "SEGFORMER_PRETRAINED_LOAD_REPORT "


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


def _candidate_backbone_keys(key):
    """Return key variants for a checkpoint loaded into the bare backbone.

    Official FAN classification checkpoints wrap backbone tensors in a
    leading ``backbone.`` namespace, while ``SegFormer.backbone`` expects the
    same tensors without that outer namespace.  Only leading wrapper tokens
    are removed; an internal name such as ``patch_embed.backbone`` is left
    untouched.
    """
    variants = []

    def add(candidate):
        if candidate not in variants:
            variants.append(candidate)

    candidate = key
    add(candidate)
    while True:
        prefix = next(
            (
                value
                for value in ("module.", "_orig_mod.", "model.")
                if candidate.startswith(value)
            ),
            None,
        )
        if prefix is None:
            break
        candidate = candidate[len(prefix):]
        add(candidate)
    while candidate.startswith("backbone."):
        candidate = candidate[len("backbone."):]
        add(candidate)
    return tuple(variants)


def _log_report(label, component, report, shape_mismatches):
    """Log complete sorted missing/skipped information."""
    logging.info(
        "Loaded %d compatible SegFormer %s pretrained tensors from %s: %s",
        len(report.loaded_keys),
        component,
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


def _log_structured_report(label, component, report):
    """Emit compact machine-readable evidence after a positive load."""
    loaded_keyset = "\n".join(report.loaded_keys).encode("utf-8")
    payload = {
        "checkpoint": label,
        "component": component,
        "loaded_keyset_sha256": hashlib.sha256(loaded_keyset).hexdigest(),
        "loaded_tensor_count": len(report.loaded_keys),
        "missing_tensor_count": len(report.missing_keys),
        "non_tensor_count": len(report.non_tensor_keys),
        "schema_version": 1,
        "shape_mismatched_tensor_count": len(report.shape_mismatched_keys),
        "unmatched_tensor_count": len(report.unmatched_keys),
    }
    logging.info(
        "%s%s",
        PRETRAINED_LOAD_REPORT_PREFIX,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


def _initialize_module_pretrained_weights(
    model,
    path_or_checkpoint,
    *,
    component,
    candidate_keys,
):
    """Load only shape-compatible tensors into one model component."""
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
                for candidate in candidate_keys(source_key)
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
    _log_report(label, component, report, tuple(sorted(shape_mismatches)))

    if not compatible:
        raise RuntimeError(
            "SegFormer pretrained checkpoint "
            f"{label} contains no tensors compatible with the configured {component}."
        )

    model.load_state_dict(compatible, strict=False)
    _log_structured_report(label, component, report)
    return report


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
    return _initialize_module_pretrained_weights(
        pl_model.model,
        path_or_checkpoint,
        component="model",
        candidate_keys=_candidate_model_keys,
    )


def initialize_pretrained_backbone_weights(backbone, path_or_checkpoint):
    """Initialize a bare SegFormer backbone from an official PTM.

    FAN backbone PTMs may contain bare keys or keys wrapped by
    ``module.``, ``_orig_mod.``, ``model.``, and ``backbone.``.  A positive
    compatible-tensor load is mandatory; unrelated checkpoints fail before
    training instead of silently leaving the backbone randomly initialized.
    """
    return _initialize_module_pretrained_weights(
        backbone,
        path_or_checkpoint,
        component="backbone",
        candidate_keys=_candidate_backbone_keys,
    )
