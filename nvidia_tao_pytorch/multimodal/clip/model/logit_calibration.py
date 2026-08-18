# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical logit calibration helpers for CLIP-compatible models."""

import math

import torch
import torch.nn as nn


DEFAULT_MAX_LOGIT_SCALE = math.log(100)


def default_logit_calibration(loss_type):
    """Return fallback raw scale and bias for a contrastive loss family."""
    if loss_type == 'siglip':
        return 2.3026, -10.0
    return 2.6592, 0.0


def scalar_override(value, name):
    """Convert a scalar calibration override to float with a clear error."""
    try:
        tensor = torch.as_tensor(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar, got {value!r}.") from exc
    if tensor.numel() != 1:
        raise ValueError(
            f"{name} must be a scalar, got shape {tuple(tensor.shape)}."
        )
    return float(tensor.item())


def _new_scalar_parameter(owner, value, name, reference=None):
    """Create a scalar parameter on the source dtype and device."""
    scalar = scalar_override(value, name)
    if reference is None:
        reference = next(owner.parameters(), None)
    if reference is None or not reference.is_floating_point():
        tensor = torch.tensor(scalar)
    else:
        tensor = reference.detach().new_tensor(scalar)
    return nn.Parameter(tensor)


def validate_logit_parameter(parameter, name, *, required):
    """Validate a canonical calibration parameter."""
    if parameter is None:
        if required:
            raise ValueError(f"Model does not expose required {name} parameter.")
        return None
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(
            f"Model {name} must be an nn.Parameter, got "
            f"{type(parameter).__name__}."
        )
    if parameter.numel() != 1:
        raise ValueError(
            f"Model {name} must be scalar, got shape {tuple(parameter.shape)}."
        )
    return parameter


def _normalize_scalar_parameter_(parameter):
    """Normalize a scalar parameter to 0-D without replacing its object."""
    if parameter.ndim:
        with torch.no_grad():
            parameter.set_(parameter.detach().reshape(()))
    return parameter


def _copy_override(parameter, value, name):
    """Copy an override into a canonical parameter without replacing it."""
    if value is None:
        return
    scalar = scalar_override(value, name)
    with torch.no_grad():
        parameter.fill_(scalar)


def preserve_logit_scale_ceiling_(model, value=None):
    """Keep the historical bound at or above initialized/restored scale."""
    if value is None:
        value = resolve_logit_scale_parameter(model)
    scalar = scalar_override(value, 'logit_scale')
    maximum = getattr(model, 'logit_scale_max', DEFAULT_MAX_LOGIT_SCALE)
    if maximum is None:
        maximum = DEFAULT_MAX_LOGIT_SCALE
    model.logit_scale_max = max(
        DEFAULT_MAX_LOGIT_SCALE,
        float(maximum),
        scalar,
    )


def configure_source_logit_calibration(
    owner,
    logit_scale_init=None,
    logit_bias_init=None,
    loss_type='siglip',
    bias_required=True,
):
    """Configure calibration directly on its canonical source module.

    Existing source parameters are preserved when their override is ``None``.
    Missing parameters are created on ``owner`` from an explicit override or
    the loss-family fallback. Optional CLIP bias remains absent when no
    override is requested. Scalar source parameters are normalized to 0-D in
    place while retaining parameter identity and exported scalar shapes.
    """
    fallback_scale, fallback_bias = default_logit_calibration(loss_type)

    logit_scale = getattr(owner, 'logit_scale', None)
    source_logit_scale = logit_scale
    if logit_scale is None:
        value = fallback_scale if logit_scale_init is None else logit_scale_init
        logit_scale = _new_scalar_parameter(
            owner, value, 'init_logit_scale'
        )
        owner.logit_scale = logit_scale
    logit_scale = _normalize_scalar_parameter_(validate_logit_parameter(
        logit_scale, 'logit_scale', required=True
    ))
    if source_logit_scale is not None:
        preserve_logit_scale_ceiling_(owner, logit_scale)
    else:
        maximum = getattr(owner, 'logit_scale_max', DEFAULT_MAX_LOGIT_SCALE)
        if maximum is None:
            maximum = DEFAULT_MAX_LOGIT_SCALE
        owner.logit_scale_max = max(
            DEFAULT_MAX_LOGIT_SCALE, float(maximum)
        )
    _copy_override(logit_scale, logit_scale_init, 'init_logit_scale')

    logit_bias = getattr(owner, 'logit_bias', None)
    if logit_bias is None and (bias_required or logit_bias_init is not None):
        value = fallback_bias if logit_bias_init is None else logit_bias_init
        logit_bias = _new_scalar_parameter(
            owner,
            value,
            'init_logit_bias',
            reference=logit_scale,
        )
        owner.logit_bias = logit_bias
    logit_bias = validate_logit_parameter(
        logit_bias, 'logit_bias', required=bias_required
    )
    if logit_bias is not None:
        logit_bias = _normalize_scalar_parameter_(logit_bias)
        _copy_override(logit_bias, logit_bias_init, 'init_logit_bias')

    return logit_scale, logit_bias


def _reject_legacy_logit_checkpoint(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Add a targeted load error for obsolete calibration layouts."""
    del local_metadata, strict, missing_keys, unexpected_keys
    expected_state = module.state_dict()
    expected_keys = {prefix + key for key in expected_state}
    incompatible_keys = []

    for name in ('logit_scale', 'logit_bias'):
        root_key = prefix + name
        if root_key in state_dict and root_key not in expected_keys:
            incompatible_keys.append(root_key)

    adaptor_prefix = prefix + 'adaptor.'
    incompatible_keys.extend(
        key for key in state_dict
        if key.startswith(adaptor_prefix) and key not in expected_keys
    )

    for local_key, expected_value in expected_state.items():
        if not local_key.endswith(('logit_scale', 'logit_bias')):
            continue
        checkpoint_key = prefix + local_key
        checkpoint_value = state_dict.get(checkpoint_key)
        if isinstance(checkpoint_value, torch.Tensor) and (
            checkpoint_value.shape != expected_value.shape
        ):
            incompatible_keys.append(checkpoint_key)

    if incompatible_keys:
        joined_keys = ', '.join(sorted(set(incompatible_keys)))
        error_msgs.append(
            'This checkpoint predates CLIP logit calibration ownership '
            'unification and is incompatible with the current model layout '
            f'(affected keys: {joined_keys}). Automatic migration is '
            'unsupported because legacy outer and nested calibration values '
            'can differ.'
        )


def register_logit_checkpoint_guard(model):
    """Reject obsolete CLIP calibration layouts with a clear diagnostic.

    Calibration ownership changed state-dict paths and scalar shapes. Legacy
    checkpoints are intentionally not migrated because their duplicated outer
    and nested values may disagree about which state should be authoritative.
    """
    if getattr(model, '_logit_checkpoint_guard_registered', False) is True:
        return
    model.register_load_state_dict_pre_hook(
        _reject_legacy_logit_checkpoint
    )
    model._logit_checkpoint_guard_registered = True


def resolve_logit_scale_parameter(model):
    """Return a model's canonical raw logit-scale parameter."""
    getter = getattr(model, 'get_logit_scale_parameter', None)
    parameter = (
        getter() if callable(getter) else getattr(model, 'logit_scale', None)
    )
    return validate_logit_parameter(
        parameter, 'logit_scale', required=True
    )


def resolve_logit_bias_parameter(model):
    """Return a model's optional canonical logit-bias parameter."""
    getter = getattr(model, 'get_logit_bias_parameter', None)
    parameter = (
        getter() if callable(getter) else getattr(model, 'logit_bias', None)
    )
    return validate_logit_parameter(
        parameter, 'logit_bias', required=False
    )


def named_logit_parameters(model):
    """Yield canonical calibration parameters exactly once in stable order."""
    yield 'logit_scale', resolve_logit_scale_parameter(model)
    bias = resolve_logit_bias_parameter(model)
    if bias is not None:
        yield 'logit_bias', bias


def clamp_logit_scale_(model):
    """Apply the model's raw logit-scale bounds in place."""
    clamp = getattr(model, 'clamp_logit_scale_', None)
    if callable(clamp):
        clamp()
        return

    scale = resolve_logit_scale_parameter(model)
    maximum = getattr(model, 'logit_scale_max', DEFAULT_MAX_LOGIT_SCALE)
    with torch.no_grad():
        scale.clamp_(min=0, max=maximum)
