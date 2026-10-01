# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the PyTorch Lightning module for the selected NVPanoptix3Dv2 variant.

Both variants share the frozen VGGT backbone and the metric scale head; they
diverge above the geometry layer, so each owns its own Lightning module. The
subtask scripts stay variant-agnostic and go through :func:`build_pl_model`.
"""

PANOPTIC = "panoptic"
REASONING = "reasoning"
SUPPORTED_MODEL_TYPES = (PANOPTIC, REASONING)


def pl_module_class(model_type: str):
    """Import and return the Lightning module class for ``model_type``.

    The imports are deferred because the two variants pull in disjoint and
    heavy third-party stacks (SigLIP for panoptic, Qwen/PEFT/SAM3 for
    reasoning). Importing eagerly would make either variant unusable whenever
    the other's dependencies are absent.
    """
    if model_type == PANOPTIC:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.pl_model import (
            NVPanoptix3Dv2PanopticPlModule,
        )
        return NVPanoptix3Dv2PanopticPlModule
    if model_type == REASONING:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.pl_model import (
            NVPanoptix3Dv2ReasoningPlModule,
        )
        return NVPanoptix3Dv2ReasoningPlModule
    raise ValueError(
        f"Unsupported model.model_type {model_type!r}. "
        f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
    )


def get_pl_module(experiment_config):
    """Return the Lightning module class selected by ``model.model_type``.

    Args:
        experiment_config: the experiment config, with ``model.model_type`` set
            to one of ``panoptic`` or ``reasoning``.

    Returns:
        The Lightning module class for that variant.
    """
    return pl_module_class(str(experiment_config.model.model_type))


def build_pl_model(experiment_config):
    """Instantiate the Lightning module selected by ``model.model_type``.

    Args:
        experiment_config: the experiment config.

    Returns:
        An initialized Lightning module for the configured variant.
    """
    return get_pl_module(experiment_config)(experiment_config)


def load_pl_model(experiment_config, checkpoint_path: str, **kwargs):
    """Load the configured variant's Lightning module from a checkpoint.

    Args:
        experiment_config: the experiment config.
        checkpoint_path: path to the ``.pth`` checkpoint to restore.
        **kwargs: forwarded to ``load_from_checkpoint``.

    Returns:
        The restored Lightning module.
    """
    return get_pl_module(experiment_config).load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        experiment_config=experiment_config,
        **kwargs,
    )
