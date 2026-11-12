# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 dataloader module."""

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import (
    PANOPTIC, REASONING, SUPPORTED_MODEL_TYPES,
)


def build_pl_data_module(experiment_config):
    """Build the LightningDataModule for the configured variant.

    Args:
        experiment_config: the experiment config, with ``model.model_type`` set
            to one of ``panoptic`` or ``reasoning``.

    Returns:
        An initialized LightningDataModule reading ``dataset.<variant>``.
    """
    model_type = str(experiment_config.model.model_type)
    if model_type == PANOPTIC:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
            build_data_module,
        )
        return build_data_module(
            experiment_config.dataset.panoptic,
            seed=experiment_config.train.seed,
        )
    if model_type == REASONING:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.reasoning.pl_data_module import (
            build_data_module,
        )
        return build_data_module(experiment_config.dataset.reasoning)
    raise ValueError(
        f"Unsupported model.model_type {model_type!r}. "
        f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
    )
