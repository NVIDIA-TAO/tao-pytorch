# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector module Template."""
from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PointPillar': PointPillar,
}


def build_detector(model_cfg, num_class, dataset):
    """Build the detector."""
    model = __all__[model_cfg.name](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
