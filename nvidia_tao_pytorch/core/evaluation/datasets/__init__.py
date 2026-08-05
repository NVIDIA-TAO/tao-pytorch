# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation datasets — labeled classification (KNN/retrieval) and segmentation."""

from nvidia_tao_pytorch.core.evaluation.datasets.classification import (  # noqa: F401
    ClassificationLoader,
    build_classification_loader,
    build_eval_transform,
)
from nvidia_tao_pytorch.core.evaluation.datasets.segmentation import (  # noqa: F401
    SEG_DATASETS,
    build_seg_loaders,
)
