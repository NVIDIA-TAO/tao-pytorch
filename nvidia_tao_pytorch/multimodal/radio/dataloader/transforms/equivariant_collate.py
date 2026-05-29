# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivariant batch collation for multi-view distillation.

Ported from ``data/transforms/equivariant_collate.py``.

The factory ``equivariant_collate(num_images, ...)`` returns a batch
collation function that:

1. Materialises each ``DeferImage`` via ``__call__()``
2. Generates ``valid_mask`` from ``Quad.bounds`` via ``cv2.fillPoly``
3. Computes teacher-to-student homographies via ``cv2.getPerspectiveTransform``

**Adaptation**: The output format is changed from EVFM's list-of-tuples to
the dict format expected by TAO's ``MultiTeacherDistiller``::

    {
        "img":            (B, C, H, W),          # student images
        "class":          (B,),                   # class labels (NOCLASS_IDX if absent)
        "valid_mask":     (B, H, W),              # student valid region
        "teacher_views":  [                       # per-teacher dicts
            {
                "img":              (B, C, H, W),
                "valid_mask":       (B, H, W),
                "spatial_transform": (B, 3, 3),   # teacher→student homography
            },
            ...
        ],
    }
"""

import logging
from typing import Any, Callable, Dict, Iterable, List

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

NOCLASS_IDX = -1


def _quad_from_shape(shape):
    """Return a 4-corner float32 array ``[[0,0],[W,0],[W,H],[0,H]]``."""
    H, W = shape[-2:]
    return np.array([
        [0, 0], [W, 0], [W, H], [0, H],
    ], dtype=np.float32)


_SAMPLE_BOUNDS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)


def equivariant_collate(
    num_images: int,
    patch_sizes: List[int],
    extra_map: Dict[int, str] = None,
    label_key: str = "label",
) -> Callable[..., Iterable]:
    """Factory returning a batch collation function.

    Args:
        num_images: Number of views (1 student + N teachers).
        patch_sizes: Per-view ViT patch sizes (used only for debug grid).
        extra_map: Maps extra-field offset to a name string.  If a name
            matches *label_key* it is placed in ``batch["class"]``.
        label_key: Name used in *extra_map* for the class label
            (default ``"label"``).

    Returns:
        A callable ``_stage(batch) -> dict`` suitable for
        ``wds.batched(collation_fn=...)``.
    """
    if extra_map is None:
        extra_map = {}

    def _stage(batch: List[List[Any]]) -> dict:
        # Per-view accumulators
        images: List[List[torch.Tensor]] = [[] for _ in range(num_images)]
        valid_masks: List[List[torch.Tensor]] = [[] for _ in range(num_images)]
        transforms: List[List[torch.Tensor]] = [[] for _ in range(num_images - 1)]

        # Extra-field accumulators (everything beyond the num_images view slots)
        num_extras = len(batch[0]) - num_images
        extras: List[list] = [[] for _ in range(num_extras)]

        for ex in batch:
            for i in range(num_images):
                im, bds = ex[i]
                im = im()

                teacher_crop_bounds = _quad_from_shape(im.shape)

                valid_mask = torch.zeros_like(im[0])
                cv2.fillPoly(
                    valid_mask.numpy(),
                    pts=[bds.bounds.round().int().numpy()],
                    color=(1,),
                )

                images[i].append(im)
                valid_masks[i].append(valid_mask)

                if i > 0:
                    student_tx_bounds = ex[0][1].bounds.numpy()
                    student_crop_bounds = _quad_from_shape(images[0][-1].shape)
                    teacher_tx_bounds = bds.bounds.numpy()

                    tx_sample_to_teacher_crop = cv2.getPerspectiveTransform(
                        _SAMPLE_BOUNDS, teacher_crop_bounds,
                    )
                    tx_student_to_teacher = cv2.getPerspectiveTransform(
                        teacher_tx_bounds, student_tx_bounds,
                    )
                    tx_crop_student_to_sample = cv2.getPerspectiveTransform(
                        student_crop_bounds, _SAMPLE_BOUNDS,
                    )

                    tx_teacher_sample_to_student_crop = np.matmul(
                        tx_student_to_teacher, tx_sample_to_teacher_crop,
                    )
                    tx_teacher_sample_to_student_sample = np.matmul(
                        tx_crop_student_to_sample, tx_teacher_sample_to_student_crop,
                    )

                    transform_mat = torch.from_numpy(
                        tx_teacher_sample_to_student_sample
                    ).float()
                    transforms[i - 1].append(transform_mat)

            for j in range(num_extras):
                extras[j].append(ex[num_images + j])

        # Stack per-view tensors
        for g in range(num_images):
            images[g] = torch.stack(images[g])
            valid_masks[g] = torch.stack(valid_masks[g])
        for t in range(num_images - 1):
            transforms[t] = torch.stack(transforms[t])

        # Collect named extras
        named_extras = {}
        for j in range(num_extras):
            name = extra_map.get(j, f"extra_{j}")
            vals = extras[j]
            if not isinstance(vals[0], str):
                try:
                    vals = torch.tensor(vals)
                except (ValueError, TypeError):
                    pass
            named_extras[name] = vals

        # Build TAO dict format
        class_labels = named_extras.pop(label_key, None)
        if class_labels is None:
            class_labels = torch.full(
                (images[0].shape[0],), NOCLASS_IDX, dtype=torch.long,
            )

        result = {
            "img": images[0],
            "class": class_labels,
            "valid_mask": valid_masks[0],
            "teacher_views": [
                {
                    "img": images[i],
                    "valid_mask": valid_masks[i],
                    "spatial_transform": transforms[i - 1],
                }
                for i in range(1, num_images)
            ],
        }

        # Attach remaining extras at top level
        for k, v in named_extras.items():
            result[k] = v

        return result

    return _stage
