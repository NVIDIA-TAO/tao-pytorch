# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transform pipeline builder.

``get_pipeline()`` assembles a ``CompositeTransform`` chain for a single
view (student or teacher).  It handles:

  - Hi-res vs lo-res paths (threshold: 512 px)
  - Unified vs individual deformations
  - Stochastic resolutions with per-teacher patch-size rescaling
  - Shift equivariance (jittered teacher crops)
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from timm.layers import to_2tuple

from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.base import (
    CompositeTransform,
    Identity,
    RandomSize,
    ScopedRNG,
    Size,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.spatial import (
    CenterCropTransform,
    MaxSizeTransform,
    PadToTransform,
    RandomCropTransform,
    RandomFlipTransform,
    RandomPerspectiveTransform,
    RandomRotationTransform,
    RandomZoomTransform,
)

logger = logging.getLogger(__name__)


def _get_zoom(scale: Union[float, Tuple[float, float]] = 1.0, random_seed: int = None, retain_dims: bool = False):
    """Return a RandomZoomTransform or Identity depending on *scale*.

    Args:
        retain_dims: If True, zoom only updates the geometric transform (STM) and
            never shrinks DeferImage.target_size. Set True for teachers so that
            the output is always exactly crop_size regardless of zoom factor.
    """
    if isinstance(scale, float) and scale != 1.0:
        return RandomZoomTransform(min_ratio=scale, max_ratio=1 / scale, retain_dims=retain_dims, random_seed=random_seed)
    if isinstance(scale, tuple):
        return RandomZoomTransform(min_ratio=scale[0], max_ratio=scale[1], retain_dims=retain_dims, random_seed=random_seed)
    return Identity()


def _get_deformations(crop_size: Union[Size, RandomSize],
                      rng: np.random.Generator,
                      unified_seed: int,
                      unified_scale: Union[float, Tuple[float, float]] = 1.0,
                      unified_stretch: float = 1.0,
                      flip_prob: float = 0.0,
                      rot_max: float = 0.0,
                      perspective_scale: Tuple[float, float] = (0, 0),
                      individual_scale: Union[float, Tuple[float, float]] = 1.0,
                      patch_size: int = None,
                      jitter: bool = False,
                      full_equivariance: bool = False,
                      retain_zoom_dims: bool = False,
                      ):
    """Build the deformation sub-pipeline (zoom, crop, flip, rotate, perspective).

    Args:
        retain_zoom_dims: Passed to _get_zoom. When True the zoom only updates
            the spatial transform matrix and never shrinks target_size below
            crop_size. Should be True for all teacher views so the materialised
            tensor is always exactly crop_size regardless of zoom factor.
    """
    ret = []

    jitter = jitter and not full_equivariance

    def get_seed():
        return rng.bit_generator.random_raw()

    ret.append(_get_zoom(unified_scale, random_seed=unified_seed + 41, retain_dims=retain_zoom_dims))

    if unified_stretch != 1.0:
        ret.extend([
            RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1 / unified_stretch, fixed_width=True, retain_dims=False, random_seed=unified_seed + 42),
            RandomZoomTransform(min_ratio=unified_stretch, max_ratio=1 / unified_stretch, fixed_height=True, retain_dims=False, random_seed=unified_seed + 43),
        ])

    ret.append(RandomCropTransform(image_size=crop_size, random_seed=unified_seed + 44,
                                   quant_shift=patch_size if jitter else None, jitter=jitter, jit_seed=get_seed()))

    if full_equivariance:
        if flip_prob > 0:
            ret.append(RandomFlipTransform(prob_apply=flip_prob, random_seed=get_seed()))
        ret.append(_get_zoom(individual_scale, random_seed=get_seed()))

        ret.append(CenterCropTransform(image_size=crop_size))

        if max(perspective_scale) > 0:
            ret.append(RandomPerspectiveTransform(scale=perspective_scale, prob_apply=1.0, random_seed=get_seed()))

        if rot_max > 0:
            ret.append(RandomRotationTransform(abs_max=rot_max, random_seed=get_seed()))

    return ret


def get_pipeline(student_size: Union[int, Tuple[int, int]],
                 img_size: Union[int, Tuple[int, int]],
                 patch_size: int,
                 is_train: bool, is_teacher: bool,
                 max_img_size: int,
                 rng: np.random.Generator,
                 unified_seed: int,
                 full_equivariance: bool = False,
                 shift_equivariance: bool = False,
                 stochastic_size_args: Optional[Dict[str, Any]] = None,
                 stochastic_teacher: bool = False,
                 student_patch_size: Optional[int] = None,
                 scoped_rng: Optional[ScopedRNG] = None,
                 perf_test_simple_aug: bool = False,
                 aug_config: Optional[Dict[str, Any]] = None,
                 ) -> CompositeTransform:
    """Build the transform chain for a single view.

    Args:
        student_size: Student model input size.
        img_size: Target image size for this view.
        patch_size: ViT patch size for this view.
        is_train: Whether the student is training.
        is_teacher: Whether this view is for a teacher.
        max_img_size: Maximum image dimension in the dataset.
        rng: Numpy RNG for seeding sub-transforms.
        unified_seed: Shared seed for transforms that must agree across views.
        full_equivariance: Enable full equivariant augmentation.
        shift_equivariance: Enable shift-equivariant jittered crops for teachers.
        stochastic_size_args: Dict with keys ``min_size``, ``max_size``,
            ``resolutions``, ``fixed_aspect`` for stochastic resolution.
        stochastic_teacher: Whether this teacher uses stochastic resolution.
        student_patch_size: Student's patch size (for teacher resolution rescaling).
        scoped_rng: Shared ``ScopedRNG`` for stochastic resolution.
        perf_test_simple_aug: Minimal resize + center-crop for benchmarking.
        aug_config: Augmentation configuration.

    Returns:
        A ``CompositeTransform`` that takes ``(DeferImage, Quad)`` tuples.
    """
    student_size: Tuple[int, int] = to_2tuple(student_size)
    img_size: Tuple[int, int] = to_2tuple(img_size)

    _aug = aug_config or {}
    _flip_cfg = (_aug.get("random_flip") or {})
    _rot_cfg = (_aug.get("random_rotate") or {})
    _persp_cfg = (_aug.get("perspective_distortion") or {})
    cfg_flip_prob = float(_flip_cfg.get("hflip_probability", 0.1))
    _angle_range = _rot_cfg.get("angle_range", [-5.0, 5.0])
    assert len(_angle_range) == 2, f"angle_range must have exactly 2 elements, got {len(_angle_range)}"
    lo, hi = float(_angle_range[0]), float(_angle_range[1])
    assert abs(lo + hi) < 1e-6, f"WDS pipeline requires symmetric angle_range, got [{lo}, {hi}]"
    cfg_rot_max = hi
    _persp_raw = _persp_cfg.get("scale", [0.01, 0.05])
    cfg_persp_scale = (float(_persp_raw[0]), float(_persp_raw[1]))

    is_hi_res = max_img_size > 512

    jitter_size = patch_size

    transforms = []
    base_size = Size(*img_size)

    if perf_test_simple_aug:
        transforms = [
            MaxSizeTransform(image_size=base_size, smallest=True),
            CenterCropTransform(base_size),
        ]
        return CompositeTransform(transforms)

    if (is_train or stochastic_teacher) and stochastic_size_args:
        assert base_size.width == base_size.height
        if student_patch_size != patch_size:
            def patch_tx(v):
                return v * patch_size // student_patch_size
        else:
            patch_tx = None
        crop_size = RandomSize(
            scoped_rng,
            base_size.height,
            min_size=stochastic_size_args.get('min_size', None),
            max_size=stochastic_size_args.get('max_size', None),
            step_size=patch_size,
            resolutions=stochastic_size_args.get('resolutions', None),
            fixed_aspect=stochastic_size_args.get('fixed_aspect', True),
            transform=patch_tx,
        )
    else:
        crop_size = base_size

    if not is_hi_res:
        transforms.append(MaxSizeTransform(image_size=crop_size, smallest=True))

        if is_train:
            transforms.extend(_get_deformations(
                crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 1.1),
                flip_prob=cfg_flip_prob,
                individual_scale=.95,
                rot_max=cfg_rot_max,
                perspective_scale=cfg_persp_scale,
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance,
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(
                crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(1.0, 1.1),
                patch_size=jitter_size,
                jitter=shift_equivariance,
                retain_zoom_dims=True,
            ))
    else:
        transforms.append(MaxSizeTransform(image_size=crop_size, smallest=True))

        if is_train:
            transforms.extend(_get_deformations(
                crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                rot_max=1.0,
                perspective_scale=cfg_persp_scale,
                patch_size=patch_size,
                jitter=False,
                full_equivariance=full_equivariance,
            ))
        elif is_teacher:
            transforms.extend(_get_deformations(
                crop_size,
                rng=rng,
                unified_seed=unified_seed,
                unified_scale=(0.9, 1.4),
                patch_size=jitter_size,
                jitter=shift_equivariance,
                retain_zoom_dims=True,
            ))

    rand_pad = shift_equivariance and is_train and not is_teacher

    def get_seed():
        return rng.bit_generator.random_raw()

    if is_train and not is_teacher:
        crop = RandomCropTransform(image_size=crop_size, quant_shift=patch_size, random_seed=get_seed())
    else:
        crop = CenterCropTransform(crop_size)

    transforms.extend([
        PadToTransform(image_size=crop_size, quant_pad=patch_size, rand_pad=rand_pad),
        crop,
    ])

    transforms = CompositeTransform(transforms)

    return transforms
