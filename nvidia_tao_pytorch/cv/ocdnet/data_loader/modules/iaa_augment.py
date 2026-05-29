#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augmentation module backed by albumentations.

Replaces the previous imgaug-based implementation. imgaug is unmaintained
since 2020 and breaks on NumPy 2.x (np.sctypes, np.complex were removed).
"""
import albumentations as A
import numpy as np


def _as_range(value):
    """Coerce a list/tuple range from YAML into the (low, high) tuple albumentations expects."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _build_op(op_type, args):
    """Map an imgaug-style spec ({type, args}) onto an albumentations transform.

    Supports the ops actually invoked from OCDNet experiment specs (Fliplr,
    Affine, Resize, Sometimes, GaussianBlur). Add more here if a new spec
    needs them.
    """
    args = dict(args or {})
    if op_type == 'Fliplr':
        return A.HorizontalFlip(p=args.get('p', 0.5))
    if op_type == 'Affine':
        return A.Affine(
            rotate=_as_range(args.get('rotate', 0)),
            scale=_as_range(args.get('scale', 1.0)),
            translate_percent=_as_range(args.get('translate_percent', 0)),
            shear=_as_range(args.get('shear', 0)),
            p=args.get('p', 1.0),
        )
    if op_type == 'Resize':
        # imgaug `Resize(size=[a, b])` samples a uniform scale factor in [a, b]
        # and changes the actual output dimensions (not just the content within
        # a fixed canvas). `A.Affine(scale=...)` would only shrink/grow content,
        # so we use `A.RandomScale`, whose `scale_limit` is signed delta from 1.
        size = args.get('size', 1.0)
        if isinstance(size, (list, tuple)):
            scale_limit = (size[0] - 1.0, size[1] - 1.0)
        else:
            scale_limit = (size - 1.0, size - 1.0)
        return A.RandomScale(scale_limit=scale_limit, p=args.get('p', 1.0))
    if op_type == 'GaussianBlur':
        return A.GaussianBlur(
            sigma_limit=_as_range(args.get('sigma', (0, 0))),
            blur_limit=0,  # let albumentations derive kernel size from sigma
            p=args.get('p', 1.0),
        )
    if op_type == 'Sometimes':
        # imgaug `Sometimes(p, then_list=X)` ≡ "apply X with probability p".
        # `then_list` may be a single dict or a list of dicts.
        outer_p = args.get('p', 0.5)
        then_list = args.get('then_list', [])
        if isinstance(then_list, dict):
            then_list = [then_list]
        if len(then_list) == 1:
            spec = then_list[0]
            child_args = dict(spec.get('args', {}))
            # albumentations has no Sometimes wrapper; push the gate into the
            # child's native `p` so it fires with probability outer_p.
            child_args['p'] = outer_p
            return _build_op(spec['type'], child_args)
        # Multiple children: wrap in a sub-Compose gated by outer_p so the
        # block fires atomically (matches imgaug semantics for then_list).
        children = [_build_op(s['type'], s.get('args', {})) for s in then_list]
        return A.Compose(children, p=outer_p)
    raise ValueError(f"Unsupported augmenter type: {op_type!r}")


class AugmenterBuilder:
    """Builds an albumentations Compose pipeline from imgaug-style YAML specs."""

    def build(self, augmenter_args):
        """Build the pipeline.

        augmenter_args is a list of `{'type': <name>, 'args': {...}}` dicts (or
        positional `[name, *args]` lists, kept for backward compatibility with
        older specs).
        """
        if not augmenter_args:
            return None
        ops = []
        for spec in augmenter_args:
            if isinstance(spec, dict):
                op_type, args = spec['type'], spec.get('args', {})
            elif isinstance(spec, list) and spec:
                op_type, args = spec[0], (spec[1] if len(spec) > 1 else {})
            else:
                raise RuntimeError(f"unknown augmenter arg: {spec!r}")
            ops.append(_build_op(op_type, args))
        return A.Compose(
            ops,
            # `remove_invisible=False` preserves OOB keypoints so polygons keep
            # their vertex count after augmentation (the downstream
            # MakeBorderMap/MakeShrinkMap stages rely on the (N, 4, 2) shape).
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )


class IaaAugment():
    """Apply augmenters to an image and its associated text polygons in lockstep."""

    def __init__(self, augmenter_args):
        """Initialize."""
        self.augmenter_args = augmenter_args
        self.transform = AugmenterBuilder().build(self.augmenter_args)

    def __call__(self, data):
        """Augment the image and co-transform polygon keypoints with the same params."""
        if self.transform is None:
            return data

        image = data['img']
        polys = data.get('text_polys', [])

        flat_keypoints = []
        poly_lengths = []
        for poly in polys:
            poly_lengths.append(len(poly))
            flat_keypoints.extend((float(p[0]), float(p[1])) for p in poly)

        result = self.transform(image=image, keypoints=flat_keypoints)
        data['img'] = result['image']

        new_keypoints = result['keypoints']
        new_polys = []
        cursor = 0
        for n in poly_lengths:
            new_polys.append([(kp[0], kp[1]) for kp in new_keypoints[cursor:cursor + n]])
            cursor += n
        data['text_polys'] = np.array(new_polys) if new_polys else np.empty((0, 0, 2))
        return data
