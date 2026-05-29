# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial transform classes for the deferred-image pipeline.

All transforms operate on ``(DeferImage, Quad)`` tuples, accumulating
spatial transforms into the 3x3 STM rather than modifying pixel data.
"""

import logging
import math
from typing import Tuple, Union

import cv2
import numpy
import torch

from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.base import (
    Size,
    RandomSize,
    RandomTransformBase,
    linear_sample,
    log_linear_sample,
    clamped_normal,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.defer_image import DeferImage

logger = logging.getLogger(__name__)


class CropTransform:
    """Crops to a fixed sub-window at (x, y) with given size."""

    def __init__(self, x, y, image_size: Union[Size, RandomSize]):
        if x < 0 or y < 0:
            raise ValueError("x and y must be >= 0")

        self.x = x
        self.y = y
        self.image_size = image_size

        self.translation = torch.tensor([-self.x, -self.y], dtype=torch.float32)

    def __call__(self, *items, **kwargs):
        """Apply a fixed crop to ``items``."""
        crop_size = self.image_size()

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(self.x, self.y, crop_size.width, crop_size.height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(self.translation)


class RandomCropTransform(RandomTransformBase):
    """Randomly crops to a specified size.

    The first item in ``*items`` must have a ``.shape`` attribute.
    """

    def __init__(self, image_size: Union[Size, RandomSize], random_seed: int = None,
                 quant_shift: int = None, jitter: bool = False, jit_seed: int = None):
        self.image_size = image_size

        self.rng = numpy.random.default_rng(random_seed)
        self.quant_shift = quant_shift
        self.jitter = jitter
        self.jitrng = numpy.random.default_rng(jit_seed)

    def __call__(self, *items, crop_window=None, **kwargs):
        """Apply a (possibly deterministic) random crop to ``items``."""
        self._prepare_rng([self.rng, self.jitrng, self.image_size])

        image_shape = items[0].shape

        base_size = self.image_size()

        crop_width = min(base_size.width, image_shape[-1])
        crop_height = min(base_size.height, image_shape[-2])
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        slide_x = image_shape[-1] - crop_width
        slide_y = image_shape[-2] - crop_height

        x_fact = self.rng.random()
        y_fact = self.rng.random()

        x = x_fact * slide_x
        y = y_fact * slide_y

        if self.jitter:
            assert self.quant_shift is not None, "`quant_shift` must be specified when jittering"
            x_jit = round(self.jitrng.normal(scale=0.5))
            y_jit = round(self.jitrng.normal(scale=0.5))

            x2 = x + x_jit * self.quant_shift
            y2 = y + y_jit * self.quant_shift

            if 0 <= x2 <= slide_x:
                x = x2
            if 0 <= y2 <= slide_y:
                y = y2
        elif self.quant_shift is not None:
            x = int(x / self.quant_shift) * self.quant_shift
            y = int(y / self.quant_shift) * self.quant_shift

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))


class CenterCropTransform:
    """Center-crops to a specified size.

    The first item in ``*items`` must have a ``.shape`` attribute.
    """

    def __init__(self, image_size: Size):
        self.image_size = image_size

    def __call__(self, *items, crop_window=None, **kwargs):
        """Apply a center crop to ``items``."""
        image_shape = items[0].shape

        base_size = self.image_size()

        crop_width = base_size.width
        crop_height = base_size.height
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        x = (image_shape[-1] - crop_width) / 2
        y = (image_shape[-2] - crop_height) / 2

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))


class ScaleTransform:
    """Scales items to a target size.

    The first item in ``*items`` must have a ``.shape`` attribute.
    """

    def __init__(self, image_size: Size):
        self.target_size = image_size

    def __call__(self, *items, **kwargs):
        """Apply a scale transform to ``items``."""
        image_size = items[0].shape

        target_size = self.target_size()

        height_factor = float(target_size.height) / image_size[-2]
        width_factor = float(target_size.width) / image_size[-1]

        for item in items:
            apply_fn = getattr(item, 'scale', None)
            if apply_fn:
                apply_fn(torch.tensor([width_factor, height_factor]))


class MinSizeTransform:
    """Scales to ensure both dimensions are at least the specified size.

    Preserves aspect ratio. No-op if both dimensions already exceed the
    minimums.
    """

    def __init__(self, min_width, min_height):
        if min_width < 1 or min_height < 1:
            raise ValueError("Target sizes must be at least 1.")
        self.min_width = int(round(min_width))
        self.min_height = int(round(min_height))
        self.aspect = self.min_width / self.min_height

    def __call__(self, *items, **kwargs):
        """Scale ``items`` up so their shortest side meets the minimum size."""
        image_size = items[0].shape

        image_aspect = image_size[-1] / image_size[-2]

        if self.aspect < image_aspect:
            scale = self.min_height / image_size[-2]
        else:
            scale = self.min_width / image_size[-1]

        if scale <= 1.0:
            return

        scale_tensor = torch.tensor([scale, scale], dtype=torch.float32)

        for item in items:
            apply_fn = getattr(item, 'scale', None)
            if apply_fn:
                apply_fn(scale_tensor)


class MaxSizeTransform:
    """Scales so that the largest (or smallest) dimension fits the target.

    Args:
        image_size: target size.
        smallest: if True, constrain the *smallest* dimension instead.
    """

    def __init__(self, image_size: Union[Size, RandomSize], smallest: bool = False):
        self.image_size = image_size
        self.smallest = smallest

    def __call__(self, *items, **kwargs):
        """Scale ``items`` down so their longest side respects the maximum size."""
        size = self.image_size()

        image_size = items[0].shape
        scale_1 = size.width / image_size[-1]
        scale_2 = size.height / image_size[-2]

        def comp(a, b):
            if not self.smallest:
                return a < b
            else:
                return b < a

        scale = scale_1 if comp(scale_1, scale_2) else scale_2

        scale_tensor = torch.tensor([scale, scale], dtype=torch.float32)

        for item in items:
            apply_fn = getattr(item, 'scale', None)

            if apply_fn:
                apply_fn(scale_tensor)


class RandomZoomTransform(RandomTransformBase):
    """Randomly zooms items by a scale factor.

    Zooming is performed about the center of the image.
    """

    def __init__(self, min_ratio, max_ratio, retain_dims=True, batch_size=1,
                 fixed_width=False, fixed_height=False, random_seed: int = None):
        if min_ratio <= 0:
            raise ValueError("Min ratio must be > 0")
        if max_ratio <= min_ratio:
            raise ValueError("max_ratio must be > min_ratio")

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.batch_size = batch_size
        self.counter = 0
        self.scale = 1.0
        self.retain_dims = retain_dims
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """Apply a random zoom (possibly per-axis) to ``items``."""
        self._prepare_rng(self.rng)

        image_size = items[0].shape

        if self.counter == 0:
            if self.min_ratio < 1 < self.max_ratio:
                self.scale = log_linear_sample(self.min_ratio, self.max_ratio, self.rng)
            else:
                self.scale = linear_sample(self.min_ratio, self.max_ratio, self.rng)
        self.counter = (self.counter + 1) % self.batch_size

        scale_w = self.scale if not self.fixed_width else 1
        scale_h = self.scale if not self.fixed_height else 1
        scale_tensor = torch.tensor([scale_w, scale_h], dtype=torch.float32)

        trans_tensor = torch.tensor([-image_size[-1] / 2, -image_size[-2] / 2], dtype=torch.float32)
        trans_tensor2 = torch.tensor([image_size[-1] / 2 * scale_w, image_size[-2] / 2 * scale_h], dtype=torch.float32)

        for item in items:
            scale_fn = getattr(item, 'scale', None)
            trans_fn = getattr(item, 'translate', None)

            if trans_fn is not None:
                trans_fn(trans_tensor)

            if scale_fn is not None:
                scale_fn(scale_tensor, retain_dims=self.retain_dims)

            if trans_fn is not None:
                trans_fn(trans_tensor2)


class RandomFlipTransform(RandomTransformBase):
    """Randomly flips items horizontally about the image center."""

    def __init__(self, prob_apply=0.5, batch_size=1, random_seed: int = None):
        if prob_apply < 0 or prob_apply > 1:
            raise ValueError("prob_apply must be between 0 and 1 inclusive.")

        self.prob_apply = prob_apply
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """Apply a random horizontal flip to ``items``."""
        self._prepare_rng(self.rng)

        if self.counter == 0:
            r_val = self.rng.random()
            self.do_apply = r_val <= self.prob_apply
        self.counter = (self.counter + 1) % self.batch_size

        if not self.do_apply:
            return

        image_size = items[0].shape

        for item in items:
            flip_fn = getattr(item, 'flip', None)

            if flip_fn is not None:
                flip_fn(image_size[-1] / 2)


class RandomTranslationTransform(RandomTransformBase):
    """Randomly translates items along both axes."""

    def __init__(self, std_dev, abs_max=0, batch_size=1, random_seed: int = None):
        self.std_dev = std_dev ** 2
        self.abs_max = abs_max
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """Apply a random translation to ``items``."""
        self._prepare_rng(self.rng)

        if self.counter == 0:
            self.translation = numpy.array([
                clamped_normal(self.std_dev, -self.abs_max, self.abs_max, rng=self.rng),
                clamped_normal(self.std_dev, -self.abs_max, self.abs_max, rng=self.rng),
            ])
        self.counter = (self.counter + 1) % self.batch_size

        # EVFM uses a local variable here which is a latent bug for
        # batch_size > 1; using self.translation is the intended behavior.
        trans_vec = torch.from_numpy(self.translation)

        for item in items:
            trans_fn = getattr(item, 'translate', None)

            if trans_fn is not None:
                trans_fn(trans_vec)


class RandomRotationTransform(RandomTransformBase):
    """Randomly rotates items about the image center."""

    def __init__(self, abs_max=90, batch_size=1, random_seed: int = None):
        self.abs_max = abs_max
        self.batch_size = batch_size
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """Apply a random rotation to ``items``."""
        self._prepare_rng(self.rng)

        if self.counter == 0:
            rotation = linear_sample(-self.abs_max, self.abs_max, rng=self.rng)
            rotation = (numpy.pi / 180) * rotation
            self.rotation = rotation
        self.counter = (self.counter + 1) % self.batch_size

        image_size = items[0].shape

        translate_x = image_size[-1] / 2
        translate_y = image_size[-2] / 2

        trans_vec = torch.tensor([-translate_x, -translate_y],
                                 dtype=torch.float32)

        rotation = torch.tensor([
            [math.cos(self.rotation), -math.sin(self.rotation)],
            [math.sin(self.rotation), math.cos(self.rotation)]
        ], dtype=torch.float32)

        for item in items:
            trans_fn = getattr(item, 'translate', None)
            rot_fn = getattr(item, 'rotate', None)

            if (trans_fn is None) ^ (rot_fn is None):
                raise ValueError("One of 'translate' or 'rotate' is defined, but not both.")

            if trans_fn is not None:
                trans_fn(trans_vec)
                rot_fn(rotation)

                tv2 = torch.tensor([
                    items[0].shape[-1] / 2,
                    items[0].shape[-2] / 2
                ], dtype=torch.float32)

                trans_fn(tv2)


class RandomPerspectiveTransform(RandomTransformBase):
    """Randomly applies a perspective deformation.

    Uses ``cv2.getPerspectiveTransform`` to compute the 3x3 homography
    from randomly offset corners.
    """

    def __init__(self, scale: Union[float, Tuple[float, float]] = (0.05, 0.1),
                 batch_size=1, prob_apply=0.5, random_seed: int = None):
        if not isinstance(scale, tuple):
            scale = (0.0, scale)
        self.scale = scale
        self.batch_size = batch_size
        self.prob_apply = prob_apply
        self.counter = 0

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, **kwargs):
        """Apply a random perspective warp to ``items``."""
        self._prepare_rng(self.rng)

        if self.counter == 0:
            scale = linear_sample(*self.scale, self.rng)
            offsets = clamped_normal(scale, -0.45, 0.45, self.rng, size=(4, 2)).astype(numpy.float32)
            self.sampled_offsets = offsets

        if self.rng.random() > self.prob_apply:
            return

        self.counter = (self.counter + 1) % self.batch_size

        image: DeferImage = items[0]

        h, w = image.shape[-2:]
        in_bds = numpy.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=numpy.float32)

        offsets = numpy.copy(self.sampled_offsets)
        offsets[:, 0] *= w
        offsets[:, 1] *= h

        out_bds = in_bds + offsets
        min_out = numpy.min(out_bds, axis=0)
        out_bds -= min_out

        H = cv2.getPerspectiveTransform(in_bds, out_bds)

        H = torch.from_numpy(H).float()

        H_inv = torch.inverse(H)

        image.apply_stm(H_inv)
        for i in range(1, len(items)):
            items[i].apply_stm(H.T, perspective=True)

        target_width = numpy.max(out_bds[:, 0]) - numpy.min(out_bds[:, 0])
        target_height = numpy.max(out_bds[:, 1]) - numpy.min(out_bds[:, 1])

        image.target_size = image.target_size[:-2] + (target_height, target_width)


class PadToTransform(RandomTransformBase):
    """Pads items to at least the specified size.

    Supports random pad placement, quantised to ``quant_pad`` boundaries.
    """

    def __init__(self, image_size: Union[Size, RandomSize], random_seed: int = None,
                 quant_pad: int = None, rand_pad: bool = True):
        self.image_size = image_size
        self.quant_pad = quant_pad
        self.rand_pad = rand_pad

        self.rng = numpy.random.default_rng(random_seed)

    def __call__(self, *items, crop_window=None, **kwargs):
        """Pad ``items`` to the configured size."""
        self._prepare_rng(self.rng)

        crop_size = self.image_size()

        crop_width = crop_size.width
        crop_height = crop_size.height
        if crop_window is not None:
            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

        image_size = items[0].shape
        crop_width = max(crop_width, image_size[-1])
        crop_height = max(crop_height, image_size[-2])

        pad_w = crop_width - image_size[-1]
        pad_h = crop_height - image_size[-2]

        if pad_w == 0 and pad_h == 0:
            return

        x = (self.rng.random() if self.rand_pad else 0) * -pad_w
        y = (self.rng.random() if self.rand_pad else 0) * -pad_h

        if self.quant_pad is not None:
            x = int(x / self.quant_pad) * self.quant_pad
            y = int(y / self.quant_pad) * self.quant_pad

        for item in items:
            apply_fn = getattr(item, 'clip_translate', None)
            if apply_fn:
                apply_fn(x, y, crop_width, crop_height)
            else:
                apply_fn = getattr(item, 'translate', None)
                if apply_fn:
                    apply_fn(torch.tensor([-x, -y], dtype=torch.float32))
