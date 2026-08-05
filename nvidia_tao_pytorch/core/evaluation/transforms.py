# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from vfm-eval/c-radiov4 common/transforms.py (Apache-2.0) — preserved
# EXACTLY to keep segmentation mIoU comparable to the paper numbers.

"""Segmentation augmentation transforms matching the mmseg pipeline (RADIO repo).

Train pipeline (mmseg radio_linear_8xb2-80k_ade20k-512x512.py + ade20k.py):
    RandomResize(scale=(2048,512), ratio_range=(0.5,2.0), keep_ratio=True)
    RandomCrop(crop_size, cat_max_ratio=0.75)
    RandomFlip(prob=0.5)
    PhotoMetricDistortion
    Pad(size_divisor=patch_size)
    [normalize]

Val pipeline:
    Resize(scale=(2048,512), keep_ratio=True)
    Pad(size_divisor=patch_size)
    [normalize]

All transforms operate on ``(PIL.Image[RGB], PIL.Image[mask])`` pairs.
``PhotoMetricDistortion`` uses cv2 (BGR↔HSV) to match mmseg's mmcv internals.
``Normalize`` converts to a float tensor in ``[0, 1]`` and, by default, does NOT
apply mean/std — RADIO normalizes internally via its ``input_conditioner``. Pass
``mean``/``std`` for backbones whose adapter does not normalize (e.g. NV-DINOv2).
"""

import math
import random

import cv2
import numpy as np
import torch
from PIL import Image

# CLIP normalization constants (CLIPImageProcessor for C-RADIOv4-H).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pil_to_cv2_rgb(img: Image.Image) -> np.ndarray:
    """PIL RGB → uint8 numpy RGB array."""
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _cv2_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    """uint8 numpy RGB array → PIL RGB image."""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


class Compose:
    """Apply a sequence of ``(image, mask) → (image, mask)`` transforms."""

    def __init__(self, transforms):
        """Store the ordered transform list."""
        self.transforms = transforms

    def __call__(self, image, mask):
        """Run each transform in order."""
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomResizeKeepRatio:
    """mmseg ``RandomResize(scale=(2048,512), ratio_range=(0.5,2.0), keep_ratio=True)``."""

    def __init__(self, scale=(2048, 512), ratio_range=(0.5, 2.0)):
        """Init long/short side targets and ratio range."""
        self.long_side = max(scale)
        self.short_side = min(scale)
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, image, mask):
        """Resize keeping aspect ratio with a random scale factor."""
        r = random.uniform(self.min_ratio, self.max_ratio)
        target_short = self.short_side * r
        target_long = self.long_side * r

        w, h = image.size
        sf = target_short / min(h, w)
        if max(h, w) * sf > target_long:
            sf = target_long / max(h, w)

        new_w = max(int(round(w * sf)), 1)
        new_h = max(int(round(h * sf)), 1)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask


class ResizeKeepRatio:
    """mmseg ``Resize(scale=(2048,512), keep_ratio=True)`` — val only (r=1)."""

    def __init__(self, scale=(2048, 512)):
        """Init long/short side targets."""
        self.long_side = max(scale)
        self.short_side = min(scale)

    def __call__(self, image, mask):
        """Resize keeping aspect ratio so the short side hits the target."""
        w, h = image.size
        sf = self.short_side / min(h, w)
        if max(h, w) * sf > self.long_side:
            sf = self.long_side / max(h, w)

        new_w = max(int(round(w * sf)), 1)
        new_h = max(int(round(h * sf)), 1)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask


class RandomCropWithCatMaxRatio:
    """mmseg ``RandomCrop(crop_size, cat_max_ratio=0.75)`` with retry on class dominance."""

    def __init__(self, crop_size=(512, 512), cat_max_ratio: float = 0.75,
                 ignore_index: int = 255, max_tries: int = 10):
        """Init crop size, dominance ratio, ignore index, retry budget."""
        self.crop_h, self.crop_w = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.max_tries = max_tries

    def _pad(self, image, mask):
        """Pad up to the crop size (image with 0, mask with ignore_index)."""
        w, h = image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h == 0 and pad_w == 0:
            return image, mask

        new_img = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
        new_img.paste(image, (0, 0))

        mask_arr = np.array(mask)
        padded = np.full((h + pad_h, w + pad_w), self.ignore_index, dtype=mask_arr.dtype)
        padded[:h, :w] = mask_arr
        return new_img, Image.fromarray(padded)

    def __call__(self, image, mask):
        """Crop, retrying until no class exceeds ``cat_max_ratio`` of valid px."""
        image, mask = self._pad(image, mask)
        w, h = image.size
        mask_arr = np.array(mask)

        top = left = 0
        for attempt in range(self.max_tries):
            top = random.randint(0, h - self.crop_h)
            left = random.randint(0, w - self.crop_w)
            crop = mask_arr[top:top + self.crop_h, left:left + self.crop_w]
            valid = crop[crop != self.ignore_index]
            if len(valid) == 0:
                break
            _, counts = np.unique(valid, return_counts=True)
            if counts.max() / len(valid) < self.cat_max_ratio:
                break
            if attempt == self.max_tries - 1:
                break

        image = image.crop((left, top, left + self.crop_w, top + self.crop_h))
        crop_arr = mask_arr[top:top + self.crop_h, left:left + self.crop_w]
        mask = Image.fromarray(crop_arr)
        return image, mask


class RandomFlip:
    """Horizontal flip with probability ``prob``."""

    def __init__(self, prob: float = 0.5):
        """Init flip probability."""
        self.prob = prob

    def __call__(self, image, mask):
        """Flip image (and mask) left-right with probability ``prob``."""
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class PhotoMetricDistortion:
    """mmseg/mmdet ``PhotoMetricDistortion`` (cv2 BGR↔HSV), preserved exactly."""

    def __init__(self, brightness_delta: float = 32.0, contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_delta: float = 18.0):
        """Init photometric jitter ranges (cv2 uint8 H is half-degrees → hue/2)."""
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta_cv2 = hue_delta / 2.0

    def _convert(self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """Apply ``img * alpha + beta`` and clip to uint8."""
        img = img.astype(np.float32) * alpha + beta
        return np.clip(img, 0, 255).astype(np.uint8)

    def __call__(self, image, mask):
        """Apply brightness/contrast/saturation/hue/channel-swap distortions."""
        img = _pil_to_cv2_rgb(image)

        if random.randint(0, 1):
            img = self._convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))

        mode = random.randint(0, 1)
        if mode == 1 and random.randint(0, 1):
            img = self._convert(img, alpha=random.uniform(*self.contrast_range))

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        if random.randint(0, 1):
            img_hsv[:, :, 1] *= random.uniform(*self.saturation_range)

        if random.randint(0, 1):
            img_hsv[:, :, 0] += random.uniform(-self.hue_delta_cv2, self.hue_delta_cv2)
            img_hsv[:, :, 0] = img_hsv[:, :, 0] % 180.0

        img_hsv = np.clip(img_hsv, [0, 0, 0], [180, 255, 255]).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if mode == 0 and random.randint(0, 1):
            img = self._convert(img, alpha=random.uniform(*self.contrast_range))

        if random.randint(0, 1):
            img = img[:, :, np.random.permutation(3)]

        return _cv2_rgb_to_pil(img), mask


class PadToDivisor:
    """mmseg ``Pad(size_divisor=...)`` — pads image (0) and mask (ignore)."""

    def __init__(self, divisor: int = 16, img_pad_val: int = 0, seg_pad_val: int = 255):
        """Init pad divisor and pad values."""
        self.divisor = divisor
        self.img_pad_val = img_pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, image, mask):
        """Pad bottom/right up to a multiple of ``divisor``."""
        w, h = image.size
        new_w = math.ceil(w / self.divisor) * self.divisor
        new_h = math.ceil(h / self.divisor) * self.divisor

        if new_w != w or new_h != h:
            padded_img = Image.new("RGB", (new_w, new_h), (self.img_pad_val,) * 3)
            padded_img.paste(image, (0, 0))
            image = padded_img

            if mask is not None:
                mask_arr = np.array(mask)
                padded_mask = np.full((new_h, new_w), self.seg_pad_val, dtype=mask_arr.dtype)
                padded_mask[:h, :w] = mask_arr
                mask = Image.fromarray(padded_mask)

        return image, mask


class Normalize:
    """PIL image → float32 CHW tensor in ``[0, 1]``; mask → int64 tensor.

    By default applies NO mean/std (RADIO normalizes internally). Pass ``mean``
    and ``std`` (per-channel, in ``[0,1]`` scale) for backbones whose adapter
    does not normalize — e.g. NV-DINOv2 with ImageNet stats.
    """

    def __init__(self, mean=None, std=None):
        """Optionally store per-channel mean/std for in-transform normalization."""
        self.mean = torch.tensor(mean).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(3, 1, 1) if std is not None else None

    def __call__(self, image, mask):
        """Convert to tensors, optionally normalizing the image."""
        img_t = torch.from_numpy(
            np.array(image.convert("RGB"), dtype=np.float32)
        ).permute(2, 0, 1) / 255.0
        if self.mean is not None and self.std is not None:
            img_t = (img_t - self.mean) / self.std

        if mask is not None:
            mask_t = torch.from_numpy(np.array(mask)).long()
            return img_t, mask_t
        return img_t, None


def get_train_transforms(crop_size=(512, 512), pad_divisor: int = 16,
                         scale=(2048, 512), mean=None, std=None):
    """Full mmseg-parity train augmentation pipeline.

    ``crop_size`` defaults to 512×512 (HF/patch-16-multiple). The paper's 518×518
    is valid via the torchhub/mmseg path. Pass ``mean``/``std`` to normalize.
    """
    return Compose([
        RandomResizeKeepRatio(scale=scale, ratio_range=(0.5, 2.0)),
        RandomCropWithCatMaxRatio(crop_size=crop_size, cat_max_ratio=0.75, ignore_index=255),
        RandomFlip(prob=0.5),
        PhotoMetricDistortion(),
        PadToDivisor(divisor=pad_divisor),
        Normalize(mean=mean, std=std),
    ])


def get_val_img_transform(pad_divisor: int = 16, scale=(2048, 512), mean=None, std=None):
    """Val image pipeline: resize (keep ratio) + pad-to-divisor + to-tensor."""
    return Compose([
        ResizeKeepRatio(scale=scale),
        PadToDivisor(divisor=pad_divisor),
        Normalize(mean=mean, std=std),
    ])


def get_val_img_transform_tiling(mean=None, std=None):
    """Val pipeline for tiled inference — no resize, just to-tensor (tiling splits later)."""
    return Compose([
        Normalize(mean=mean, std=std),
    ])
