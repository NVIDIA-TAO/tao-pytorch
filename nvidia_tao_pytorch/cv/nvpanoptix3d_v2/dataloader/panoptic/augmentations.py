# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry-safe photometric augmentation for ScanNet++ multi-view samples."""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.utils import (
    get_config_value,
)


class GeometrySafePhotometricAugmentation:
    """Moderate RGB-only augmentation shared by all views of one sample.

    No operation changes image dimensions or pixel coordinates. Parameters
    are sampled once per multi-view sample, then applied identically to every
    view so correspondence cues are not corrupted by view-specific jitter.
    """

    def __init__(self, config):
        self.brightness = float(get_config_value(config, "brightness", 0.10))
        self.contrast = float(get_config_value(config, "contrast", 0.10))
        self.saturation = float(get_config_value(config, "saturation", 0.15))
        self.gamma = float(get_config_value(config, "gamma", 0.10))
        self.exposure_ev = float(
            get_config_value(config, "exposure_ev", 0.15)
        )
        self.color_jitter_prob = float(
            get_config_value(config, "color_jitter_prob", 0.8)
        )
        self.gamma_exposure_prob = float(
            get_config_value(config, "gamma_exposure_prob", 0.5)
        )
        self.grayscale_prob = float(
            get_config_value(config, "grayscale_prob", 0.05)
        )

        for name in ("brightness", "contrast", "saturation", "exposure_ev"):
            if getattr(self, name) < 0:
                raise ValueError(f"photometric {name} must be non-negative")
        if not 0 <= self.gamma < 1:
            raise ValueError("photometric gamma must be in [0, 1)")
        for name in (
            "color_jitter_prob", "gamma_exposure_prob", "grayscale_prob",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"photometric {name} must be in [0, 1]")

    @classmethod
    def from_config(cls, config):
        """Build the augmentation from config, or None when it is disabled."""
        if config is None or not bool(get_config_value(config, "enabled", False)):
            return None
        return cls(config)

    def sample(self, rng):
        """Sample one immutable recipe for a complete multi-view sample."""
        color_ops = []
        if rng.random() < self.color_jitter_prob:
            color_ops = [
                ("brightness", rng.uniform(
                    1 - self.brightness, 1 + self.brightness,
                )),
                ("contrast", rng.uniform(
                    1 - self.contrast, 1 + self.contrast,
                )),
                ("saturation", rng.uniform(
                    1 - self.saturation, 1 + self.saturation,
                )),
            ]
            rng.shuffle(color_ops)

        gamma = 1.0
        exposure_ev = 0.0
        if rng.random() < self.gamma_exposure_prob:
            gamma = rng.uniform(1 - self.gamma, 1 + self.gamma)
            exposure_ev = rng.uniform(-self.exposure_ev, self.exposure_ev)

        return {
            "color_ops": tuple(color_ops),
            "gamma": gamma,
            "exposure_ev": exposure_ev,
            "grayscale": rng.random() < self.grayscale_prob,
        }

    @staticmethod
    def apply(img: Image.Image, recipe) -> Image.Image:
        """Apply an RGB-only recipe without changing spatial geometry."""
        result = img
        enhancers = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }
        for operation, factor in recipe["color_ops"]:
            result = enhancers[operation](result).enhance(factor)

        gamma = float(recipe["gamma"])
        exposure_ev = float(recipe["exposure_ev"])
        if gamma != 1.0 or exposure_ev != 0.0:
            pixels = np.asarray(result, dtype=np.float32) / 255.0
            pixels = np.power(np.clip(pixels, 0.0, 1.0), gamma)
            pixels *= 2.0 ** exposure_ev
            pixels = np.clip(np.round(pixels * 255.0), 0, 255).astype(np.uint8)
            result = Image.fromarray(pixels)

        if recipe["grayscale"]:
            result = ImageOps.grayscale(result).convert("RGB")
        return result
