# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SegFormer augmentation probability wiring."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torchvision")
from nvidia_tao_pytorch.cv.segformer.dataloader import augmentation


class FakeColorJitter:
    """Record applications while preserving the input image."""

    applications = 0

    def __init__(self, brightness, contrast, saturation, hue):
        if saturation is None or hue is None:
            raise TypeError("disabled ColorJitter components cannot be reconstructed from None")
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = None if saturation == 0 else saturation
        self.hue = None if hue == 0 else hue

    def __call__(self, image):
        type(self).applications += 1
        return image


def make_augmentor(probability):
    color = SimpleNamespace(
        enable=True,
        color_probability=probability,
        brightness=0.3,
        contrast=0.3,
        saturation=0.0,
        hue=0.0,
    )
    return augmentation.CDDataAugmentation(img_size=8, random_color=color)


def test_color_probability_zero_skips_jitter(monkeypatch):
    FakeColorJitter.applications = 0
    monkeypatch.setattr(augmentation.transforms, "ColorJitter", FakeColorJitter)
    make_augmentor(0.0).transform([np.zeros((8, 8, 3), dtype=np.uint8)], [], to_tensor=False)
    assert FakeColorJitter.applications == 0


def test_color_probability_one_applies_jitter(monkeypatch):
    FakeColorJitter.applications = 0
    monkeypatch.setattr(augmentation.transforms, "ColorJitter", FakeColorJitter)
    make_augmentor(1.0).transform([np.zeros((8, 8, 3), dtype=np.uint8)], [], to_tensor=False)
    assert FakeColorJitter.applications == 1


def test_zero_saturation_and_hue_are_applied_without_reconstruction(monkeypatch):
    FakeColorJitter.applications = 0
    monkeypatch.setattr(augmentation.transforms, "ColorJitter", FakeColorJitter)
    make_augmentor(1.0).transform([np.zeros((8, 8, 3), dtype=np.uint8)], [], to_tensor=False)
    assert FakeColorJitter.applications == 1
