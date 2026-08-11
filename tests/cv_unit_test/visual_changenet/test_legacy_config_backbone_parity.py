# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity guard for the legacy visual_changenet config duplicate (bug 6465432).

nvidia_tao_pytorch/cv/visual_changenet/config/default_config.py is a legacy
duplicate that the TAO-1950 rewire orphaned (the training scripts import the
canonical nvidia_tao_pytorch/config/visual_changenet/default_config instead).
It is kept in sync for consistency; this asserts its backbone type options do
not drift from the canonical config again - which is how it fell behind the
DINOv3 backbones in the 7.1.0 report.
"""
from dataclasses import fields

from nvidia_tao_pytorch.config.visual_changenet.default_config import (
    BackboneConfig as CanonicalBackboneConfig,
)
from nvidia_tao_pytorch.cv.visual_changenet.config.default_config import (
    BackboneConfig as LegacyBackboneConfig,
)

DINOV3_DOWNSTREAM_BACKBONES = [
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
    "vit_7b_dinov3",
]


def _type_options(backbone_config_cls):
    """Return the backbone ``type`` field's raw valid_options string."""
    (type_field,) = [f for f in fields(backbone_config_cls) if f.name == "type"]
    return type_field.metadata["valid_options"]


def test_legacy_backbone_options_match_canonical():
    assert _type_options(LegacyBackboneConfig) == _type_options(CanonicalBackboneConfig)


def test_legacy_backbone_has_dinov3():
    options = _type_options(LegacyBackboneConfig).split(",")
    missing = [b for b in DINOV3_DOWNSTREAM_BACKBONES if b not in options]
    assert not missing, f"legacy config backbone enum missing DINOv3 options: {missing}"
