# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for default_specs stdout mode."""

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utils.default_specs import DefaultConfig


def test_results_dir_is_optional_in_structured_config():
    config = OmegaConf.structured(DefaultConfig)
    assert config.results_dir is None
