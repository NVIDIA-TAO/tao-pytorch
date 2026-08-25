# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional SAM3 backbone dependency."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cv_unit
def test_missing_sam3_is_silent_until_requested():
    """Importing the SAM3 module stays silent when its package is unavailable."""
    script = """
import importlib.abc
import importlib.util
import sys
import types

import torch


class Registry:
    def register(self):
        return lambda factory: factory


class BackboneBase(torch.nn.Module):
    pass


class Logger:
    def __init__(self):
        self.debug_messages = []

    def debug(self, message, *args):
        self.debug_messages.append(message % args)


class BlockSam3Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sam3" or fullname.startswith("sam3."):
            raise ModuleNotFoundError("No module named 'sam3'")
        return None


sys.meta_path.insert(0, BlockSam3Finder())

for package_name in (
    "nvidia_tao_pytorch",
    "nvidia_tao_pytorch.core",
    "nvidia_tao_pytorch.cv",
):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package

tlt_logging = types.ModuleType("nvidia_tao_pytorch.core.tlt_logging")
tlt_logging.logger = Logger()
sys.modules[tlt_logging.__name__] = tlt_logging

backbone_package_name = "nvidia_tao_pytorch.cv.backbone_v2"
backbone_package = types.ModuleType(backbone_package_name)
backbone_package.__path__ = []
backbone_package.BACKBONE_REGISTRY = Registry()
sys.modules[backbone_package_name] = backbone_package

backbone_base = types.ModuleType(f"{backbone_package_name}.backbone_base")
backbone_base.BackboneBase = BackboneBase
sys.modules[backbone_base.__name__] = backbone_base

module_name = f"{backbone_package_name}.sam3"
spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
sam3 = importlib.util.module_from_spec(spec)
sys.modules[module_name] = sam3
spec.loader.exec_module(sam3)

assert sam3.build_sam3_image_model is None
assert tlt_logging.logger.debug_messages == [
    "SAM3 is unavailable: No module named 'sam3'"
]
try:
    sam3.get_sam3_model()
except ImportError as error:
    assert "pip install git+https://github.com/facebookresearch/sam3.git" in str(error)
else:
    raise AssertionError("SAM3 use should fail when its optional package is absent")
"""
    sam3_path = Path(__file__).resolve().parents[3].joinpath(
        "nvidia_tao_pytorch",
        "cv",
        "backbone_v2",
        "sam3.py",
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(sam3_path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "Failed to import SAM3" not in result.stdout
    assert "Failed to import SAM3" not in result.stderr
