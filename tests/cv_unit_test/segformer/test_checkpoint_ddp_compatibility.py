# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Guard SegFormer activation checkpoints against reentrant DDP failures."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "nvidia_tao_pytorch"


def checkpoint_modes(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text())
    modes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "checkpoint":
            modes.extend(
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "use_reentrant"
                and isinstance(keyword.value, ast.Constant)
            )
    return modes


def test_segformer_checkpoints_are_nonreentrant_for_ddp():
    paths = (
        "cv/segformer/model/backbones/fan.py",
        "cv/segformer/model/backbones/adapter_modules.py",
    )
    for path in paths:
        modes = checkpoint_modes(path)
        assert modes, f"expected activation-checkpoint calls in {path}"
        assert modes == [False] * len(modes)
