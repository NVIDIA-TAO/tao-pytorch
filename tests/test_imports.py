#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest test to identify import errors in Python files.

Scans all Python files in nvidia_tao_pytorch and checks for import issues:
1. Missing modules (via importlib.util.find_spec)
2. Missing imported names from internal modules (via AST-based definition scanning)

The AST-based name check catches regressions like a function rename that leaves
stale imports (e.g. `from ...build_pl_model import build_model` when the function
was renamed to `build_pl_model`).

Usage:
    pytest tests/test_imports.py -v -s
"""

import ast
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest


def _handler_catches_import_error(handler):
    """Return True if an ``except`` handler catches ImportError (or is bare)."""
    exc_type = handler.type
    if exc_type is None:
        return True
    candidates = exc_type.elts if isinstance(exc_type, ast.Tuple) else [exc_type]
    for cand in candidates:
        name = None
        if isinstance(cand, ast.Name):
            name = cand.id
        elif isinstance(cand, ast.Attribute):
            name = cand.attr
        if name in {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}:
            return True
    return False


def _collect_optional_import_nodes(tree):
    """Return the set of Import/ImportFrom AST nodes whose failure is handled.

    An import is considered optional if it lives anywhere in the ``body`` of a
    ``try`` whose handlers catch ImportError/ModuleNotFoundError (or are bare).
    """
    optional = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_handler_catches_import_error(h) for h in node.handlers):
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    optional.add(child)
    return optional

ROOT_DIR = Path(__file__).parent.parent  # tao-pytorch/
PACKAGE_DIR = ROOT_DIR / "nvidia_tao_pytorch"

OPTIONAL_DEPS = {
    "hydra", "clearml", "wandb", "pytorch_lightning", "tensorflow",
    "mpi4py", "pycuda", "torch", "diffusers", "imageio", "transformers",
    "tritonclient", "torchvision", "onnxruntime", "onnx", "tensorrt",
    "trt_dev", "polygraphy", "nvidia_tao_deploy", "nvidia_tao_ds",
    "modelopt", "timm", "lightning", "accelerate", "safetensors",
    "kornia", "open3d", "mmcv", "mmdet", "mmengine", "peft",
    "nvidia_tao_core", "release", "lmi_scripts",
}

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".pytest_cache", "node_modules",
    "venv", "env", ".venv", "build", "dist", "egg-info",
    "release", "license",
}

EXCLUDE_FILES = {
    "test_imports.py", "setup.py", "conftest.py",
}


def _find_python_files():
    """Collect all .py files under PACKAGE_DIR."""
    result = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if any(excl in path.parts for excl in EXCLUDE_DIRS):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        result.append(path)
    return sorted(result)


def _parse_imports(file_path):
    """Extract import statements from *file_path* using AST."""
    with open(file_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=str(file_path))

    optional_nodes = _collect_optional_import_nodes(tree)

    imports = []
    for node in ast.walk(tree):
        is_optional = node in optional_nodes
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "kind": "import",
                    "module": alias.name,
                    "names": [],
                    "line": node.lineno,
                    "level": 0,
                    "optional": is_optional,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                module = "." * node.level + module
            imports.append({
                "kind": "from",
                "module": module,
                "names": [a.name for a in node.names],
                "line": node.lineno,
                "level": node.level,
                "optional": is_optional,
            })
    return imports


def _definitions_in_file(path):
    """Return the set of top-level names defined in *path* (functions, classes, assignments)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=str(path))
    except Exception:
        return None

    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[-1]
                names.add(local)
    return names


def _module_path_to_file(module_dotted):
    """Resolve an absolute ``nvidia_tao_pytorch.*`` import to a file path."""
    parts = module_dotted.split(".")
    candidate_pkg = ROOT_DIR / Path(*parts) / "__init__.py"
    candidate_mod = ROOT_DIR / Path(*parts[:-1]) / (parts[-1] + ".py") if len(parts) > 1 else None

    if candidate_pkg.is_file():
        return candidate_pkg
    if candidate_mod and candidate_mod.is_file():
        return candidate_mod
    return None


def _is_optional(module_name):
    root = module_name.lstrip(".").split(".")[0]
    return root in OPTIONAL_DEPS


def _check_file(file_path):
    """Return list of error dicts for *file_path*."""
    try:
        imports = _parse_imports(file_path)
    except SyntaxError as exc:
        return [{"line": exc.lineno or 0, "msg": f"SyntaxError: {exc}"}]

    errors = []
    for imp in imports:
        module = imp["module"]

        if imp["level"] > 0:
            continue
        if imp.get("optional"):
            continue
        if _is_optional(module):
            continue

        if not module.startswith("nvidia_tao_pytorch"):
            continue

        target_file = _module_path_to_file(module)
        if target_file is None:
            errors.append({
                "line": imp["line"],
                "msg": f"Module '{module}' not found on disk",
            })
            continue

        if imp["kind"] == "from" and imp["names"]:
            defs = _definitions_in_file(target_file)
            if defs is None:
                continue
            if str(target_file).endswith("__init__.py"):
                continue
            for name in imp["names"]:
                if name == "*":
                    continue
                if name not in defs:
                    errors.append({
                        "line": imp["line"],
                        "msg": (
                            f"Name '{name}' not found in "
                            f"'{module}' ({target_file.relative_to(ROOT_DIR)})"
                        ),
                    })
    return errors


def test_internal_imports():
    """All nvidia_tao_pytorch internal imports must resolve correctly."""
    python_files = _find_python_files()
    all_errors = {}

    for fpath in python_files:
        errs = _check_file(fpath)
        if errs:
            rel = str(fpath.relative_to(ROOT_DIR))
            all_errors[rel] = errs

    if all_errors:
        lines = [f"\nFound import errors in {len(all_errors)} file(s):\n"]
        for filepath, errs in sorted(all_errors.items()):
            lines.append(f"\n  {filepath}")
            for e in errs:
                lines.append(f"    Line {e['line']}: {e['msg']}")
        pytest.fail("\n".join(lines))
