# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-specific leaf actions for SSL data refinement."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .grit import GRIT_FORMULA_VERSION, score_grit_frame  # noqa: F401
    from .grit_pipeline import (  # noqa: F401
        derive_grit_observations,
        relative_layer_numbers,
    )

_LAZY_EXPORTS = {
    "GRIT_FORMULA_VERSION": (".grit", "GRIT_FORMULA_VERSION"),
    "derive_grit_observations": (".grit_pipeline", "derive_grit_observations"),
    "relative_layer_numbers": (".grit_pipeline", "relative_layer_numbers"),
    "score_grit_frame": (".grit", "score_grit_frame"),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load optional GRIT dependencies only when their APIs are requested."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
