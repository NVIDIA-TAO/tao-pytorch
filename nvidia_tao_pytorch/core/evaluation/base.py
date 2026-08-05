# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator protocol + registry for shared embedding-quality evaluation.

An ``Evaluator`` is one self-contained metric (KNN, retrieval, segmentation
linear probe, ...). It runs in two modes off the *same* vote/scoring math:

- **Offline** (SSL ``evaluate`` action, vfm-eval harness): a one-shot driver
  builds an :class:`EvalContext`, calls :meth:`Evaluator.run` on each enabled
  evaluator, and merges the returned ``{metric: value}`` dicts into a single
  ``results.json``. KNN uses the FAISS path; features are cached to disk.
- **Online** (RADIO distillation validation): the distiller's Lightning hooks
  (``on_validation_epoch_start`` / ``validation_step`` / ``on_validation_epoch_end``)
  call the three online hooks below, so in-training KNN reuses the exact same
  weighted-vote math without a FAISS dependency in the train loop.

  ====================================  ===================================
  Lightning hook                        Evaluator online hook
  ====================================  ===================================
  ``on_validation_epoch_start``         :meth:`build_index`
  ``validation_step``                   :meth:`score_batch`
  ``on_validation_epoch_end``           :meth:`aggregate`
  ====================================  ===================================

Adding a new metric = drop in an ``@register_evaluator`` class — no orchestrator
changes. Two axes distinguish evaluators:

- ``requires_fit``: training-free (KNN, retrieval) vs train-a-head (seg probe).
- ``feature_level``: ``"global"`` (pooled CLS/summary embedding) vs ``"dense"``
  (patch feature map ``[B, C, h, w]`` for dense tasks).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import torch


@dataclass
class EvalContext:
    """Shared services handed to every evaluator.

    Attributes:
        model: The wrapped backbone — a :class:`ModelAdapter` exposing the
            ``(summary, features)`` contract (eval mode, on ``device``). The
            offline driver builds this via ``build_adapter``; the online RADIO
            path wraps the live student.
        network: Network name, e.g. ``"nvdinov2"`` / ``"mae"`` / ``"radio"`` —
            selects the per-network adapter and any protocol defaults.
        device: Compute device.
        distributed: Whether ``torch.distributed`` is initialized.
        build_loader: ``(dataset_type, root, **kwargs) -> DataLoader`` factory
            for the labeled (folder/WDS) eval datasets. Batches follow the
            uniform ``{"image", "label", "path"}`` contract.
        cfg: The ``evaluate`` config block; each evaluator reads its own
            sub-block (``cfg.knn``, ``cfg.retrieval``, ``cfg.segmentation``).
        results_dir: Output directory for ``results.json``, seg-head
            checkpoints, and the feature cache.
        cache_dir: Optional directory for cached feature tensors (offline). When
            ``None``, features are recomputed each run.
    """

    model: torch.nn.Module
    network: str
    device: torch.device
    distributed: bool
    build_loader: Callable[..., Any]
    cfg: Any
    results_dir: Optional[str] = None
    cache_dir: Optional[str] = None


class Evaluator(ABC):
    """Base class for a single embedding-evaluation metric (offline + online)."""

    #: registry key; also the config sub-block name (cfg.<name>.enabled).
    name: str = ""
    #: whether ``run`` trains a head before scoring.
    requires_fit: bool = False
    #: ``"global"`` pooled embedding or ``"dense"`` patch feature map.
    feature_level: str = "global"
    #: whether this evaluator implements the online hooks (in-training validation).
    supports_online: bool = False

    def enabled(self, cfg: Any) -> bool:
        """True when this evaluator's config sub-block is enabled."""
        block = getattr(cfg, self.name, None)
        return bool(getattr(block, "enabled", False))

    # ------------------------------------------------------------------ offline
    @abstractmethod
    def run(self, ctx: EvalContext) -> Dict[str, float]:
        """Run the full metric (offline) and return a flat ``{metric: value}`` dict."""
        raise NotImplementedError

    # ------------------------------------------------------------------- online
    # The online hooks are optional; only evaluators with ``supports_online =
    # True`` (e.g. KNN) implement them. They share the offline scoring math but
    # are driven by Lightning validation hooks instead of a one-shot run().
    def build_index(self, ctx: EvalContext) -> None:
        """Online: build the reference set (e.g. extract+gather train embeddings).

        Called once at ``on_validation_epoch_start``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support online evaluation "
            f"(supports_online={self.supports_online})."
        )

    def score_batch(self, ctx: EvalContext, batch: Any) -> None:
        """Online: score one validation batch against the reference set.

        Called per ``validation_step``; accumulates running stats.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support online evaluation."
        )

    def aggregate(self, ctx: EvalContext) -> Dict[str, float]:
        """Online: reduce accumulated stats across ranks into ``{metric: value}``.

        Called at ``on_validation_epoch_end``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support online evaluation."
        )


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
EVALUATOR_REGISTRY: Dict[str, Type[Evaluator]] = {}


def register_evaluator(cls: Type[Evaluator]) -> Type[Evaluator]:
    """Class decorator: register an Evaluator under its ``name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'.")
    EVALUATOR_REGISTRY[cls.name] = cls
    return cls


def build_enabled_evaluators(cfg: Any) -> List[Evaluator]:
    """Instantiate every registered evaluator whose config block is enabled."""
    return [cls() for _, cls in EVALUATOR_REGISTRY.items() if cls().enabled(cfg)]
