# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature caching for offline evaluation — extract once, skip recompute.

Ported from vfm-eval/c-radiov4 ``eval_knn._build_database`` and
``cache_features.py`` (Apache-2.0). The offline KNN / retrieval paths extract a
full split's L2-normalized summary embeddings once and persist them, so repeated
evaluations (sweeping ``k``, comparing checkpoints) skip the expensive backbone
forward pass. ImageNet embeddings run multiple GB — point ``cache_dir`` at a
scratch mount, not the home directory.

Two cache granularities:

- **Split-level embedding DB** (KNN/retrieval): one ``.pt`` per split holding
  ``{embeddings[N,D], labels[N]}`` — :func:`load_embedding_cache` /
  :func:`save_embedding_cache`.
- **Per-image dense features** (segmentation linear probe): one ``.pt`` per image
  holding ``{features[D,h,w], mask[H,W], orig_size}`` — :func:`save_dense_cache`.
  Used by the segmentation evaluator (Epic B3) to train the head at full speed.

Caching is rank-aware: on a shared filesystem all ranks read a cache that exists;
only rank 0 writes. Callers pass ``is_main`` rather than importing distributed
state here, keeping this module dependency-light.
"""

import logging
import os
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def embedding_cache_path(
    cache_dir: str, model_tag: str, resolution: int, split: str, suffix: str = "",
) -> str:
    """Build the canonical cache filename for a split's embedding DB.

    Mirrors c-radiov4's ``{model_tag}_{res}px_{split}{suffix}.pt`` naming so a
    cache extracted by either front-end is interchangeable.
    """
    safe_tag = model_tag.replace("/", "_")
    return os.path.join(cache_dir, f"{safe_tag}_{resolution}px_{split}{suffix}.pt")


def load_embedding_cache(path: Optional[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load ``(embeddings, labels)`` from ``path`` if it exists, else ``None``."""
    if not path or not os.path.exists(path):
        return None
    logger.info("[cache] loading features: %s", path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["embeddings"], data["labels"]


def save_embedding_cache(
    path: Optional[str], embeddings: torch.Tensor, labels: torch.Tensor, is_main: bool = True,
) -> None:
    """Save ``(embeddings, labels)`` to ``path`` (only on the main rank)."""
    if not path or not is_main:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, path)
    logger.info("[cache] saved features to %s", path)


def cached_embeddings(
    path: Optional[str],
    compute_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    is_main: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached ``(embeddings, labels)`` if present, else compute and cache.

    Args:
        path: cache file path, or ``None`` to disable caching.
        compute_fn: zero-arg callable returning ``(embeddings, labels)``. It is
            responsible for any distributed all-gather before returning.
        is_main: whether this rank should write the cache.
    """
    cached = load_embedding_cache(path)
    if cached is not None:
        return cached
    embeddings, labels = compute_fn()
    save_embedding_cache(path, embeddings, labels, is_main=is_main)
    return embeddings, labels


def save_dense_cache(
    cache_dir: str, index: int, features: torch.Tensor, mask: torch.Tensor,
    orig_size: Tuple[int, int],
) -> None:
    """Persist one image's dense features + mask for the segmentation probe.

    Stores ``features`` as ``bfloat16`` (halves disk for the large spatial maps),
    mirroring c-radiov4 ``cache_features.py``. ``features`` is ``[D, h, w]``,
    ``mask`` is ``[H, W]`` (int64, with the dataset's label mapping already applied).
    """
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(
        {
            "features": features.to(torch.bfloat16),
            "mask": mask,
            "orig_size": (int(orig_size[0]), int(orig_size[1])),
        },
        os.path.join(cache_dir, f"{index:06d}.pt"),
    )
