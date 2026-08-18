# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding I/O, provenance, caching, and text->video search helpers.

Kept dependency-light (numpy + h5py only) so it can be imported from both the
inference script and the Lightning model without pulling in heavy deps.

HDF5 layout (unchanged from the legacy inference output, plus provenance attrs):
- dataset ``embeddings`` : float32 (N, D)
- dataset ``video_ids`` / ``texts`` / ``image_paths`` : vlen-str source ids
- attrs: ``embedding_type``, ``embedding_dim``, ``num_*`` count, and provenance
  (``model_type``, ``checkpoint``, ``checkpoint_sha``, ``text_encoder``,
  ``normalized``) so cached embeddings are only reused when compatible.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


# embedding_type -> (default filename, items dataset key, count attr key)
_TYPE_TO_KEYS = {
    "image": ("image_embeddings.h5", "image_paths", "num_images"),
    "video": ("video_embeddings.h5", "video_ids", "num_videos"),
    "text": ("text_embeddings.h5", "texts", "num_texts"),
}

# Provenance attrs that must match for a cached embedding file to be reused.
_PROVENANCE_KEYS = ("model_type", "checkpoint_sha", "text_encoder")


def default_embeddings_filename(embedding_type: str) -> str:
    """Return the conventional file name for an embedding type."""
    return _TYPE_TO_KEYS[embedding_type][0]


def resolve_embeddings_path(
    explicit_file: Optional[str], results_dir: str, embedding_type: str
) -> str:
    """Resolve where an embedding file lives.

    Uses the explicit path if given, else ``<results_dir>/<default name>``.
    """
    if explicit_file:
        return explicit_file
    return os.path.join(results_dir, default_embeddings_filename(embedding_type))


def checkpoint_fingerprint(checkpoint_path: Optional[str]) -> str:
    """Cheap, stable fingerprint of a checkpoint (path + size + mtime).

    Avoids hashing multi-GB weight files while still changing when the
    checkpoint changes.
    """
    if not checkpoint_path:
        return ""
    try:
        st = os.stat(checkpoint_path)
        raw = f"{checkpoint_path}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        raw = str(checkpoint_path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_provenance(experiment_config, normalized: bool = False) -> Dict[str, str]:
    """Build provenance attrs from the experiment config."""
    model_cfg = getattr(experiment_config, "model", None)
    ckpt = getattr(getattr(experiment_config, "inference", None), "checkpoint", None) or ""
    return {
        "model_type": str(getattr(model_cfg, "type", "") or ""),
        "checkpoint": str(ckpt),
        "checkpoint_sha": checkpoint_fingerprint(ckpt),
        "text_encoder": str(getattr(model_cfg, "text_encoder", "") or ""),
        "normalized": bool(normalized),
    }


def provenance_compatible(
    attrs: Dict, provenance: Dict, keys: Tuple[str, ...] = _PROVENANCE_KEYS
) -> Tuple[bool, Optional[str]]:
    """Check whether a cached file's attrs are compatible with this run.

    Returns ``(ok, mismatched_key)``. Missing attrs (older files) are treated
    as compatible to stay backward compatible.
    """
    for k in keys:
        if k in provenance and k in attrs:
            if str(attrs[k]) != str(provenance[k]):
                return False, k
    return True, None


def write_embeddings_h5(
    path: str,
    ids: List[str],
    embeddings: np.ndarray,
    embedding_type: str,
    provenance: Optional[Dict] = None,
) -> None:
    """Write embeddings + ids + provenance to an HDF5 file."""
    if embedding_type not in _TYPE_TO_KEYS:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    _, items_key, count_key = _TYPE_TO_KEYS[embedding_type]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "embeddings", data=embeddings,
            compression="gzip", compression_opts=4,
        )
        dt = h5py.special_dtype(vlen=str)
        items_dataset = f.create_dataset(items_key, (len(ids),), dtype=dt)
        for i, item in enumerate(ids):
            items_dataset[i] = str(item)
        f.attrs[count_key] = len(ids)
        f.attrs["embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
        f.attrs["embedding_type"] = embedding_type
        for k, v in (provenance or {}).items():
            f.attrs[k] = v


def read_embeddings_h5(path: str) -> Tuple[List[str], np.ndarray, Dict]:
    """Read embeddings, ids, and attrs from an HDF5 file."""
    with h5py.File(path, "r") as f:
        emb = np.asarray(f["embeddings"][:], dtype=np.float32)
        etype = f.attrs.get("embedding_type")
        items_key = _TYPE_TO_KEYS.get(etype, (None, None, None))[1] if etype else None
        if not items_key or items_key not in f:
            # Fall back to whichever vlen-str dataset exists.
            for _, key, _ in _TYPE_TO_KEYS.values():
                if key in f:
                    items_key = key
                    break
        if items_key and items_key in f:
            ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f[items_key][:]
            ]
        else:
            ids = [str(i) for i in range(len(emb))]
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    return ids, emb, attrs


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def text_to_video_search(
    video_ids: List[str],
    video_emb: np.ndarray,
    texts: List[str],
    text_emb: np.ndarray,
    metric: str = "cosine",
    normalize: bool = True,
    top_k: int = 10,
) -> Tuple[List[Dict], np.ndarray]:
    """Rank, for each text query, the most similar video clips.

    ``metric='cosine'`` ranks by descending cosine similarity; ``metric='knn'``
    ranks by ascending Euclidean (L2) distance. Returns (results, score_matrix)
    where score_matrix is (num_texts, num_videos).
    """
    V = np.asarray(video_emb, dtype=np.float32)
    T = np.asarray(text_emb, dtype=np.float32)
    if normalize:
        V = _l2_normalize(V)
        T = _l2_normalize(T)
    k = max(1, min(int(top_k), len(video_ids)))

    if metric == "knn":
        # squared L2: |t|^2 + |v|^2 - 2 t.v
        vv = (V * V).sum(axis=1)[None, :]
        tt = (T * T).sum(axis=1)[:, None]
        d2 = tt + vv - 2.0 * (T @ V.T)
        np.clip(d2, 0.0, None, out=d2)
        scores = np.sqrt(d2)
        order = np.argsort(scores, axis=1)[:, :k]  # ascending distance
        score_name = "distance"
    elif metric == "cosine":
        scores = T @ V.T
        order = np.argsort(-scores, axis=1)[:, :k]  # descending similarity
        score_name = "score"
    else:
        raise ValueError(f"Unknown search_metric: {metric}. Use 'cosine' or 'knn'.")

    results = []
    for j, query in enumerate(texts):
        ranked = [
            {
                "rank": r + 1,
                "video_id": video_ids[idx],
                score_name: float(scores[j, idx]),
            }
            for r, idx in enumerate(order[j])
        ]
        results.append({"query": query, "metric": metric, "results": ranked})
    return results, scores


def write_search_results(results: List[Dict], path: str) -> None:
    """Write per-query search results as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")


def _tensor_stats(name: str, values: np.ndarray) -> Dict:
    """Summary stats for an embedding/score array."""
    finite = bool(np.isfinite(values).all())
    out = {
        "name": name,
        "shape": list(values.shape),
        "finite": finite,
    }
    if values.size:
        out.update({
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        })
        if values.ndim == 2:
            out["norm_mean"] = float(np.linalg.norm(values, axis=1).mean())
    return out


def write_similarity_stats(
    path: str,
    scores: np.ndarray,
    video_emb: np.ndarray,
    text_emb: np.ndarray,
    metric: str,
) -> None:
    """Write a sanity summary of the score matrix and the two embedding sets."""
    pcts = {}
    if scores.size:
        for p in (1, 25, 50, 75, 99):
            pcts[f"p{p}"] = float(np.percentile(scores, p))
    payload = {
        "metric": metric,
        "score_matrix": _tensor_stats("score_matrix", scores),
        "score_percentiles": pcts,
        "video_embeddings": _tensor_stats("video_embeddings", np.asarray(video_emb)),
        "text_embeddings": _tensor_stats("text_embeddings", np.asarray(text_emb)),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
