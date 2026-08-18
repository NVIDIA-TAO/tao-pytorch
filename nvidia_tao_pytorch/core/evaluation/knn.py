# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KNN Top-1 classification evaluator — offline (FAISS) + online (distributed_topk).

This is the **source of truth** for KNN evaluation, consumed by the SSL
``evaluate`` action (offline), RADIO distillation validation (online), and the
vfm-eval harness. It vendors two paper-faithful implementations of the *same*
protocol:

- **Offline** (``KNNEvaluator.run``): extract the full train/val splits' summary
  embeddings (cached to disk), then FAISS ``IndexFlatIP`` cosine search on the
  gathered embeddings — ported from vfm-eval/c-radiov4 ``eval_knn.py`` (Apache-2.0).
  When ``faiss`` is unavailable in the container the search falls back to a
  chunked brute-force matmul top-K (DESIGN open item #4).
- **Online** (``KNNEvaluator.build_index`` / ``score_batch`` / ``aggregate``):
  ``distributed_topk`` across ranks with no FAISS dependency in the train loop —
  ported from ``multimodal/radio/distillation/knn_classification.py``. RADIO's
  distiller validation hooks call these (Epic C makes that module a re-export shim).

Both share the weighted-majority vote ``exp(sim / T)`` and the protocol
constants — **k=20, T=0.07** — so numbers stay comparable to the paper
(C-RADIOv4-H ImageNet KNN Top-1 = 86.59%). Summary embeddings are the CLS /
pooled vector, L2-normalized; distance is cosine.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nvidia_tao_pytorch.core.distributed.comm import (
    get_global_rank,
    get_world_size,
    is_dist_avail_and_initialized,
)
from nvidia_tao_pytorch.core.evaluation.base import EvalContext, Evaluator, register_evaluator
from nvidia_tao_pytorch.core.evaluation.caching import cached_embeddings, embedding_cache_path

logger = logging.getLogger(__name__)

# Protocol constants — preserve exactly (paper-validated; do not parameterize away).
KNN_TEMPERATURE = 0.07   # from DINO / arXiv:1805.01978 §3.4; fixed in RADIO knn_classification.py
KNN_DEFAULT_K = 20

# Optional FAISS — present in some containers, absent in others (DESIGN open item #4).
try:  # pragma: no cover - import availability is environment-dependent
    import faiss
    import numpy as np
    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


# ---------------------------------------------------------------------------
# Shared vote math (identical across offline + online; T=0.07)
# ---------------------------------------------------------------------------
def _get_vote_cls(sim: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Weighted majority vote over neighbors: weight = ``exp(sim / T)``.

    Args:
        sim: ``[Q, K]`` neighbor cosine similarities.
        labels: ``[Q, K]`` neighbor class labels.
        num_classes: total number of classes.

    Returns:
        ``[Q]`` predicted class ids.
    """
    weights = torch.exp(sim / KNN_TEMPERATURE)
    cls_vec = torch.zeros(weights.shape[0], num_classes, dtype=weights.dtype, device=weights.device)
    cls_vec.scatter_add_(dim=1, index=labels, src=weights)
    return cls_vec.argmax(dim=1)


# ---------------------------------------------------------------------------
# Online path: distributed_topk (ported from radio knn_classification.py)
# ---------------------------------------------------------------------------
def _pad(tensor: torch.Tensor, dim0: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad ``tensor`` along dim-0 up to ``dim0`` and return a validity mask."""
    valid_mask = torch.ones(dim0, dtype=torch.bool, device=tensor.device)
    valid_mask[tensor.shape[0]:].fill_(False)
    if tensor.shape[0] == dim0:
        return tensor, valid_mask
    ret = torch.empty(dim0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    ret[:tensor.shape[0]].copy_(tensor)
    return ret, valid_mask


def _all_to_all(t: torch.Tensor) -> torch.Tensor:
    """All-to-all over the rank (dim-0) axis, restacking the results."""
    input_tensors = list(t)
    output_tensors = [torch.empty_like(v) for v in input_tensors]
    dist.all_to_all(output_tensors, input_tensors)
    return torch.stack(output_tensors)


def distributed_topk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    distributed: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find top-K nearest neighbors across all ranks.

    Args:
        queries: ``[Q, C]`` query embeddings (L2-normalized).
        keys: ``[D, C]`` key embeddings on this rank (L2-normalized).
        labels: ``[D]`` class labels for ``keys``.
        K: number of nearest neighbors.
        distributed: whether to use distributed communication.

    Returns:
        ``(max_sim, max_labels)`` each of shape ``[Q, K]`` for this rank's queries.
    """
    if distributed:
        world_size = dist.get_world_size()
        max_queries = torch.tensor(queries.shape[0], dtype=torch.int64, device=queries.device)
        dist.all_reduce(max_queries, dist.ReduceOp.MAX)
        max_queries = max_queries.item()

        queries, valid_mask = _pad(queries, max_queries)
        all_queries = torch.empty(
            world_size, queries.shape[0], queries.shape[1],
            dtype=queries.dtype, device=queries.device,
        )
        dist.all_gather_into_tensor(all_queries, queries)
    else:
        all_queries = queries.unsqueeze(0)
        valid_mask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    # all_queries: [W, Q, C] · keys: [D, C] -> similarity [W, Q, D]
    similarity = torch.matmul(all_queries, keys.T)
    max_sim, max_idxs = torch.topk(similarity, k=K, dim=2, largest=True, sorted=False)
    max_labels = labels[max_idxs.flatten()].reshape_as(max_idxs)

    if distributed:
        # max_sim/max_labels are AllQueries -> KeysForThisRank; all-to-all
        # rearranges to QueriesForThisRank -> AllKeys.
        max_sim = _all_to_all(max_sim)
        max_labels = _all_to_all(max_labels)

    # [N, K*W]
    max_sim = max_sim.permute(1, 2, 0).flatten(1)
    max_labels = max_labels.permute(1, 2, 0).flatten(1)

    if distributed:
        max_sim, max_idxs = torch.topk(max_sim, k=K, dim=1, largest=True, sorted=False)
        max_labels = torch.gather(max_labels, dim=1, index=max_idxs)

    return max_sim[valid_mask], max_labels[valid_mask]


def knn_predict(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    K: int,
    num_classes: int,
    distributed: bool,
) -> torch.Tensor:
    """Predict class ids for ``query_embeddings`` via weighted KNN vote.

    Returns ``[Q]`` predicted class ids (per-rank queries under ``distributed``).
    """
    max_sim, max_labels = distributed_topk(
        queries=query_embeddings, keys=train_embeddings, labels=train_labels,
        K=K, distributed=distributed,
    )
    return _get_vote_cls(max_sim, max_labels, num_classes=num_classes)


def knn_top1_accuracy(
    train_split_embeddings: torch.Tensor,
    train_split_labels: torch.Tensor,
    K: int,
    output: torch.Tensor,
    target: torch.Tensor,
    distributed: bool,
    num_classes: int = 1000,
) -> torch.Tensor:
    """KNN Top-1 accuracy (%) for one query batch — back-compatible with radio.

    Mirrors the original ``radio.distillation.knn_classification.knn_top1_accuracy``
    signature so the distiller (Epic C) can re-export from here unchanged.
    """
    vote_id = knn_predict(
        train_split_embeddings, train_split_labels, output, K, num_classes, distributed)
    num_correct = (target == vote_id).sum()
    return 100.0 * num_correct / output.size(0)


# ---------------------------------------------------------------------------
# Offline path: FAISS IndexFlatIP, with a brute-force fallback (B7)
# ---------------------------------------------------------------------------
def _knn_top1_faiss(train_emb, train_lbl, val_emb, val_lbl, num_classes, K) -> float:
    """FAISS ``IndexFlatIP`` cosine KNN (GPU if available). Ported from eval_knn.py."""
    dim = train_emb.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(train_emb.cpu().numpy().astype(np.float32))
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    except Exception:  # pragma: no cover - GPU faiss optional
        index = cpu_index

    train_lbl_np = train_lbl.cpu().numpy()
    val_np = val_emb.cpu().numpy().astype(np.float32)
    val_lbl_np = val_lbl.cpu().numpy()

    chunk, num_correct = 2048, 0
    for start in range(0, len(val_np), chunk):
        end = min(start + chunk, len(val_np))
        D, idxs = index.search(val_np[start:end], K)
        vote_id = _get_vote_cls(
            torch.from_numpy(D), torch.from_numpy(train_lbl_np[idxs]), num_classes=num_classes)
        num_correct += (vote_id == torch.from_numpy(val_lbl_np[start:end])).sum().item()
    return 100.0 * num_correct / len(val_np)


def _knn_top1_bruteforce(train_emb, train_lbl, val_emb, val_lbl, num_classes, K) -> float:
    """Chunked brute-force matmul top-K — FAISS-free fallback (same vote math).

    Runs on GPU when available. Used when ``faiss`` is not installed in the
    container; numerically identical to the FAISS path (cosine on L2-normed vecs).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_emb = train_emb.to(device)
    train_lbl = train_lbl.to(device)
    chunk, num_correct = 2048, 0
    for start in range(0, val_emb.shape[0], chunk):
        q = val_emb[start:start + chunk].to(device)
        sim = q @ train_emb.T                                   # [b, N_train]
        top_sim, top_idx = torch.topk(sim, k=K, dim=1, largest=True, sorted=False)
        top_lbl = train_lbl[top_idx]
        vote_id = _get_vote_cls(top_sim, top_lbl, num_classes=num_classes)
        num_correct += (vote_id == val_lbl[start:start + chunk].to(device)).sum().item()
    return 100.0 * num_correct / val_emb.shape[0]


def knn_top1_offline(train_emb, train_lbl, val_emb, val_lbl, num_classes,
                     K=KNN_DEFAULT_K, use_faiss=True) -> float:
    """Offline KNN Top-1 (%): FAISS when available + requested, else brute-force fallback.

    ``use_faiss=False`` forces the brute-force matmul path even when faiss is
    installed — useful when faiss-gpu is unstable in the container (it can
    segfault during ``IndexFlatIP`` search). The brute-force path is numerically
    identical (same cosine + weighted vote).
    """
    if use_faiss and _HAS_FAISS:
        return _knn_top1_faiss(train_emb, train_lbl, val_emb, val_lbl, num_classes, K)
    if use_faiss and not _HAS_FAISS:
        logger.warning("[KNN] faiss not available — using brute-force top-K fallback.")
    return _knn_top1_bruteforce(train_emb, train_lbl, val_emb, val_lbl, num_classes, K)


# ---------------------------------------------------------------------------
# Distributed + batch helpers
# ---------------------------------------------------------------------------
def _is_main() -> bool:
    return get_global_rank() == 0


def _all_gather_cat(tensor: torch.Tensor, total_size: Optional[int] = None) -> torch.Tensor:
    """All-gather equal-size tensors from every rank and cat on dim-0 (CPU result).

    ``DistributedSampler(drop_last=False)`` pads each rank to an equal count, so
    sizes match across ranks; pass ``total_size`` to strip that padding.
    """
    if not is_dist_avail_and_initialized() or get_world_size() == 1:
        return tensor
    device = torch.device("cuda", torch.cuda.current_device())
    t = tensor.to(device).contiguous()
    gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(gathered, t)
    out = torch.cat(gathered, dim=0).cpu()
    return out[:total_size] if total_size is not None else out


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read images/labels from a batch, tolerating both key conventions.

    Core datasets emit ``{image, label}``; RADIO's WDS loaders emit ``{img, class}``.
    """
    if isinstance(batch, dict):
        images = batch["image"] if "image" in batch else batch["img"]
        labels = batch["label"] if "label" in batch else batch["class"]
        return images, labels
    return batch[0], batch[1]   # (images, labels) tuple


@torch.no_grad()
def extract_summary_database(loader, model, device, amp: bool, total_size=None):
    """Extract L2-normalized summary embeddings + labels for a whole split.

    Multi-GPU: each rank embeds its ``DistributedSampler`` shard, then results are
    all-gathered to every rank and padding stripped to ``total_size``.
    """
    embeddings, labels_list = [], []
    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            summary, _ = model(images)
        embeddings.append(F.normalize(summary.float(), p=2, dim=1).cpu())
        labels_list.append(labels.cpu())
    emb = torch.cat(embeddings, dim=0)
    lbl = torch.cat(labels_list, dim=0)
    emb = _all_gather_cat(emb, total_size)
    lbl = _all_gather_cat(lbl, total_size)
    return emb, lbl


# ---------------------------------------------------------------------------
# Evaluator (dual-mode)
# ---------------------------------------------------------------------------
@register_evaluator
class KNNEvaluator(Evaluator):
    """KNN Top-1 classification on the summary (CLS) embedding."""

    name = "knn"
    requires_fit = False
    feature_level = "global"
    supports_online = True

    # ---- config helpers ----
    def _cfg(self, ctx: EvalContext):
        return getattr(ctx.cfg, self.name)

    def _build_loader(self, ctx: EvalContext, root: str, split: str):
        """Build a classification loader via the ctx factory (falls back to core)."""
        from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
            IMAGENET_MEAN, IMAGENET_STD,
        )
        cfg = self._cfg(ctx)
        # Normalize in the dataset only when the adapter does not (imagenet_normalize);
        # explicit mean/std on the cfg, if present, take precedence.
        mean = getattr(cfg, "mean", None)
        std = getattr(cfg, "std", None)
        if mean is None and std is None and getattr(cfg, "imagenet_normalize", False):
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        kwargs = dict(
            batch_size=getattr(cfg, "batch_size", 128),
            num_workers=getattr(cfg, "num_workers", 8),
            crop=getattr(cfg, "crop", 224),
            resize=getattr(cfg, "resize", None),
            interpolation=getattr(cfg, "interpolation", "bicubic"),
            mean=mean,
            std=std,
            distributed=ctx.distributed,
            num_classes=getattr(cfg, "num_classes", 1000),
            label_key=getattr(cfg, "label_key", "cls"),
        )
        if split == "train" and getattr(cfg, "max_train_samples", None) is not None:
            kwargs["max_samples"] = cfg.max_train_samples
        builder = ctx.build_loader
        if builder is None:
            from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
                build_classification_loader,
            )
            builder = build_classification_loader
        return builder(getattr(cfg, "dataset_type", "image_folder"), root, **kwargs)

    # ---- offline ----
    def run(self, ctx: EvalContext):
        """Offline KNN: extract+cache train/val embeddings, FAISS search, Top-1."""
        cfg = self._cfg(ctx)
        K = getattr(cfg, "k", KNN_DEFAULT_K)
        amp = getattr(cfg, "amp", True)

        train = self._build_loader(ctx, cfg.train_root, "train")
        val = self._build_loader(ctx, cfg.val_root, "val")
        num_classes = train.num_classes

        def _extract(loader_info, split, suffix=""):
            path = None
            if ctx.cache_dir:
                tag = getattr(cfg, "cache_tag", ctx.network)
                res = getattr(cfg, "crop", 224)
                path = embedding_cache_path(ctx.cache_dir, tag, res, split, suffix)
            return cached_embeddings(
                path,
                lambda: extract_summary_database(
                    loader_info.loader, ctx.model, ctx.device, amp,
                    total_size=loader_info.total_size),
                is_main=_is_main(),
            )

        logger.info("[KNN] extracting train database...")
        suffix = f"_top{cfg.max_train_samples}" if getattr(cfg, "max_train_samples", None) else ""
        train_emb, train_lbl = _extract(train, "train", suffix)
        logger.info("[KNN] extracting val database...")
        val_emb, val_lbl = _extract(val, "val")

        # FAISS search + accuracy only on rank 0 (it holds the full gathered set).
        # Non-main ranks contribute no metrics, so they return an empty dict rather
        # than a placeholder accuracy that could be mistaken for a real score.
        use_faiss = getattr(cfg, "use_faiss", True)
        if _is_main():
            logger.info("[KNN] computing Top-1 (k=%d, T=%.2f, faiss=%s)...",
                        K, KNN_TEMPERATURE, use_faiss and _HAS_FAISS)
            acc = knn_top1_offline(train_emb, train_lbl, val_emb, val_lbl, num_classes, K,
                                   use_faiss=use_faiss)
            logger.info("[KNN] Top-1: %.2f%%", acc)
            return {"knn_top1": acc}
        return {}

    # ---- online (RADIO distillation validation hooks) ----
    def build_index(self, ctx: EvalContext):
        """Embed the training split on this rank to build the KNN reference set.

        Unlike the offline path, embeddings are kept per-rank on device;
        ``distributed_topk`` gathers the *queries* during scoring instead.
        """
        cfg = self._cfg(ctx)
        amp = getattr(cfg, "amp", True)
        train = self._build_loader(ctx, cfg.train_root, "train")
        embeddings, labels_list = [], []
        with torch.no_grad():
            for batch in train.loader:
                images, labels = _unpack_batch(batch)
                images = images.to(ctx.device, non_blocking=True)
                with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=amp):
                    summary, _ = ctx.model(images)
                embeddings.append(F.normalize(summary.float(), p=2, dim=1))
                labels_list.append(labels.to(ctx.device))
        self._train_emb = torch.cat(embeddings, dim=0)
        self._train_lbl = torch.cat(labels_list, dim=0)
        self._num_classes = train.num_classes
        self._correct = torch.zeros((), dtype=torch.long, device=ctx.device)
        self._total = torch.zeros((), dtype=torch.long, device=ctx.device)

    @torch.no_grad()
    def score_batch(self, ctx: EvalContext, batch):
        """Score one val batch against the per-rank index; accumulate correct/total."""
        cfg = self._cfg(ctx)
        K = getattr(cfg, "k", KNN_DEFAULT_K)
        amp = getattr(cfg, "amp", True)
        images, labels = _unpack_batch(batch)
        images = images.to(ctx.device, non_blocking=True)
        labels = labels.to(ctx.device, non_blocking=True)
        with torch.autocast(ctx.device.type, dtype=torch.bfloat16, enabled=amp):
            summary, _ = ctx.model(images)
        summary = F.normalize(summary.float(), p=2, dim=1)
        vote_id = knn_predict(
            self._train_emb, self._train_lbl, summary, K, self._num_classes, ctx.distributed)
        # distributed_topk returns this rank's own queries -> align with labels.
        self._correct += (vote_id == labels[:vote_id.shape[0]]).sum()
        self._total += vote_id.shape[0]

    def aggregate(self, ctx: EvalContext):
        """Reduce correct/total across ranks into ``{knn_top1: acc}``."""
        if ctx.distributed and is_dist_avail_and_initialized():
            dist.all_reduce(self._correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._total, op=dist.ReduceOp.SUM)
        total = max(int(self._total.item()), 1)
        acc = 100.0 * int(self._correct.item()) / total
        # free the index
        self._train_emb = self._train_lbl = None
        return {"knn_top1": acc}
