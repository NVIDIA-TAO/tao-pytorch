# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-to-image retrieval evaluator (Recall@k / mAP / NDCG@k) — OURS.

Not part of c-radiov4. Uses the summary (global) embedding: query and gallery
images are embedded via the model adapter, L2-normalized, and ranked by cosine
similarity. Relevance is same-class membership. Reuses the CLIP evaluation
metric helpers (``compute_ap`` / ``compute_ndcg``) so retrieval math is shared
with the multimodal stack; Recall@k is the fraction of queries with at least one
correct match in the top-k.

When query and gallery come from the same set (single ``root``), self-matches are
excluded (diagonal masked). Extraction reuses ``knn.extract_summary_database`` so
the distributed all-gather + normalization path is identical to KNN.
"""

import logging

import numpy as np

from nvidia_tao_pytorch.core.evaluation.base import EvalContext, Evaluator, register_evaluator
from nvidia_tao_pytorch.core.evaluation.knn import _is_main, extract_summary_database

logger = logging.getLogger(__name__)


@register_evaluator
class RetrievalEvaluator(Evaluator):
    """Recall@k / mAP / NDCG@k image-to-image retrieval on the summary embedding."""

    name = "retrieval"
    requires_fit = False
    feature_level = "global"
    supports_online = False

    def _cfg(self, ctx: EvalContext):
        return getattr(ctx.cfg, self.name)

    def _loader(self, ctx, root, split):
        """Build a labeled classification loader for the query/gallery set."""
        # Reuse the KNN evaluator's loader builder logic via the classification API.
        from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
            IMAGENET_MEAN, IMAGENET_STD, build_classification_loader,
        )
        cfg = self._cfg(ctx)
        mean = std = None
        if getattr(cfg, "imagenet_normalize", False):
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        builder = ctx.build_loader or build_classification_loader
        return builder(
            getattr(cfg, "dataset_type", "image_folder"), root,
            batch_size=getattr(cfg, "batch_size", 128),
            num_workers=getattr(cfg, "num_workers", 8),
            crop=getattr(cfg, "crop", 224),
            resize=getattr(cfg, "resize", None),
            interpolation=getattr(cfg, "interpolation", "bicubic"),
            mean=mean, std=std, distributed=ctx.distributed,
            num_classes=getattr(cfg, "num_classes", 1000),
            label_key=getattr(cfg, "label_key", "cls"),
        )

    def run(self, ctx: EvalContext):
        """Embed query+gallery, rank by cosine sim, return Recall@k / mAP / NDCG@k."""
        # Lazy import: keep core/evaluation import light and independent of the
        # multimodal/clip package import chain.
        from nvidia_tao_pytorch.multimodal.clip.model.evaluation.metrics import (
            compute_ap, compute_ndcg,
        )
        cfg = self._cfg(ctx)
        amp = getattr(cfg, "amp", True)
        k_values = list(getattr(cfg, "k_values", (1, 5, 10)))

        gallery_root = cfg.root
        query_root = getattr(cfg, "query_root", None)
        same_set = not query_root or query_root == gallery_root

        gal = self._loader(ctx, gallery_root, "gallery")
        gal_emb, gal_lbl = extract_summary_database(
            gal.loader, ctx.model, ctx.device, amp, total_size=gal.total_size)
        if same_set:
            qry_emb, qry_lbl = gal_emb, gal_lbl
        else:
            qry = self._loader(ctx, query_root, "query")
            qry_emb, qry_lbl = extract_summary_database(
                qry.loader, ctx.model, ctx.device, amp, total_size=qry.total_size)

        results = {}
        if not _is_main():
            return results   # rank 0 holds the full gathered set

        # Cosine sim on L2-normalized embeddings (extract_summary_database normalizes).
        sims = (qry_emb @ gal_emb.T).numpy()
        gal_lbl_np = gal_lbl.numpy()
        qry_lbl_np = qry_lbl.numpy()
        if same_set:
            np.fill_diagonal(sims, -np.inf)   # exclude self-match

        recall_hits = dict.fromkeys(k_values, 0)
        aps, ndcgs = [], {k: [] for k in k_values}

        for i in range(sims.shape[0]):
            order = np.argsort(-sims[i])
            sorted_labels = (gal_lbl_np[order] == qry_lbl_np[i]).astype(np.int32)
            if sorted_labels.sum() == 0:
                continue   # no positive in gallery for this query
            aps.append(compute_ap(sorted_labels))
            for k in k_values:
                recall_hits[k] += int(sorted_labels[:k].any())
                ndcgs[k].append(compute_ndcg(sorted_labels, k))

        n = len(aps)
        if n == 0:
            logger.warning("[retrieval] no queries had a positive gallery match.")
            return {"retrieval_mAP": 0.0}
        results["retrieval_mAP"] = 100.0 * float(np.mean(aps))
        for k in k_values:
            results[f"retrieval_recall@{k}"] = 100.0 * recall_hits[k] / n
            results[f"retrieval_ndcg@{k}"] = 100.0 * float(np.mean(ndcgs[k]))
        logger.info("[retrieval] mAP %.2f%% over %d queries", results["retrieval_mAP"], n)
        return results
