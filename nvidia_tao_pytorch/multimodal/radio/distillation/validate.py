# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KNN-based validation for distillation training.

Top-1 evaluation using k Nearest Neighbors.

This consists in the following steps:
- on each rank, run model through a shard of the training split of the
  evaluation dataset and collect visual embeddings and labels.
- gather embeddings from all ranks; for ImageNet-1k, this is ~1.2M
  embeddings in total and each rank gets its own copy
  (NOTE: this may not scale to bigger evaluation datasets!)
- on each rank, for each batch of the sharded validation split of the
  evaluation dataset:
  - measure similarity between training and validation embeddings
  - extract top-K similarities (K elements per item in the validation batch)
  - extract classes associated with top-K elements
  - extract top class prediction using a weighted majority vote.
  - compare against ground truth and calculate accuracy.

The functions below are designed to plug into Lightning hooks:

- ``build_knn_index``  → call from ``on_validation_epoch_start``
- ``knn_eval_batch``   → call from ``validation_step``
- ``knn_eval_end``     → call from ``on_validation_epoch_end``
"""

import logging
import time
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.radio.distillation.knn_classification import (
    knn_top1_accuracy,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def build_knn_index(
    model: nn.Module,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    train_loader,
    device: torch.device,
    distributed: bool = False,
    max_train_batches: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Embed the training split to build the KNN index.

    Call this from ``on_validation_epoch_start`` so the index is ready
    before ``validation_step`` processes val batches.

    Args:
        model: Student backbone with ``forward_pre_logits(x) -> (summary, features)``.
        normalize_fn: Callable that normalizes a ``[0, 1]`` image tensor.
        train_loader: WDS DataLoader over the training split.
        device: Target device for computation.
        distributed: Whether distributed communication is available.
        max_train_batches: Cap on train-split batches (useful for debugging).

    Returns:
        ``(embeddings, labels)`` tensors, or ``None`` if no batches were
        processed.
    """
    model.eval()
    torch.cuda.empty_cache()

    # Collect all embeddings and labels in arrays.
    all_train_embeddings = []
    all_train_labels = []

    logger.info("Embedding training split for KNN index...")
    t0 = time.time()

    # Note: we are going through the *training* split of the evaluation
    # dataset here.
    for batch_idx, batch in enumerate(train_loader):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break

        images = batch["img"].to(device, non_blocking=True)
        labels = batch["class"].to(device, non_blocking=True)

        images = normalize_fn(images)
        summary, _ = model.forward_pre_logits(images)
        summary = F.normalize(summary, p=2, dim=1)

        all_train_embeddings.append(summary)
        # Add labels to an array, we will concatenate them later.
        all_train_labels.append(labels)

        if batch_idx % 50 == 0:
            logger.info("  Train embed batch %d", batch_idx)

    if not all_train_embeddings:
        logger.warning("No training batches processed — skipping KNN validation")
        return None

    # Convert all the lists of tensors into tensors.
    all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    num_train = torch.tensor(
        all_train_labels.shape[0], dtype=torch.int64, device=device,
    )
    if distributed:
        dist.reduce(num_train, dst=0, op=dist.ReduceOp.SUM)
    logger.info(
        "Embedded %d training samples in %.1fs (total across ranks: %d)",
        all_train_labels.shape[0],
        time.time() - t0,
        num_train.item(),
    )

    torch.cuda.empty_cache()
    return all_train_embeddings, all_train_labels


@torch.no_grad()
def knn_eval_batch(
    model: nn.Module,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    batch: dict,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    K: int = 20,
    num_classes: int = 1000,
    distributed: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Evaluate a single val batch against the KNN index.

    Call this from ``validation_step``.

    Args:
        model: Student backbone.
        normalize_fn: Callable that normalizes a ``[0, 1]`` image tensor.
        batch: Dict with ``img`` and ``class`` keys.
        train_embeddings: ``[N_train, C]`` L2-normalized training embeddings.
        train_labels: ``[N_train]`` training class labels.
        device: Target device.
        K: Number of nearest neighbors.
        num_classes: Number of classes.
        distributed: Whether distributed communication is available.

    Returns:
        ``(knn_acc, batch_size)`` — accuracy for this batch (0–100) and
        the number of samples.
    """
    images = batch["img"].to(device, non_blocking=True)
    labels = batch["class"].to(device, non_blocking=True)

    images = normalize_fn(images)
    summary, _ = model.forward_pre_logits(images)
    summary = F.normalize(summary, p=2, dim=1)

    knn_acc = knn_top1_accuracy(
        train_split_embeddings=train_embeddings,
        train_split_labels=train_labels,
        K=K,
        output=summary,
        target=labels,
        distributed=distributed,
        num_classes=num_classes,
    )

    return knn_acc, labels.shape[0]
