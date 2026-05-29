# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K-Nearest Neighbors classification for validation metrics."""

from typing import Tuple

import torch
import torch.distributed as dist


def _get_vote_cls(
    max_sim: torch.Tensor,
    max_ids: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Weighted majority vote from top-K neighbors.

    Uses temperature-scaled exponential weighting following
    https://arxiv.org/pdf/1805.01978.pdf, Section 3.4.
    """
    # https://arxiv.org/pdf/1805.01978.pdf, Section 3.4 (Also used by DINO)
    weights = torch.exp(max_sim / 0.07)
    cls_vec = torch.zeros(
        weights.shape[0], num_classes, dtype=weights.dtype, device=weights.device,
    )
    cls_vec.scatter_add_(dim=1, index=max_ids, src=weights)

    # The predicted ID is the one with the most vote weight
    vote_id = torch.argmax(cls_vec, dim=1)
    return vote_id


def _pad(tensor: torch.Tensor, dim0: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad *tensor* along dim-0 to *dim0* and return a validity mask."""
    valid_mask = torch.ones(dim0, dtype=torch.bool, device=tensor.device)
    valid_mask[tensor.shape[0]:].fill_(False)

    if tensor.shape[0] == dim0:
        # If there is no padding to be done, return the original tensor.
        return tensor, valid_mask

    # Copy valid elements into a new tensor.
    ret = torch.empty(dim0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    ret[:tensor.shape[0]].copy_(tensor)
    return ret, valid_mask


def _all_to_all(t: torch.Tensor) -> torch.Tensor:
    # Unroll the world dim into a list of tensors
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
        labels: ``[D]`` class labels for *keys*.
        K: number of nearest neighbors.
        distributed: whether to use distributed communication.

    Returns:
        ``(max_sim, max_labels)`` each of shape ``[Q, K]``.
    """
    if distributed:
        world_size = dist.get_world_size()
        max_queries = torch.tensor(
            queries.shape[0], dtype=torch.int64, device=queries.device,
        )
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
        valid_mask = torch.ones(
            queries.shape[0], dtype=torch.bool, device=queries.device,
        )

    # all_queries: W,Q,C
    # keys: D,C

    # W,Q,D
    similarity = torch.matmul(all_queries, keys.T)

    # W,Q,K
    max_sim, max_idxs = torch.topk(similarity, k=K, dim=2, largest=True, sorted=False)
    max_labels = labels[max_idxs.flatten()].reshape_as(max_idxs)

    if distributed:
        # Queries is the concatenated list of queries for all ranks,
        # which means that max_sim and max_idxs is AllQueries -> KeysForThisRank
        # All to all will then rearrange these to QueriesForThisRank -> AllKeys
        max_sim = _all_to_all(max_sim)
        max_labels = _all_to_all(max_labels)

    # Reduce the per-rank similarities
    # N,K*W
    max_sim = max_sim.permute(1, 2, 0).flatten(1)
    max_labels = max_labels.permute(1, 2, 0).flatten(1)

    if distributed:
        max_sim, max_idxs = torch.topk(max_sim, k=K, dim=1, largest=True, sorted=False)
        max_labels = torch.gather(max_labels, dim=1, index=max_idxs)

    max_sim = max_sim[valid_mask]
    max_labels = max_labels[valid_mask]

    return max_sim, max_labels


def knn_top1_accuracy(
    train_split_embeddings: torch.Tensor,
    train_split_labels: torch.Tensor,
    K: int,
    output: torch.Tensor,
    target: torch.Tensor,
    distributed: bool,
    num_classes: int = 1000,
) -> torch.Tensor:
    """K-NN Top-1 classification accuracy.

    Args:
        train_split_embeddings: ``[N_train, C]`` L2-normalized training embeddings.
        train_split_labels: ``[N_train]`` training class labels.
        K: number of nearest neighbors for majority vote.
        output: ``[B, C]`` L2-normalized query embeddings.
        target: ``[B]`` ground-truth class IDs.
        distributed: whether to use distributed communication.
        num_classes: total number of classes (default 1000 for ImageNet).

    Returns:
        Top-1 accuracy as a scalar tensor (percentage, 0–100).
    """
    max_sim, max_labels = distributed_topk(
        queries=output,
        keys=train_split_embeddings,
        labels=train_split_labels,
        K=K,
        distributed=distributed,
    )

    # Get a weighted vote for each validation sample.
    vote_id = _get_vote_cls(max_sim, max_labels, num_classes=num_classes)

    # Compare against ground truth.
    correct = target == vote_id

    # Get total number of correct predictions and calculate accuracy.
    num_correct = correct.sum()
    acc1 = 100.0 * num_correct / output.size(0)

    return acc1
