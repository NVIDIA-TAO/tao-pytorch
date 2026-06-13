# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the KNN evaluator vote math (offline FAISS + brute-force + online)."""

import pytest
import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.core.evaluation import knn as knn_mod
from nvidia_tao_pytorch.core.evaluation.knn import (
    KNN_DEFAULT_K,
    KNN_TEMPERATURE,
    _get_vote_cls,
    knn_predict,
    knn_top1_offline,
)

NUM_CLASSES = 3
K = 10


def _separable(n_per, dim=16, seed=0):
    """n_per L2-normalized samples around each of NUM_CLASSES orthonormal prototypes."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.eye(NUM_CLASSES, dim)
    embs, lbls = [], []
    for c in range(NUM_CLASSES):
        e = protos[c] + 0.01 * torch.randn(n_per, dim, generator=g)
        embs.append(F.normalize(e, dim=1))
        lbls.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(embs), torch.cat(lbls)


@pytest.mark.unit
def test_protocol_constants():
    """KNN protocol constants must stay at the paper-validated values."""
    assert KNN_TEMPERATURE == 0.07
    assert KNN_DEFAULT_K == 20


@pytest.mark.unit
def test_get_vote_cls_separable():
    """A single dominant neighbor class wins the weighted vote."""
    sim = torch.tensor([[0.9, 0.8, 0.1]])
    labels = torch.tensor([[1, 1, 2]])
    assert _get_vote_cls(sim, labels, num_classes=NUM_CLASSES).item() == 1


@pytest.mark.unit
def test_bruteforce_separable_100(monkeypatch):
    """Brute-force fallback (FAISS forced off) classifies separable data ~100%."""
    monkeypatch.setattr(knn_mod, "_HAS_FAISS", False)
    tr_e, tr_l = _separable(30)
    va_e, va_l = _separable(10, seed=1)
    acc = knn_top1_offline(tr_e, tr_l, va_e, va_l, num_classes=NUM_CLASSES, K=K)
    assert acc > 99.0


@pytest.mark.unit
def test_use_faiss_false_forces_bruteforce():
    """use_faiss=False uses the brute-force path even when faiss is installed."""
    tr_e, tr_l = _separable(30)
    va_e, va_l = _separable(10, seed=5)
    acc = knn_top1_offline(tr_e, tr_l, va_e, va_l, num_classes=NUM_CLASSES, K=K, use_faiss=False)
    assert acc > 99.0


@pytest.mark.unit
def test_online_knn_predict_separable_100():
    """Online distributed_topk path (single process) classifies separable data ~100%."""
    tr_e, tr_l = _separable(30)
    va_e, va_l = _separable(10, seed=2)
    vote = knn_predict(tr_e, tr_l, va_e, K=K, num_classes=NUM_CLASSES, distributed=False)
    acc = 100.0 * (vote == va_l).float().mean().item()
    assert acc > 99.0


@pytest.mark.unit
@pytest.mark.skipif(not knn_mod._HAS_FAISS, reason="faiss not installed")
def test_faiss_matches_bruteforce(monkeypatch):
    """FAISS and brute-force agree on separable data."""
    tr_e, tr_l = _separable(30)
    va_e, va_l = _separable(10, seed=3)
    acc_faiss = knn_top1_offline(tr_e, tr_l, va_e, va_l, num_classes=NUM_CLASSES, K=K)
    monkeypatch.setattr(knn_mod, "_HAS_FAISS", False)
    acc_bf = knn_top1_offline(tr_e, tr_l, va_e, va_l, num_classes=NUM_CLASSES, K=K)
    assert abs(acc_faiss - acc_bf) < 1e-6
