# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-sample DINO training loader."""

import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler

from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.per_sample_loader import PerSampleBatchLoader


class _IdDataset(Dataset):
    """Returns the index and the id of the worker that produced it."""

    def __init__(self, size=16, fail_at=None):
        self.size = size
        self.fail_at = fail_at

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx == self.fail_at:
            raise RuntimeError("boom")
        info = torch.utils.data.get_worker_info()
        return {"idx": idx, "worker": -1 if info is None else info.id}


def _collate(samples):
    return {
        "idx": [s["idx"] for s in samples],
        "worker": [s["worker"] for s in samples],
    }


def _loader(dataset, batch_size=4, num_workers=2):
    return PerSampleBatchLoader(
        dataset,
        SequentialSampler(dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=False,
    )


@pytest.mark.ssl_unit
def test_order_len_and_drop_last():
    loader = _loader(_IdDataset(size=10), batch_size=4)
    assert len(loader) == 2
    batches = list(loader)
    assert [b["idx"] for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.ssl_unit
def test_one_batch_uses_multiple_workers():
    batches = list(_loader(_IdDataset(size=16), batch_size=4, num_workers=2))
    assert all(len(set(b["worker"])) > 1 for b in batches)


@pytest.mark.ssl_unit
def test_epochs_can_be_iterated_repeatedly():
    loader = _loader(_IdDataset(size=8), batch_size=4)
    assert len(list(loader)) == len(list(loader)) == 2


@pytest.mark.ssl_unit
def test_worker_error_propagates():
    with pytest.raises(RuntimeError, match="boom"):
        list(_loader(_IdDataset(size=8, fail_at=5), batch_size=4))


@pytest.mark.ssl_unit
def test_early_exit_does_not_hang():
    loader = _loader(_IdDataset(size=64), batch_size=4)
    for _ in loader:
        break
    assert len(list(loader)) == 16


@pytest.mark.ssl_unit
def test_exposes_sampler_and_dataset_for_lightning():
    dataset = _IdDataset(size=8)
    sampler = SequentialSampler(dataset)
    loader = PerSampleBatchLoader(dataset, sampler, 4, 1, _collate, persistent_workers=False)
    assert loader.sampler is sampler and loader.dataset is dataset and loader.batch_size == 4


@pytest.mark.ssl_unit
def test_requires_workers():
    with pytest.raises(ValueError):
        PerSampleBatchLoader(_IdDataset(), SequentialSampler(_IdDataset()), 4, 0, _collate)
