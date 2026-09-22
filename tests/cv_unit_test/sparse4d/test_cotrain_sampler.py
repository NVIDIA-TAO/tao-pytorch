# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for route-synchronized Sparse4D co-training sampling."""

import json
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.cv.sparse4d.dataloader.callbacks import (
    PklResampleCallback,
)
from nvidia_tao_pytorch.cv.sparse4d.dataloader.sampler import GroupInBatchSampler
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule


pytestmark = pytest.mark.cv_unit


class _SamplerDataset:
    """Small dataset stub exposing the sampler's sequence contract."""

    def __init__(
        self,
        scene_names,
        ann_file="",
        sync_route=True,
        real_block_prob=0.5,
        scene_switch_iters=1,
    ):
        self.data_infos = [{"scene_name": scene_name} for scene_name in scene_names]
        self.flag = np.arange(len(scene_names), dtype=np.int64)
        self.scene_flag = np.arange(len(scene_names), dtype=np.int64)
        self.same_scene_in_batch = True
        self.sequences_split_num = 1
        self.keep_consistent_seq_aug = True
        self.sync_route = sync_route
        self.real_scene_keywords = ["Real"]
        self.real_block_prob = real_block_prob
        self.scene_switch_iters = scene_switch_iters
        self.ann_file = ann_file

    def __len__(self):
        return len(self.data_infos)

    @staticmethod
    def get_augmentation():
        return None


def _route(dataset, scene_idx):
    name = dataset.data_infos[scene_idx]["scene_name"]
    return "real" if "Real" in name else "synthetic"


def test_distributed_scene_blocks_stay_on_the_same_route():
    """Rank slicing of the shared scene sequence never mixes supervision routes."""
    dataset = _SamplerDataset(
        ["RealA", "RealB", "SyntheticA", "SyntheticB"],
        real_block_prob=0.5,
    )
    rank0 = GroupInBatchSampler(dataset, batch_size=1, world_size=2, rank=0, seed=17)
    rank1 = GroupInBatchSampler(dataset, batch_size=1, world_size=2, rank=1, seed=17)
    scenes0 = rank0.scene_indices_per_rank_idx[0]
    scenes1 = rank1.scene_indices_per_rank_idx[1]

    observed_routes = set()
    for _ in range(30):
        route0 = _route(dataset, next(scenes0))
        route1 = _route(dataset, next(scenes1))
        assert route0 == route1
        observed_routes.add(route0)
    assert observed_routes == {"real", "synthetic"}


@pytest.mark.parametrize("cadence", [0, -1])
def test_distributed_route_sync_requires_positive_cadence(cadence):
    """Different sequence lengths must not silently desynchronize DDP routes."""
    dataset = _SamplerDataset(["RealA", "SyntheticA"], scene_switch_iters=cadence)
    with pytest.raises(ValueError, match="requires scene_switch_iters > 0"):
        GroupInBatchSampler(dataset, batch_size=1, world_size=2, rank=0, seed=17)


def test_reloaded_loader_varies_seed_by_epoch_reproducibly(monkeypatch):
    """Same seed/epoch reproduces order; the next epoch advances it."""
    monkeypatch.setenv("RANK", "0")
    dm = Sparse4DDataModule.__new__(Sparse4DDataModule)
    dm.trainer = SimpleNamespace(current_epoch=0)
    dm.train_config = {"seed": 17, "num_gpus": 1, "num_nodes": 1}
    dm.train_dataset = _SamplerDataset([f"Synthetic{i}" for i in range(20)])
    dm.batch_size = 1
    dm.num_workers = 0
    def order(epoch):
        dm.trainer.current_epoch = epoch
        sampler = dm.train_dataloader().batch_sampler
        return list(islice(sampler._infinite_group_indices(), 20))
    assert order(0) == order(0)
    assert order(0) != order(1)
    assert order(1) == order(1)


def test_real_block_probability_edges_select_expected_route():
    """Explicit zero and one probabilities select only synthetic or real blocks."""
    names = ["RealA", "RealB", "SyntheticA", "SyntheticB"]
    for probability, expected_route in ((0.0, "synthetic"), (1.0, "real")):
        dataset = _SamplerDataset(names, real_block_prob=probability)
        sampler = GroupInBatchSampler(
            dataset, batch_size=1, world_size=2, rank=0, seed=3
        )
        scenes = sampler.scene_indices_per_rank_idx[0]
        assert {_route(dataset, next(scenes)) for _ in range(20)} == {expected_route}


def test_synthetic_weight_sidecar_is_loaded_and_refreshed(tmp_path):
    """Per-scene synthetic weights are discovered beside the split file."""
    ann_file = tmp_path / "train.txt"
    sidecar = tmp_path / "train.synth_weights.json"
    sidecar.write_text(
        json.dumps({"SyntheticA": 0.25, "SyntheticB": 2.0}),
        encoding="utf-8",
    )
    dataset = _SamplerDataset(
        ["RealA", "SyntheticA", "SyntheticB"],
        ann_file=str(ann_file),
    )
    sampler = GroupInBatchSampler(dataset, batch_size=1, seed=5)
    torch.testing.assert_close(sampler._synth_weights, torch.tensor([0.25, 2.0]))

    dataset.data_infos[1]["scene_name"] = "RealB"
    sampler.update_from_dataset()
    assert sampler._real_scene_idxs == [0, 1]
    assert sampler._synth_scene_idxs == [2]
    torch.testing.assert_close(sampler._synth_weights, torch.tensor([2.0]))


def test_fixed_scene_cadence_refills_same_scene_until_boundary():
    """Short groups do not make ranks switch scenes before the shared boundary."""
    dataset = _SamplerDataset(
        ["SceneA", "SceneB"],
        sync_route=False,
        scene_switch_iters=3,
    )
    sampler = GroupInBatchSampler(dataset, batch_size=1, seed=11)
    batches = iter(sampler)
    indices = [next(batches)[0]["idx"] for _ in range(4)]

    assert indices[0] == indices[1] == indices[2]
    assert indices[3] != indices[2]


def test_invalid_real_block_probability_is_rejected():
    """Invalid route probabilities fail during sampler construction."""
    dataset = _SamplerDataset(["RealA", "SyntheticA"], real_block_prob=1.1)
    with pytest.raises(ValueError, match="real_block_prob"):
        GroupInBatchSampler(dataset, batch_size=1, seed=0)


def test_pkl_resample_rebuilds_sampler_and_retires_workers():
    """Epoch resampling cannot leave the next loader on stale worker copies."""
    events = []

    class Dataset:
        def resample_pkls(self, epoch):
            events.append(("resample", epoch))

    class BatchSampler:
        def update_from_dataset(self):
            events.append(("sampler", None))

    class Iterator:
        def _shutdown_workers(self):
            events.append(("shutdown", None))

    loader = type("Loader", (), {})()
    loader.dataset = Dataset()
    loader.batch_sampler = BatchSampler()
    loader._iterator = Iterator()
    trainer = type("Trainer", (), {})()
    trainer.current_epoch = 2
    trainer.global_step = 30
    trainer.train_dataloader = loader

    PklResampleCallback(num_iters_per_epoch=10).on_train_epoch_end(
        trainer, pl_module=None
    )

    assert events == [
        ("resample", 3),
        ("sampler", None),
        ("shutdown", None),
    ]
    assert loader._iterator is None


class _EpochDataset(torch.utils.data.Dataset):
    """Expose worker-side copies of a resampled epoch without external files."""

    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.epoch

    def resample_pkls(self, epoch):
        self.epoch = epoch


def test_resampling_retires_real_persistent_workers():
    """Exercise the private worker retirement API in the installed Torch runtime."""
    loader = torch.utils.data.DataLoader(
        _EpochDataset(), num_workers=1, persistent_workers=True,
        multiprocessing_context="spawn",
    )
    iterator = iter(loader)
    try:
        assert next(iterator).item() == 0
        trainer = SimpleNamespace(current_epoch=2, global_step=12, train_dataloader=loader)
        PklResampleCallback(4).on_train_epoch_end(trainer, pl_module=None)
        assert loader._iterator is None
        assert all(not worker.is_alive() for worker in iterator._workers)
        assert next(iter(loader)).item() == 3
    finally:
        iterator._shutdown_workers()
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
