# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Samplers for Sparse4D dataset."""

import time
import itertools

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from nvidia_tao_pytorch.core.tlt_logging import logging


def _build_group_map(flag, num_groups):
    """Build group_idx -> [sample_idxs] mapping using argsort+split (O(n log n))."""
    sorted_indices = np.argsort(flag, kind='stable')
    sorted_flags = flag[sorted_indices]
    split_points = np.where(np.diff(sorted_flags) != 0)[0] + 1
    groups_split = np.split(sorted_indices, split_points)
    unique_groups = np.unique(flag)
    group_to_samples = {
        group_idx: groups_split[i].tolist()
        for i, group_idx in enumerate(unique_groups)
    }
    return {
        group_idx: group_to_samples.get(group_idx, [])
        for group_idx in range(num_groups)
    }


class GroupInBatchSampler(Sampler):
    """Sampler that returns groups of items from the same sequence.

    This sampler maintains sequence consistency for temporal data and ensures
    samples from the same scene/sequence are grouped together in batches.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 skip_prob=0.5,
                 sequence_flip_prob=0.1):
        """Initialize GroupInBatchSampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            world_size: Number of distributed processes
            rank: Current process rank
            seed: Random seed
            skip_prob: Probability to skip frames
            sequence_flip_prob: Probability to flip sequence order
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size if world_size is not None else 1
        self.rank = rank if rank is not None else 0
        self.seed = self._sync_random_seed(seed)
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

        self.size = len(self.dataset)

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag

        t0 = time.time()
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        assert self.groups_num >= self.global_batch_size, \
            f"groups_num {self.groups_num} should be greater or equal than global_batch_size {self.global_batch_size}"

        self.group_idx_to_sample_idxs = _build_group_map(self.flag, self.groups_num)

        if hasattr(self.dataset, 'same_scene_in_batch') and self.dataset.same_scene_in_batch:
            assert hasattr(self.dataset, 'scene_flag'), "scene_flag is not present in the dataset"
            self.scene_flag = self.dataset.scene_flag
            self.scene_sizes = np.bincount(self.scene_flag)
            self.scenes_num = len(self.scene_sizes)
            self.sequences_split_num = self.dataset.sequences_split_num

            assert self.sequences_split_num >= self.batch_size, \
                f"sequences_split_num: {self.sequences_split_num} should be >= batch_size: {self.batch_size}"

            self.scene_idx_to_sample_idxs = _build_group_map(self.scene_flag, self.scenes_num)

            self.scene_idx_to_group_idxs = {
                scene_idx: sorted(list(set(self.flag[self.scene_idx_to_sample_idxs[scene_idx]])))
                for scene_idx in range(self.scenes_num)
            }

            self.scene_indices_per_rank_idx = [
                self._scene_indices_per_rank_idx(rank_idx)
                for rank_idx in range(self.world_size)
            ]

            self.group_indices_per_scene_and_local_sample_idx = [
                [
                    self._group_indices_per_scene_and_local_sample_idx(scene_idx, local_sample_idx)
                    for local_sample_idx in range(self.batch_size)
                ]
                for scene_idx in range(self.scenes_num)
            ]
        else:
            self.group_indices_per_global_sample_idx = [
                self._group_indices_per_global_sample_idx(
                    self.rank * self.batch_size + local_sample_idx
                )
                for local_sample_idx in range(self.batch_size)
            ]

        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]

        logging.info(
            f"GroupInBatchSampler: built {self.groups_num} groups over {self.size} samples "
            f"in {time.time() - t0:.2f}s"
        )

    def _sync_random_seed(self, seed=None):
        """Synchronize random seed across distributed processes."""
        if seed is None:
            seed = np.random.randint(2**31)

        if self.world_size > 1:
            # Use PyTorch's distributed API to sync seed
            rank, _ = self.rank, self.world_size

            if rank == 0:
                random_num = torch.tensor(seed, dtype=torch.int32, device="cuda")
            else:
                random_num = torch.tensor(0, dtype=torch.int32, device="cuda")

            if dist.is_available() and dist.is_initialized():
                dist.broadcast(random_num, src=0)

            return random_num.item()

        return seed

    def _infinite_group_indices(self):
        """Generate infinite sequence of random group indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        """Get group indices for a global sample index."""
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def _infinite_scene_indices(self):
        """Generate infinite sequence of random scene indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.scenes_num, generator=g).tolist()

    def _scene_indices_per_rank_idx(self, rank_idx):
        """Get scene indices for a rank index."""
        yield from itertools.islice(
            self._infinite_scene_indices(),
            rank_idx,
            None,
            self.world_size,
        )

    def _infinite_group_indices_per_scene_idx(self, scene_idx):
        """Generate infinite sequence of random group indices for a scene."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        permuted_indices = torch.randperm(
            len(self.scene_idx_to_group_idxs[scene_idx]), generator=g
        ).tolist()
        while True:
            yield from [
                self.scene_idx_to_group_idxs[scene_idx][i] for i in permuted_indices
            ]

    def _group_indices_per_scene_and_local_sample_idx(self, scene_idx, local_sample_idx):
        """Get group indices for a scene and local sample index."""
        yield from itertools.islice(
            self._infinite_group_indices_per_scene_idx(scene_idx),
            local_sample_idx,
            None,
            self.batch_size,
        )

    def __iter__(self):
        """Iterate over batches."""
        while True:
            curr_batch = []
            if hasattr(self.dataset, 'same_scene_in_batch') and self.dataset.same_scene_in_batch:
                if any(len(buffer) == 0 for buffer in self.buffer_per_local_sample):
                    # If any buffer is empty, get a new scene_idx
                    try:
                        new_scene_idx = next(self.scene_indices_per_rank_idx[self.rank])
                    except StopIteration:
                        return
                    # Reset all buffers
                    self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

            for local_sample_idx in range(self.batch_size):
                skip = (np.random.uniform() < self.skip_prob and len(self.buffer_per_local_sample[local_sample_idx]) > 1)

                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    if hasattr(self.dataset, 'same_scene_in_batch') and self.dataset.same_scene_in_batch:
                        try:
                            new_group_idx = next(
                                self.group_indices_per_scene_and_local_sample_idx[new_scene_idx][local_sample_idx]
                            )
                        except StopIteration:
                            return
                    else:
                        # Get group_idx by local_sample_idx in batch
                        try:
                            new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                        except StopIteration:
                            return

                    # Get sample indices for this group
                    self.buffer_per_local_sample[local_sample_idx] = self.group_idx_to_sample_idxs[new_group_idx].copy()

                    # Randomly flip sequence if needed
                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[local_sample_idx] = self.buffer_per_local_sample[local_sample_idx][::-1]

                    # Generate consistent augmentation for sequence if needed
                    if self.dataset.keep_consistent_seq_aug:
                        self.aug_per_local_sample[local_sample_idx] = self.dataset.get_augmentation()

                # Generate new augmentation for each frame if not keeping consistent
                if not self.dataset.keep_consistent_seq_aug:
                    self.aug_per_local_sample[local_sample_idx] = self.dataset.get_augmentation()

                # Skip frames if needed
                if skip:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)

                # Add to batch
                curr_batch.append(
                    dict(
                        idx=self.buffer_per_local_sample[local_sample_idx].pop(0),
                        aug_config=self.aug_per_local_sample[local_sample_idx],
                    )
                )

            yield curr_batch

    def update_from_dataset(self):
        """Refresh internal state after the dataset has been resampled.

        Must be called whenever dataset.data_infos, dataset.flag, or
        dataset.scene_flag are rebuilt (e.g. by resample_pkls).
        """
        t0 = time.time()
        self.size = len(self.dataset)
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)

        self.group_idx_to_sample_idxs = _build_group_map(self.flag, self.groups_num)

        if hasattr(self.dataset, 'same_scene_in_batch') and self.dataset.same_scene_in_batch:
            self.scene_flag = self.dataset.scene_flag
            self.scene_sizes = np.bincount(self.scene_flag)
            self.scenes_num = len(self.scene_sizes)

            self.scene_idx_to_sample_idxs = _build_group_map(self.scene_flag, self.scenes_num)

            self.scene_idx_to_group_idxs = {
                scene_idx: sorted(list(set(self.flag[self.scene_idx_to_sample_idxs[scene_idx]])))
                for scene_idx in range(self.scenes_num)
            }

            self.scene_indices_per_rank_idx = [
                self._scene_indices_per_rank_idx(rank_idx)
                for rank_idx in range(self.world_size)
            ]
            self.group_indices_per_scene_and_local_sample_idx = [
                [
                    self._group_indices_per_scene_and_local_sample_idx(scene_idx, local_sample_idx)
                    for local_sample_idx in range(self.batch_size)
                ]
                for scene_idx in range(self.scenes_num)
            ]
        else:
            self.group_indices_per_global_sample_idx = [
                self._group_indices_per_global_sample_idx(
                    self.rank * self.batch_size + local_sample_idx
                )
                for local_sample_idx in range(self.batch_size)
            ]

        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]

        logging.info(
            f"GroupInBatchSampler: rebuilt {self.groups_num} groups over {self.size} samples "
            f"in {time.time() - t0:.2f}s"
        )

    def __len__(self):
        """Get dataset length."""
        return self.size

    def set_epoch(self, epoch):
        """Set the epoch for reproducibility."""
        self.epoch = epoch
