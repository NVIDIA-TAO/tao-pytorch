# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampler Module for Re-Identification."""

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
_LOCAL_PROCESS_GROUP = None


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities, then for each identity, randomly samples K instances, therefore batch size is N*K.

    RandomIdentitySampler is a subclass of torch.utils.data.sampler. It overrides the __iter__ and __len__ methods based
    on the batch_size, num_samples and num_pids_per_batch. It ensures that for each identity, K consecutive identities
    are obtained.

    Args:
        data_source (list): A list of tuples, where each tuple contains (img_path, pid, camid).
        num_instances (int): Number of instances per identity in a batch.
        batch_size (int): Number of examples in a batch.

    Attributes:
        data_source (list): The list of data provided as input.
        batch_size (int): The number of examples per batch.
        num_instances (int): The number of instances per identity.
        num_pids_per_batch (int): The number of unique identities per batch.
        index_dic (defaultdict): A dictionary where the keys are unique identities (pid) and the values are
                                 lists of indices corresponding to the identities in the data_source.
        pids (list): A list of unique identities (pid) in the data.
        length (int): The estimated number of examples in an epoch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        """Initialize the sampler with the data, batch size, and number of instances.

        Args:
            data_source (list): The list of data.
            batch_size (int): The size of each batch of data.
            num_instances (int): The number of instances per identity.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # Estimate number of examples in an epoch
        self._adjust_length()

    def _adjust_length(self):
        """
        Estimates the total number of batches per GPU for an epoch.

        Updates:
            length (int): Adjusted number of examples per GPU for the epoch.
        """
        # Calculate the total number of complete batches that can be formed.
        total_batches = 0
        for pid in self.pids:
            num_idxs = len(self.index_dic[pid])
            batches_per_pid = num_idxs // self.num_instances
            total_batches += batches_per_pid

        # Adjust for the number of PIDs per batch
        total_full_batches = total_batches // self.num_pids_per_batch
        self.length = total_full_batches * self.batch_size

    def __iter__(self):
        """Create an iterator for the sampler.

        Returns:
            final_idxs (iterator): An iterator over the indices of the images to be included in the batch.
        """
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            else:
                np.random.shuffle(idxs)

            # Gather batches for each PID
            for _ in range(len(idxs) // self.num_instances):
                batch_idxs = idxs[:self.num_instances]
                idxs = idxs[self.num_instances:]
                batch_idxs_dict[pid].append(batch_idxs)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                if batch_idxs_dict[pid]:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if not batch_idxs_dict[pid]:
                        avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        """Return the length of the sampler.

        Returns:
            length (int): The total number of images to be included in the epoch.
        """
        return self.length


class RandomIdentitySamplerDDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
        data_source (list): list of (img_path, pid, camid).
        num_instances (int): number of instances per identity in a batch.
        batch_size (int): number of examples in a batch.
        num_gpus (int): number of gpus used for training.

    Attributes:
        data_source (list): The input data provided as a list of tuples.
        batch_size (int): The total batch size across all GPUs.
        world_size (int): The number of GPUs used for training.
        num_instances (int): The number of instances per identity in a batch.
        mini_batch_size (int): The size of each mini-batch that will be processed on a single GPU.
        num_pids_per_batch (int): The number of unique identities per mini-batch on a single GPU.
        index_dic (defaultdict): A dictionary mapping each identity (pid) to a list of its instances' indices in the data_source.
        pids (list): A list of unique identities (pid) present in the data_source.
        length (int): The estimated total number of examples that will be processed in an epoch, adjusted for the distributed context.
        rank (int): The rank of the current GPU in the distributed training setup.
    """

    def __init__(self, data_source, batch_size, num_instances, num_gpus):
        """Initialize the sampler with the data, batch size, and number of instances.

        Args:
            data_source (list): The list of data.
            batch_size (int): The size of each batch of data.
            num_instances (int): The number of instances per identity.
            num_gpus (int): The number of gpus used for training.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = num_gpus
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # Estimate number of examples in an epoch
        self._adjust_length()

        self.rank = get_global_rank()

    def _adjust_length(self):
        """
        Estimates the total number of batches per GPU for an epoch.

        Updates:
            length (int): Adjusted number of examples per GPU for the epoch.
        """
        # Calculate the total number of instances available for sampling.
        total_instances = sum(min(len(self.index_dic[pid]), self.num_instances * (len(self.index_dic[pid]) // self.num_instances)) for pid in self.pids)

        # Calculate the total number of complete batches that can be formed.
        total_batches = total_instances // (self.num_pids_per_batch * self.num_instances)

        # Adjust length based on the total number of batches and distribute across GPUs.
        self.length = total_batches * self.num_pids_per_batch * self.num_instances // self.world_size

    def __iter__(self):
        """Create an iterator for the sampler.

        Returns:
            final_idxs (iterator): An iterator over the indices of the images to be included in the batch.
        """
        np.random.seed(42)
        self._seed = 42
        final_idxs = self._sample_list()

        # Calculate the length per node and adjust final_idxs
        length_per_node = int(math.floor(len(final_idxs) / self.world_size))
        final_idxs = self._fetch_current_node_idxs(final_idxs, length_per_node)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def _fetch_current_node_idxs(self, final_idxs, length):
        """
        Distributes sampled indices across GPUs by selecting a segment for the current node.

        Returns:
            final_idxs (list): Segmented indices for the current GPU.
        """
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []

        # Adjust the range to ensure all indices are covered
        for i in range(0, block_num * self.world_size, self.world_size):
            start = self.mini_batch_size * self.rank + self.mini_batch_size * i
            end = min(start + self.mini_batch_size, total_num)
            index_target.extend(range(start, end))

        final_idxs = [final_idxs[i] for i in index_target]
        return final_idxs

    def _sample_list(self):
        """
        Generates indices for balanced batch sampling in distributed training.

        Returns:
            batch_indices (list): Indices for balanced batch creation.
        """
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        """Return the length of the sampler.

        Returns:
            length (int): The total number of images to be included in the epoch.
        """
        return self.length
