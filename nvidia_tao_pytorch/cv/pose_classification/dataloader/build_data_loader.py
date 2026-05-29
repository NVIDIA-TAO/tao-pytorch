# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build torch data loader."""
from torch.utils.data import DataLoader, distributed, RandomSampler, BatchSampler

from nvidia_tao_pytorch.cv.pose_classification.dataloader.skeleton_feeder import SkeletonFeeder
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized


def build_dataloader(stage, data_path, label_map, label_path=None,
                     random_choose=False, random_move=False,
                     window_size=-1, debug=False, mmap=True,
                     batch_size=1, shuffle=False,
                     num_workers=4, pin_mem=False):
    """
    Build a torch DataLoader from a given data path and label map.

    This function first constructs a SkeletonFeeder dataset from the provided parameters. It then uses this dataset
    to build a DataLoader object, which is a generator that allows for iteration over batches of the dataset.

    Args:
        data_path (str): Path to the data in a NumPy array.
        label_map (dict): Dictionary mapping labels to their corresponding indices.
        label_path (str, optional): Path to the labels in a pickle file. Defaults to None.
        random_choose (bool, optional): Specifies whether to randomly choose a portion of the input sequence. Defaults to False.
        random_move (bool, optional): Specifies whether to randomly move the input sequence. Defaults to False.
        window_size (int, optional): The length of the output sequence. -1 means the same as original length.
        debug (bool, optional): If True, the function will run in debug mode. Defaults to False.
        mmap (bool, optional): If True, memory-mapping mode is used for loading data. Defaults to True.
        batch_size (int, optional): The number of samples per batch. Defaults to 1.
        shuffle (bool, optional): If True, data will be reshuffled at every epoch. Defaults to False.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 4.
        pin_mem (bool, optional): If True, data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance with specified dataset and parameters.
    """
    dataset = SkeletonFeeder(data_path=data_path,
                             label_path=label_path,
                             label_map=label_map,
                             random_choose=random_choose,
                             random_move=random_move,
                             window_size=window_size,
                             debug=debug,
                             mmap=mmap)

    if stage in ("train", None):
        if is_dist_avail_and_initialized():
            train_sampler = distributed.DistributedSampler(dataset, shuffle=True)
        else:
            train_sampler = RandomSampler(dataset)

        dataloader = DataLoader(dataset=dataset,
                                num_workers=num_workers,
                                pin_memory=pin_mem,
                                batch_sampler=BatchSampler(train_sampler, batch_size, drop_last=True))

    if stage in ("val", "test", "predict", None):
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=pin_mem)

    return dataloader
