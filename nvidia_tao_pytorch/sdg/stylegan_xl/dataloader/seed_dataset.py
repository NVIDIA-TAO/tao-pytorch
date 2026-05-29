# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common Seed Dataset for both StyleGAN and BigDatasetGAN"""

import torch


class SeedDataset(torch.utils.data.Dataset):
    """A custom dataset for loading integer seeds"""

    def __init__(self, seeds):
        """Initialize"""
        self.seeds = seeds

    def __len__(self):
        """Return the total number of seeds in the dataset."""
        return len(self.seeds)

    def __getitem__(self, idx):
        """Get a integer seed from the dataset."""
        seed = self.seeds[idx]
        return seed

# # Example list of seeds
# seeds = [42, 123, 256, 512, 1024]
# seed_dataset = SeedDataset(seeds)
