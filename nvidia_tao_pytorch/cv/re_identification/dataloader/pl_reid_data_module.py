# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-Identification Data Module"""

from typing import Optional
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import build_dataloader


class REIDDataModule(pl.LightningDataModule):
    """Lightning DataModule for Re-Identification."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            self.train_loader, self.val_loader, _, _ = build_dataloader(cfg=self.experiment_spec, is_train=True)

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        return self.train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        return self.val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        _, test_loader, _, _ = build_dataloader(self.experiment_spec, is_train=False)
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        _, predict_loader, _, _ = build_dataloader(self.experiment_spec, is_train=False)
        return predict_loader
