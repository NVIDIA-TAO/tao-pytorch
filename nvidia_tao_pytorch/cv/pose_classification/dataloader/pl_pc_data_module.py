# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose Classification Data Module"""

import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.pose_classification.dataloader.build_data_loader import build_dataloader


class PCDataModule(pl.LightningDataModule):
    """Lightning DataModule for Pose Classification."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_config = experiment_spec
        self.dataset_config = experiment_spec.dataset

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = \
            build_dataloader(stage="train",
                             data_path=self.dataset_config["train_dataset"]["data_path"],
                             label_path=self.dataset_config["train_dataset"]["label_path"],
                             label_map=self.dataset_config["label_map"],
                             random_choose=self.dataset_config["random_choose"],
                             random_move=self.dataset_config["random_move"],
                             window_size=self.dataset_config["window_size"],
                             mmap=True,
                             batch_size=self.dataset_config["batch_size"],
                             shuffle=False,
                             num_workers=self.dataset_config["num_workers"],
                             pin_mem=False)
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = \
            build_dataloader(stage="val",
                             data_path=self.dataset_config["val_dataset"]["data_path"],
                             label_path=self.dataset_config["val_dataset"]["label_path"],
                             label_map=self.dataset_config["label_map"],
                             batch_size=self.dataset_config["batch_size"],
                             num_workers=self.dataset_config["num_workers"])
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = \
            build_dataloader(stage="test",
                             data_path=self.experiment_config["evaluate"]["test_dataset"]["data_path"],
                             label_path=self.experiment_config["evaluate"]["test_dataset"]["label_path"],
                             label_map=self.dataset_config["label_map"],
                             mmap=True,
                             batch_size=self.dataset_config["batch_size"],
                             num_workers=self.dataset_config["num_workers"])

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = \
            build_dataloader(stage="predict",
                             data_path=self.experiment_config["inference"]["test_dataset"]["data_path"],
                             label_map=self.dataset_config["label_map"],
                             mmap=True,
                             batch_size=self.dataset_config["batch_size"],
                             num_workers=self.dataset_config["num_workers"])

        return predict_loader
