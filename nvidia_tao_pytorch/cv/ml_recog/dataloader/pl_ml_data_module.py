# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric Learning Data Module"""

from typing import Optional
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.ml_recog.dataloader.build_data_loader import build_dataloader, build_inference_dataloader


class MLDataModule(pl.LightningDataModule):
    """Lightning DataModule for Metric Learning."""

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
            (self.train_loader, self.query_loader, self.gallery_loader,
                self.dataset_dict) = build_dataloader(cfg=self.experiment_spec, mode="train")
            self.class_dict = self.dataset_dict["query"].class_dict
        elif stage == 'test':
            _, _, self.test_loader, self.dataset_dict = build_dataloader(self.experiment_spec, mode="eval")
            self.class_dict = self.dataset_dict["query"].class_dict
        elif stage == 'predict':
            _, _, _, self.dataset_dict = build_dataloader(self.experiment_spec, mode="inference")
            self.class_dict = self.dataset_dict["gallery"].class_dict
        else:
            pass

    def train_dataloader(self):
        """Builds the dataloader for training.

        Returns:
            train_loader (torch.utils.data.Dataloader): Traininig Data.
        """
        return self.train_loader

    def test_dataloader(self):
        """Builds the dataloader for testing.

        Returns:
            test_loader (torch.utils.data.Dataloader): Testing Data.
        """
        # In reality, this dataloader isn't used but is necessary for Trainer.test() to not error
        return self.test_loader

    def predict_dataloader(self):
        """Builds the dataloader for inference.

        Returns:
            predict_loader (torch.utils.data.Dataloader): Inference Data.
        """
        predict_loader = build_inference_dataloader(self.experiment_spec)
        return predict_loader
