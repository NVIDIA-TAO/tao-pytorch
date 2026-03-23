# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optical Inspection Data Module"""

import os
from typing import Optional
import math
import pandas as pd
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader


class OIDataModule(pl.LightningDataModule):
    """Lightning DataModule for Optical Inspection."""

    def __init__(self, experiment_spec, changenet=False):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec
        if changenet:
            self.dataset_config = experiment_spec.dataset.classify
        else:
            self.dataset_config = experiment_spec.dataset
        self.model_config = experiment_spec.model

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            train_data_path = self.dataset_config["train_dataset"]["csv_path"]
            val_data_path = self.dataset_config["validation_dataset"]["csv_path"]
            self.df_train = pd.read_csv(train_data_path, dtype={'object_name': str})
            self.df_valid = pd.read_csv(val_data_path, dtype={'object_name': str})

        if stage == 'test':
            eval_data_path = self.dataset_config["test_dataset"]["csv_path"]
            logging.info("test_csv_path {}".format(eval_data_path))
            self.df_test = pd.read_csv(eval_data_path, dtype={'object_name': str})

        if stage == 'predict':
            infer_data_path = self.dataset_config["infer_dataset"]["csv_path"]
            if not os.path.exists(infer_data_path):
                raise FileNotFoundError(f"No inference csv file was found at {infer_data_path}")
            logging.info("Loading inference csv from : {}".format(infer_data_path))
            self.df_infer = pd.read_csv(infer_data_path, dtype={'object_name': str})

        if stage == 'calibration':
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            if isinstance(calib_cfg, dict):
                calib_images_dir = calib_cfg.get("images_dir", "")
            else:
                calib_images_dir = getattr(calib_cfg, "images_dir", "")

            if not calib_images_dir:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )
            logging.info("Loading calibration images from: {}".format(calib_images_dir))
            self.calib_images_dir = calib_images_dir

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = build_dataloader(df=self.df_train,
                                        weightedsampling=True,
                                        split='train',
                                        data_config=self.dataset_config
                                        )
        self.num_train_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        logging.info("Number of steps for training: {}".format(self.num_train_steps_per_epoch))
        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = build_dataloader(df=self.df_valid,
                                      weightedsampling=False,
                                      split='valid',
                                      data_config=self.dataset_config
                                      )
        self.num_val_steps_per_epoch = math.ceil(len(val_loader.dataset) / val_loader.batch_size)
        logging.info("Number of steps for validation: {}".format(self.num_val_steps_per_epoch))
        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = build_dataloader(df=self.df_test,
                                       weightedsampling=True,
                                       split='test',
                                       data_config=self.dataset_config
                                       )
        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        # Building dataloader without weighted sampling for inference.
        predict_loader = build_dataloader(df=self.df_infer,
                                          weightedsampling=False,
                                          split='infer',
                                          data_config=self.dataset_config
                                          )
        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration.

        Returns:
            calib_loader: PyTorch DataLoader used for calibration.
        """
        from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import (
            build_calib_dataloader
        )
        calib_loader = build_calib_dataloader(
            images_dir=self.calib_images_dir,
            data_config=self.dataset_config
        )
        return calib_loader
