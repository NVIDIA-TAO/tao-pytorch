# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object Detection dataset."""

from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.data_loader.dataset import CustomImageDataset


class OCDDataModule(pl.LightningDataModule):
    """Lightning DataModule for OCDNet."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec['dataset']
        self.model_config = experiment_spec['model']

        self.train_dataset_config = self.dataset_config["train_dataset"]
        self.validate_dataset_config = self.dataset_config["validate_dataset"]
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            self.train_loader = get_dataloader(self.train_dataset_config, self.experiment_spec['train']['num_gpus'] > 1)
            assert self.train_loader is not None, "Train loader does not exist."
            self.train_loader_len = len(self.train_loader)
        elif stage == 'predict':
            input_path = self.experiment_spec['inference']["input_folder"]
            width = self.experiment_spec['inference']['width']
            height = self.experiment_spec['inference']['height']
            img_mode = self.experiment_spec['inference']['img_mode']
            self.predict_dataset = CustomImageDataset(input_path, width, height, img_mode)
        elif stage == 'calibration':
            calib_cfg = self.dataset_config.get("quant_calibration_dataset", {})
            if isinstance(calib_cfg, dict):
                calib_images_dir = calib_cfg.get("images_dir", "")
            else:
                calib_images_dir = getattr(calib_cfg, "images_dir", "")

            if calib_images_dir:
                width = self.experiment_spec.get('inference', {}).get('width', 1280)
                height = self.experiment_spec.get('inference', {}).get('height', 736)
                img_mode = self.experiment_spec.get('inference', {}).get('img_mode', 'BGR')
                self.calib_dataset = CustomImageDataset(
                    calib_images_dir, width, height, img_mode
                )
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader (Dataloader): Traininig Data.

        """
        return self.train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader (Dataloader): Validation Data.

        """
        if 'validate_dataset' in self.dataset_config:
            val_loader = get_dataloader(self.validate_dataset_config, False)
        else:
            val_loader = None

        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader (Dataloader): Evaluation Data.

        """
        test_loader = get_dataloader(self.validate_dataset_config, False)

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader (Dataloader): Inference Data.

        """
        predict_loader = DataLoader(self.predict_dataset, batch_size=1)

        return predict_loader

    def calib_dataloader(self):
        """Build the dataloader for quantization calibration.

        Returns:
            calib_loader: PyTorch DataLoader used for calibration.
        """
        if self.calib_dataset is None:
            raise ValueError(
                "Calibration dataset is not initialized. "
                "Call setup(stage='calibration') first."
            )
        calib_loader = DataLoader(self.calib_dataset, batch_size=1)
        return calib_loader
