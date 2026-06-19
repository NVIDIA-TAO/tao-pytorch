# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optical Inspection Data Module"""

import os
from typing import Optional
import math
import pandas as pd
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader


# Default (ImageNet) input normalization, and the CLIP/OpenAI normalization that C-RADIO
# backbones are trained with (baked into the model's input_conditioner). When a C-RADIO
# backbone is loaded into the VCN ViT-Adapter the conditioner is stripped
# (load_state_dict(strict=False)), so the matching normalization must be applied in the
# dataloader instead of the ImageNet default — otherwise the backbone receives
# out-of-distribution inputs (std off by ~17-23% per channel).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_DEFAULT_STDS = ([0.229, 0.224, 0.225], [0.226, 0.226, 0.226])
_CRADIO_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CRADIO_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


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
        if changenet:
            self._apply_cradio_normalization()

    def _apply_cradio_normalization(self):
        """Use the backbone's expected input normalization for C-RADIO backbones.

        C-RADIO is trained with CLIP/OpenAI pixel normalization (baked into its
        input_conditioner, which is dropped when loading into the VCN ViT-Adapter). Feed the
        backbone CLIP-normalized inputs instead of the ImageNet default so it operates
        in-distribution. Only overrides when the configured normalization is still the
        ImageNet default, so explicit user settings are respected.
        """
        try:
            backbone_type = str(self.model_config.backbone.type)
        except Exception:
            return
        if "radio" not in backbone_type.lower():
            return
        aug = self.dataset_config.get("augmentation_config", None)
        if aug is None:
            return
        cur_mean = [float(x) for x in aug.get("rgb_input_mean", [])]
        cur_std = [float(x) for x in aug.get("rgb_input_std", [])]

        def _close(a, b):
            return len(a) == len(b) and all(abs(x - y) < 1e-4 for x, y in zip(a, b))

        is_default = _close(cur_mean, _IMAGENET_MEAN) and any(_close(cur_std, s) for s in _IMAGENET_DEFAULT_STDS)
        if is_default:
            aug["rgb_input_mean"] = list(_CRADIO_CLIP_MEAN)
            aug["rgb_input_std"] = list(_CRADIO_CLIP_STD)
            logging.info(
                f"C-RADIO backbone '{backbone_type}' with default ImageNet normalization "
                f"detected; overriding to CLIP normalization (mean={_CRADIO_CLIP_MEAN}, "
                f"std={_CRADIO_CLIP_STD}) to match the backbone's training distribution."
            )

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
