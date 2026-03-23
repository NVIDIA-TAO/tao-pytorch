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

"""OCRNet Data Module"""

from typing import Optional
import os
import pytorch_lightning as pl
import torch

from nvidia_tao_pytorch.cv.ocrnet.dataloader.build_dataloader import build_dataloader, translate_dataset_config
from nvidia_tao_pytorch.cv.ocrnet.dataloader.ocr_dataset import AlignCollateVal, LmdbDataset, RawDataset, RawGTDataset
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import create_logger


class OCRDataModule(pl.LightningDataModule):
    """Lightning DataModule for OCRNet."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec.dataset
        self.model_config = experiment_spec.model
        self.calib_dataset = None

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            val_log_file = os.path.join(self.experiment_spec.results_dir, "log_val.txt")
            self.console_logger = create_logger(val_log_file)

            self.train_data_path = self.dataset_config.train_dataset_dir[0]
            self.train_gt_file = self.dataset_config.train_gt_file
            self.val_data_path = self.dataset_config.val_dataset_dir
            self.val_gt_file = self.dataset_config.val_gt_file

        elif stage == 'test':

            test_log_file = os.path.join(self.experiment_spec.results_dir, "log_evaluation.txt")
            self.console_logger = create_logger(test_log_file)
            self.eval_data_path = self.experiment_spec.evaluate.test_dataset_dir
            self.eval_gt_file = self.experiment_spec.evaluate.test_dataset_gt_file

            self.opt = translate_dataset_config(self.experiment_spec)
            self.opt.batch_size = self.experiment_spec.evaluate.batch_size

            self.AlignCollate_func = AlignCollateVal(
                imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
            )

            if self.eval_gt_file:
                self.dataset = RawGTDataset(self.eval_gt_file, self.eval_data_path, self.opt)
            else:
                self.dataset = LmdbDataset(self.eval_data_path, self.opt)

        elif stage == 'predict':
            self.infer_data_path = self.experiment_spec.inference.inference_dataset_dir

            self.opt = translate_dataset_config(self.experiment_spec)
            self.opt.batch_size = self.experiment_spec.inference.batch_size

            self.AlignCollate_func = AlignCollateVal(
                imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
            )
            self.dataset = RawDataset(root=self.infer_data_path, opt=self.opt)

        elif stage == 'calibration':
            calib_cfg = getattr(self.dataset_config, "quant_calibration_dataset", None)
            if calib_cfg is None:
                calib_cfg = {}

            if hasattr(calib_cfg, "images_dir"):
                calib_images_dir = getattr(calib_cfg, "images_dir", "")
            else:
                calib_images_dir = calib_cfg.get("images_dir", "")

            if calib_images_dir:
                self.opt = translate_dataset_config(self.experiment_spec)
                self.opt.batch_size = getattr(
                    self.experiment_spec.inference, "batch_size", 1
                )
                self.AlignCollate_func = AlignCollateVal(
                    imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
                )
                self.calib_dataset = RawDataset(root=calib_images_dir, opt=self.opt)
            else:
                raise ValueError(
                    "quant_calibration_dataset.images_dir must be provided "
                    "for calibration stage."
                )

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = \
            build_dataloader(experiment_spec=self.experiment_spec,
                             data_path=self.train_data_path,
                             gt_file=self.train_gt_file)

        self.console_logger.info(f"Train dataset samples: {len(train_loader.dataset)}")
        self.console_logger.info(f"Train batch num: {len(train_loader)}")

        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = build_dataloader(experiment_spec=self.experiment_spec,
                                      data_path=self.val_data_path,
                                      shuffle=False,
                                      gt_file=self.val_gt_file)

        self.console_logger.info(f"Val dataset samples: {len(val_loader.dataset)}")
        self.console_logger.info(f"Val batch num: {len(val_loader)}")
        self.gpu_num = len(self.experiment_spec.train.gpu_ids)
        self.val_batch_num = int(len(val_loader) / self.gpu_num)

        return val_loader

    def test_dataloader(self):
        """Build the dataloader for testing.

        Returns:
            test_loader: PyTorch DataLoader used for testing.
        """
        test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_func, pin_memory=True)

        self.console_logger.info(f"data directory:\t{self.eval_data_path}")
        self.console_logger.info(f"num samples: {len(test_loader.dataset)}")

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_func, pin_memory=True)

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
        calib_loader = torch.utils.data.DataLoader(
            self.calib_dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_func, pin_memory=True)
        return calib_loader
