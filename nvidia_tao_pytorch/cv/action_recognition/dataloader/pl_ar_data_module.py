# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action Recognition Data Module"""

import os
from typing import Optional
import pytorch_lightning as pl

from nvidia_tao_pytorch.cv.action_recognition.dataloader.build_data_loader import build_dataloader, list_dataset


class ARDataModule(pl.LightningDataModule):
    """Lightning DataModule for Action Recognition."""

    def __init__(self, experiment_spec):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config (OmegaConf): dataset configuration

        """
        super().__init__()
        self.experiment_config = experiment_spec
        self.dataset_config = experiment_spec.dataset
        self.model_config = experiment_spec.model

        self.data_shape = [self.model_config.input_height, self.model_config.input_width]

    def setup(self, stage: Optional[str] = None):
        """ Prepares for each dataloader

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage == 'fit':
            train_top_dir = self.dataset_config["train_dataset_dir"]
            val_top_dir = self.dataset_config["val_dataset_dir"]
            if train_top_dir:
                self.train_dict = list_dataset(train_top_dir)
            else:
                raise ValueError("Please set the train dataset in the spec file")

            if val_top_dir:
                self.val_dict = list_dataset(val_top_dir)
            else:
                self.val_dict = {}

            print("Train dataset samples: {}".format(len(self.train_dict)))
            print("Validation dataset samples: {}".format(len(self.val_dict)))

        elif stage == 'test':
            test_dataset_dir = self.experiment_config.evaluate.test_dataset_dir
            action_set = os.listdir(test_dataset_dir)
            self.sample_dict = {}
            for action in action_set:
                action_root_path = os.path.join(test_dataset_dir, action)
                for video in os.listdir(action_root_path):
                    video_path = os.path.join(action_root_path, video)
                    self.sample_dict[video_path] = action

        elif stage == 'predict':
            inference_dataset_dir = self.experiment_config.inference.inference_dataset_dir
            self.sample_dict = {}
            for sample_id in os.listdir(inference_dataset_dir):
                sample_path = os.path.join(inference_dataset_dir, sample_id)
                self.sample_dict[sample_path] = "unknown"

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        train_loader = build_dataloader(sample_dict=self.train_dict,
                                        model_config=self.model_config,
                                        output_shape=self.data_shape,
                                        label_map=self.dataset_config["label_map"],
                                        dataset_mode="train",
                                        batch_size=self.dataset_config["batch_size"],
                                        workers=self.dataset_config["workers"],
                                        input_type=self.model_config["input_type"],
                                        shuffle=True,
                                        pin_mem=True,
                                        clips_per_video=self.dataset_config["clips_per_video"],
                                        augmentation_config=self.dataset_config["augmentation_config"]
                                        )

        return train_loader

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        val_loader = build_dataloader(sample_dict=self.val_dict,
                                      model_config=self.model_config,
                                      output_shape=self.data_shape,
                                      label_map=self.dataset_config["label_map"],
                                      dataset_mode="val",
                                      batch_size=self.dataset_config["batch_size"],
                                      workers=self.dataset_config["workers"],
                                      input_type=self.model_config["input_type"],
                                      clips_per_video=1,
                                      augmentation_config=self.dataset_config["augmentation_config"]
                                      )

        return val_loader

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        test_loader = build_dataloader(sample_dict=self.sample_dict,
                                       model_config=self.model_config,
                                       dataset_mode="val",
                                       output_shape=self.data_shape,
                                       input_type=self.model_config["input_type"],
                                       label_map=self.dataset_config["label_map"],
                                       batch_size=self.experiment_config["evaluate"]["batch_size"],
                                       workers=self.dataset_config["workers"],
                                       eval_mode=self.experiment_config["evaluate"]["video_eval_mode"],
                                       augmentation_config=self.dataset_config["augmentation_config"],
                                       num_segments=self.experiment_config["evaluate"]["video_num_segments"]
                                       )

        return test_loader

    def predict_dataloader(self):
        """Build the dataloader for inference.

        Returns:
            predict_loader: PyTorch DataLoader used for inference.
        """
        predict_loader = build_dataloader(sample_dict=self.sample_dict,
                                          model_config=self.model_config,
                                          dataset_mode="inf",
                                          output_shape=self.data_shape,
                                          input_type=self.model_config["input_type"],
                                          label_map=self.dataset_config["label_map"],
                                          batch_size=self.experiment_config.inference.batch_size,
                                          workers=self.dataset_config["workers"],
                                          eval_mode=self.experiment_config["inference"]["video_inf_mode"],
                                          augmentation_config=self.dataset_config["augmentation_config"],
                                          num_segments=self.experiment_config["inference"]["video_num_segments"]
                                          )

        return predict_loader
