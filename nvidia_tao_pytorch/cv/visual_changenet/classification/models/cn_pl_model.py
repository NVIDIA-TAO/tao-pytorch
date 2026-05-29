# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet Classification Model PyTorch Lightning Module"""

import logging
import os
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import AOIMetrics
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.changenet import build_model
from nvidia_tao_pytorch.cv.visual_changenet.classification.losses import ContrastiveLoss


# pylint:disable=too-many-ancestors
class ChangeNetPlModel(TAOLightningModule):
    """ PTL module for Visual ChangeNet Classification Model."""

    def __init__(self, experiment_spec, dm, export=False):
        """Init training for Visual ChangeNet Model."""
        super().__init__(experiment_spec)
        # Overriding what's done in super()
        self.dataset_config = experiment_spec.dataset.classify
        self.train_config = experiment_spec.train

        # Customisable architecture for classification
        self.difference_module = self.model_config.classify.difference_module
        self.learnable_difference_modules = self.model_config.classify.learnable_difference_modules

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.loss_fn = self.train_config.classify.loss
        self.cls_weight = self.train_config.classify.cls_weight
        self.n_class = self.dataset_config.num_classes
        self.batch_size = self.dataset_config.batch_size
        self.num_golden = self.dataset_config.num_golden

        #  training log
        self.epoch_acc = 0
        self.batch = None
        self.G_loss = None

        self.optimizer_G = None
        # init the model
        self._build_model(export)
        self._build_criterion()

        self.tensorboard = experiment_spec.train.tensorboard
        self.train_metrics = AOIMetrics()
        self.val_metrics = AOIMetrics()
        self.dm = dm

        self.checkpoint_filename = 'changenet_model_classify'

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        assert self.difference_module in ["learnable", "euclidean"], "Only 'learnable' and 'euclidean' difference modules supported"
        n_class = self.dataset_config.num_classes
        assert self.loss_fn in ["ce", "contrastive"], "Only CE(Cross Entropy), Contrastive loss are supported"
        self.total_samples = 0
        self.correct_predictions = 0
        if self.loss_fn == 'contrastive':
            self.margin = self.model_config.classify.train_margin_euclid
            assert self.difference_module == 'euclidean', "Contrastive loss only supports Euclidean distance module"
            self.criterion = ContrastiveLoss(self.margin)
        elif self.loss_fn == 'ce':
            assert self.difference_module == 'learnable', "CE (Cross Entropy) loss only supports learnable distance module"
            cls_weight = self.cls_weight
            assert len(cls_weight) == n_class, f"""Class weights must be provided for each class
            Provided weights for {len(cls_weight)} classes when total classes are {n_class}.
            """
            self.class_weights = torch.tensor(cls_weight)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            raise NotImplementedError(f"loss function {self.loss_fn} is not implemented")

    def configure_optimizers(self):
        """Configure optimizers for training"""
        # define optimizers
        if self.optimizer.optim == "sgd":
            self.optimizer_G = optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=self.optimizer.momentum,  # 0.9
                                         weight_decay=self.optimizer.weight_decay)  # 5e-4
        elif self.optimizer.optim == "adam":
            self.optimizer_G = optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.optimizer.weight_decay)  # 0
        elif self.optimizer.optim == "adamw":
            self.optimizer_G = optim.AdamW(self.model.parameters(), lr=self.lr,
                                           betas=[0.9, 0.999], weight_decay=self.optimizer.weight_decay)
        else:
            raise NotImplementedError("Optimizer {} is not implemented".format(self.optimizer.optim))

        # define lr schedulers
        if self.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - epoch / float(self.max_epochs + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif self.lr_policy == 'step':
            step_size = self.max_epochs // 3
            scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=step_size, gamma=0.1)
        else:
            return NotImplementedError(f'learning rate policy {self.lr_policy} is not implemented')

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = self.lr_scheduler
        optim_dict['monitor'] = self.monitor_name
        return optim_dict

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1, img_in2, _ = batch
        if self.difference_module == 'euclidean':
            self.siam_score = self.model(img_in1, img_in2)
        elif self.difference_module == 'learnable':
            self.output_diff = self.model(img_in1, img_in2)
            self.siam_score = torch.softmax(self.output_diff, dim=1)[:, 1]
        else:
            raise NotImplementedError('Only option 1 and 2 are supported')
        return self.siam_score

    def _backward_G(self):

        _, _, label = self.batch
        self.gt_classify = label.to(self.device).float()
        if self.difference_module == 'euclidean':
            self.G_loss = self.criterion(self.siam_score, self.gt_classify)

        elif self.difference_module == 'learnable':
            labels = self.gt_classify
            if self.loss_fn == 'ce':
                labels = labels.squeeze().long()
                self.G_loss = self.criterion(self.output_diff, labels.reshape(self.output_diff.shape[0]))
        return self.G_loss, self.siam_score

    def training_step(self, batch, batch_idx):
        """Training step."""
        img_in1, img_in2, label = batch
        batch_size = img_in1.shape[0]
        self.visualize_image(
            "compare_sample", img_in1,
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        if self.num_golden > 1:
            for i in range(img_in2.shape[1]):
                self.visualize_image(
                    f"golden_sample{i}", img_in2[:, i],
                    logging_frequency=self.tensorboard.infrequent_logging_frequency
                )
        else:
            self.visualize_image(
                "golden_sample", img_in2,
                logging_frequency=self.tensorboard.infrequent_logging_frequency
            )
        _ = self._forward_pass(batch)

        loss, siam_score = self._backward_G()
        self.train_metrics.update(siam_score, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        train_accuracy = self.train_metrics.compute()['total_accuracy'].item()
        train_false_positive_rate = self.train_metrics.compute()['false_alarm'].item()
        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss
        self.status_logging_dict["train_acc"] = train_accuracy
        self.status_logging_dict["train_fpr"] = train_false_positive_rate
        tensorboard_logging_dict = {
            "train_acc": train_accuracy,
            "train_fpr": train_false_positive_rate
        }
        self.visualize_histogram(
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        self.visualize_metrics(tensorboard_logging_dict)
        self.train_metrics.reset()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, _, label = batch
        batch_size = img.shape[0]
        _ = self._forward_pass(batch)
        loss, siam_score = self._backward_G()
        self.val_metrics.update(siam_score, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        val_accuracy = self.val_metrics.compute()['total_accuracy'].item()
        val_fpr = self.val_metrics.compute()['false_alarm'].item()
        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["val_acc"] = val_accuracy
            self.status_logging_dict["val_fpr"] = val_fpr
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
        validation_logging_dict = {
            "val_acc": val_accuracy,
            "val_fpr": val_fpr
        }
        self.visualize_metrics(validation_logging_dict)
        self.val_metrics.reset()
        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.margin = self.model_config["classify"]["eval_margin"]
        self.valid_metrics = AOIMetrics(self.margin)
        self.num_comp = 0

    def test_step(self, batch, batch_idx):
        """Test step"""
        siam_score = self._forward_pass(batch)
        self.valid_metrics.update(siam_score, batch[2])
        self.num_comp += len(siam_score)

    def on_test_epoch_end(self):
        """Test epoch end"""
        total_accuracy = self.valid_metrics.compute()['total_accuracy'].item()
        false_alarm = self.valid_metrics.compute()['false_alarm'].item()
        defect_accuracy = self.valid_metrics.compute()['defect_accuracy'].item()
        false_negative = self.valid_metrics.compute()['false_negative'].item()

        logging.info(
            "Tot Comp {} Total Accuracy {} False Negative {} False Alarm {} Defect Correctly Captured {} for Margin {}".format(
                self.num_comp,
                round(total_accuracy, 2),
                round(false_negative, 2),
                round(false_alarm, 2),
                round(defect_accuracy, 2),
                self.margin
            )
        )

        self.status_logging_dict = {}
        self.status_logging_dict["test_acc"] = total_accuracy
        self.status_logging_dict["test_fpr"] = false_alarm
        self.status_logging_dict["test_fnr"] = false_negative
        self.status_logging_dict["defect_acc"] = defect_accuracy
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        self.euclid = []

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference """
        siam_score = self._forward_pass(batch)
        if batch_idx == 0:
            self.euclid = siam_score
        else:
            self.euclid = torch.cat((self.euclid, siam_score), 0)

    def on_predict_epoch_end(self):
        """Predict epoch end"""
        # Gather results from all GPUs
        gathered_results = self.all_gather(self.euclid)

        # Single GPU case
        if len(gathered_results.shape) == 1:
            gathered_results = gathered_results.unsqueeze(dim=0)

        if self.trainer.is_global_zero:
            # Only the rank 0 process writes the file
            combined_results = torch.cat([tensor for tensor in gathered_results], dim=0)
            siamese_score = 'siamese_score'
            self.dm.df_infer[siamese_score] = combined_results.cpu().numpy()

            self.dm.df_infer.to_csv(
                os.path.join(self.experiment_spec.results_dir, "inference.csv"),
                header=True,
                index=False
            )
            logging.info("Completed")

        # Clear the list for the next epoch
        self.euclid = []

    def visualize_histogram(self, logging_frequency=2):
        """Visualize histograms of model parameters.

        Args:
            logging_frequency (int): The frequency at which to log the histograms.
        """
        if self.current_epoch % logging_frequency == 0 and self.tensorboard.enabled:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(f"histogram/{name}", params, self.current_epoch)

    def visualize_image(self, name, tensor, logging_frequency=10):
        """Visualize images during training.

        Args:
            name (str): The name for the image to be visualized.
            tensor (torch.Tensor): The input tensor containing the image.
            logging_frequency (int): The frequency at which to log the images.

        """
        logging_frequency_in_steps = self.dm.num_train_steps_per_epoch * logging_frequency
        is_log_step = self.global_step % logging_frequency_in_steps == 0
        if is_log_step and self.tensorboard.enabled:
            self.logger.experiment.add_images(
                f"image/{name}",
                tensor,
                global_step=self.global_step
            )

    def visualize_metrics(self, metrics_dict):
        """Visualize metrics from during train/val.

        Args:
            metrics_dict (dict): Dictionary of metric tensors to be visualized.

        Returns:
            No returns.
        """
        # Make sure the scalars are logged in TensorBoard always.
        if self.logger:
            self.logger.log_metrics(metrics_dict, step=self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "visual_changenet_classify"

        # Additional info to help with loading ViTAdapter.
        model_config = self.experiment_spec.model
        model_name = model_config.backbone['type']
        difference_module = model_config.classify.difference_module
        if "radio" in model_name and difference_module == "learnable":
            checkpoint["tao_model_type"] = "radio_learnable"
