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

"""Visual ChangeNet Segmentation Model PyTorch Lightning Module"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.utils.metric_tool import ConfuseMatrixMeter
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.utils.losses import cross_entropy, mmIoULoss
# from nvidia_tao_pytorch.cv.visual_changenet.models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.utils.utils_vis import de_norm, get_color_mapping
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet import build_model
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.utils import utils_vis as utils
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
# import nvidia_tao_pytorch.cv.visual_changenet.segmentation.utils.losses as losses
from nvidia_tao_pytorch.core.path_utils import expand_path


# pylint:disable=too-many-ancestors
class ChangeNetPlModel(TAOLightningModule):
    """ PTL module for DINO Object Detection Model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for DINO Model."""
        super().__init__(experiment_spec)
        # Overriding what's done in super()
        self.dataset_config = self.experiment_spec.dataset.segment
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference

        # init the model
        self._build_model(export)
        self._build_criterion()

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.n_class = self.dataset_config.num_classes
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        self.batch_size = self.dataset_config.batch_size

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.checkpoint_dir = self.vis_dir = self.experiment_spec.results_dir

        self.optimizer_G = None

        self.shuffle_AB = False  # args.shuffle_AB

        # # define the loss functions
        self.multi_scale_train = self.dataset_config.multi_scale_train
        self.multi_scale_infer = self.dataset_config.multi_scale_infer
        self.weights = tuple(self.train_config.segment.weights)
        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches
        self.color_map = get_color_mapping(dataset_name=self.dataset_config.data_name,
                                           color_mapping_custom=self.dataset_config.color_map,
                                           num_classes=self.n_class)

        self.checkpoint_filename = 'changenet_model_segment'

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        assert self.train_config.segment["loss"] in ['ce', 'mmiou'], "Visual ChangeNet Segmentation pipeline currently only supports ['ce', 'mmiou'] loss functions."
        if self.train_config.segment["loss"] == 'ce':
            self._pxl_loss = cross_entropy
        # elif self.train_config.segment["loss"] == 'bce':
        #     self._pxl_loss = losses.binary_ce
        # elif self.train_config.segment["loss"] == 'fl':
        #     print('\n Calculating alpha in Focal-Loss (FL) ...')
        #     alpha           = get_alpha(dataloaders['train']) # calculare class occurences #TODO: check this
        #     print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
        #     self._pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)
        # elif self.train_config.segment["loss"] == "miou":
        #     print('\n Calculating Class occurances in training set...')
        #     alpha   = np.asarray(get_alpha(dataloaders['train'])) # calculare class occurences
        #     alpha   = alpha/np.sum(alpha)
        #     # weights = torch.tensor([1.0, 1.0]).cuda()
        #     weights = 1-torch.from_numpy(alpha).cuda()
        #     print(f"Weights = {weights}")
        #     self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=self.n_class).cuda()
        elif self.train_config.segment["loss"] == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=self.n_class).cuda()
        else:
            raise NotImplementedError(self.train_config.segment["loss"])

        self.criterion = self._pxl_loss

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
            # args.lr_decay_iters
            scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=step_size, gamma=0.1)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.lr_policy)

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = self.lr_scheduler
        optim_dict['monitor'] = self.monitor_name
        return optim_dict

    def _update_metric(self):
        """
        update metric
        Calculates running metrics for train, validate and test.
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        _ = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())  # current_score

    def _visualize_pred(self, i):
        """Helper function to visualize predictions for binary change detection (LEVIR-CD)"""
        pred = torch.argmax(self.G_final_pred[i].unsqueeze(0), dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _visualize_pred_multi(self, i):
        """Helper function to visualize predictions for multi class change detection (landSCD)"""
        pred = torch.argmax(self.G_final_pred[i].unsqueeze(0), dim=1, keepdim=True)
        return pred

    def _visualize_predictions(self, batch_idx=-1, vis_afer_n_batches=100):
        """
        Visualizes two input images along with GT and prediction of
        segmentation change map as a linear grid of images.
        Used to only visualize predictions during evaluation
        input:
            flag:
                    after every 100 images, not used during validation
        """
        if np.mod(batch_idx, vis_afer_n_batches) == 0:
            for i in range(len(self.batch['A'])):
                vis_input = utils.make_numpy_grid(de_norm(self.batch['A'][i].unsqueeze(0)))
                vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B'][i].unsqueeze(0)))

                if self.n_class > 2:
                    # print("Visualising multiple classes")
                    vis_pred, _ = utils.make_numpy_grid(self._visualize_pred_multi(i),
                                                        num_class=self.n_class,
                                                        gt=self.batch['L'][i].unsqueeze(0),
                                                        color_map=self.color_map)
                    vis_gt = utils.make_numpy_grid(self.batch['L'][i].unsqueeze(0),
                                                   num_class=self.n_class,
                                                   color_map=self.color_map)
                else:
                    vis_pred = utils.make_numpy_grid(self._visualize_pred(i))
                    vis_gt = utils.make_numpy_grid(self.batch['L'][i].unsqueeze(0))

                # Combining horizontally
                line_width = 10  # width of the black line in pixels
                line = np.zeros((vis_input.shape[0], line_width, 3), dtype=np.uint8)  # create a black line

                # Combine predictions in the order of Image A, Image B, Prediction, GT.
                vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred, line, vis_gt], axis=1)

                vis = np.clip(vis, a_min=0.0, a_max=1.0)

                # Save combined visualisation in a different folder.
                vis_combined_dir = os.path.join(self.vis_dir, "combined_visualisation")
                if not os.path.exists(vis_combined_dir):
                    os.makedirs(vis_combined_dir)
                file_name = expand_path(os.path.join(
                    vis_combined_dir, self.batch['name'][i] + '.jpg'))
                plt.imsave(file_name, vis)

                # Dump Predictions only
                vis_pred_only = np.clip(vis_pred, a_min=0.0, a_max=1.0)
                file_name = expand_path(os.path.join(
                    self.vis_dir, self.batch['name'][i] + '.jpg'))
                plt.imsave(file_name, vis_pred_only)

    def _visualize_infer_output(self, batch_idx, vis_afer_n_batches=1):
        """
        Visualizes two input images along with segmentation change map prediction
        as a linear grid of images. Does not include GT segmentation change map
        during inference.
        """
        if np.mod(batch_idx, vis_afer_n_batches) == 0:
            for i in range(len(self.batch['A'])):
                vis_input = utils.make_numpy_grid(de_norm(self.batch['A'][i].unsqueeze(0)))
                vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B'][i].unsqueeze(0)))

                if self.n_class > 2:
                    # print("Visualising multiple classes")
                    vis_pred = utils.make_numpy_grid(self._visualize_pred_multi(i),
                                                     num_class=self.n_class, color_map=self.color_map)
                else:
                    vis_pred = utils.make_numpy_grid(self._visualize_pred(i))

                # Combining horizontally
                line_width = 10  # width of the black line in pixels
                line = np.zeros((vis_input.shape[0], line_width, 3), dtype=np.uint8)  # create a black line
                vis = np.concatenate([vis_input, line, vis_input2, line, vis_pred], axis=1)

                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                # Save combined visualisation in a different folder.
                vis_combined_dir = os.path.join(self.vis_dir, "combined_visualisation")
                if not os.path.exists(vis_combined_dir):
                    os.makedirs(vis_combined_dir)
                file_name = expand_path(os.path.join(
                    vis_combined_dir, self.batch['name'][i] + '.jpg'))
                plt.imsave(file_name, vis)

                # Dump Predictions only
                vis_pred_only = np.clip(vis_pred, a_min=0.0, a_max=1.0)
                file_name = expand_path(os.path.join(
                    self.vis_dir, self.batch['name'][i] + '.jpg'))
                plt.imsave(file_name, vis_pred_only)

    def _clear_cache(self):
        self.running_metric.clear()

    def _collect_epoch_states(self):
        """Collect evaluation metrics for each epoch (train, val, test)"""
        scores, mean_score_dict = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        # message = 'Scores per class'
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # message = 'Mean scores for all classes: '
        self.log_dict(mean_score_dict, on_step=False, on_epoch=True, sync_dist=True)
        return scores, mean_score_dict

    def _forward_pass(self, batch):
        """
        Perform a forward pass on the model using the given batch of data.

        Args:
            batch (dict): A dictionary containing input images 'A' and 'B'.

        Returns:
            torch.Tensor: The final predicted segmentation map.
        """
        self.batch = batch
        img_in1 = batch['A']
        img_in2 = batch['B']
        self.G_pred = self.model(img_in1, img_in2)

        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size())
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred = self.G_final_pred + F.interpolate(pred, size=self.G_pred[-1].size(2), mode="nearest")
                else:
                    self.G_final_pred = self.G_final_pred + pred
            self.G_final_pred = self.G_final_pred / len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]
        return self.G_final_pred

    def _backward_G(self):
        """
        Perform the backward pass and calculate the loss for the Generator.

        Returns:
            torch.Tensor: The loss value for the Generator.
        """
        gt = self.batch['L'].float()
        if self.multi_scale_train == "True":
            i = 0
            temp_loss = 0.0
            for pred in self.G_pred:
                if pred.size(2) != gt.size(2):
                    temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest"))
                else:
                    temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, gt)
                i += 1
            self.G_loss = temp_loss
        else:
            self.G_loss = self._pxl_loss(self.G_pred[-1], gt)

        return self.G_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        batch_size = batch['A'].shape[0]
        _ = self._forward_pass(batch)
        loss = self._backward_G()

        self._update_metric()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """Validation epoch start."""
        if not self.trainer.sanity_checking:
            # FLUSHING TRAINING EPOCH METRICS
            _, _ = self._collect_epoch_states()  # logs all evaluation metrics
        self._clear_cache()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        batch_size = batch['A'].shape[0]

        _ = self._forward_pass(batch)
        loss = self._backward_G()
        self._update_metric()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        # FLUSHING VALIDATION EPOCH METRICS
        scores, mean_scores = self._collect_epoch_states()  # logs all evaluation metrics
        self.log("val_acc", scores['acc'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._clear_cache()

        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["val_acc"] = scores['acc']
            self.status_logging_dict["val_miou"] = scores['miou']
            self.status_logging_dict["val_mf1"] = scores['mf1']
            self.status_logging_dict["val_mprecision"] = mean_scores['mprecision']
            self.status_logging_dict["val_mrecall"] = mean_scores['mrecall']
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self) -> None:
        """ Test epoch start."""
        self._clear_cache()

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate """
        _ = self._forward_pass(batch)
        loss = self._backward_G()

        # Calculate running metrics
        self._update_metric()
        self._visualize_predictions(batch_idx, vis_afer_n_batches=self.vis_after_n_batches)

        self.log("test_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    def on_test_epoch_end(self):
        """Test epoch end."""
        scores, mean_scores = self._collect_epoch_states()  # needed for update metrics

        self.status_logging_dict = {}
        self.status_logging_dict["test_acc"] = scores['acc']
        self.status_logging_dict["test_miou"] = scores['miou']
        self.status_logging_dict["test_mf1"] = scores['mf1']
        self.status_logging_dict["test_mprecision"] = mean_scores['mprecision']
        self.status_logging_dict["test_mrecall"] = mean_scores['mrecall']
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference """
        outputs = self._forward_pass(batch)
        self._visualize_infer_output(batch_idx, vis_afer_n_batches=self.vis_after_n_batches_infer)

        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "visual_changenet_segment"

        # Additional info to help with loading ViTAdapter.
        model_config = self.experiment_spec.model
        model_name = model_config.backbone['type']
        if "radio" in model_name:
            checkpoint["tao_model_type"] = "radio"
