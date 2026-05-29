# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for DepthNet. """

import os
import torch
import torch.nn.functional as F
from fairscale.optim import OSS
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

from nvidia_tao_pytorch.cv.depth_net.model.mono_depth import get_model_loss_class
from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.post_process import PostProcess
from nvidia_tao_pytorch.cv.depth_net.evaluation.mono_evaluator import MonoDepthEvaluator
from nvidia_tao_pytorch.cv.depth_net.utils.misc import save_inference_batch, vis_mono
from nvidia_tao_pytorch.cv.depth_net.model.lr_scheduler import build_lr_scheduler


class MonoDepthNetPlModel(TAOLightningModule):
    """PyTorch Lightning module for DepthNet monocular depth estimation model.

    Attributes:
        model_type (str): Type of depth model being used (relative or metric).
        align_gt (bool): Whether to align ground truth for relative depth estimation.
        max_depth (float): Maximum depth value for metric depth estimation.
        min_depth (float): Minimum depth value for metric depth estimation.
        model (nn.Module): The depth prediction model.
        criterion (nn.Module): Loss function for training.
        post_processors (PostProcess): Post-processing utilities for depth maps.
        val_evaluator (MonoDepthEvaluator): Validation metrics evaluator.
        test_evaluator (MonoDepthEvaluator): Test metrics evaluator.
    """

    def __init__(self, experiment_spec, export=False):
        """Initialize the MonoDepthNetPlModel.

        Args:
            experiment_spec (object): Experiment configuration specification containing
                model, dataset, and training parameters.
            export (bool, optional): Whether the model is being used for export.
                Defaults to False.
        """
        super().__init__(experiment_spec)

        # init the model
        self.checkpoint_filename = 'dn_model'

        self._build_model_criterion(export)
        self.post_processors = PostProcess()

    def _build_model_criterion(self, export=False):
        """Build the depth prediction model and loss criterion.
        Args:
            export (bool, optional): Whether the model is being used for export.
                Defaults to False.
        Note:
            - For relative depth models: Sets align_gt=True and clears depth bounds
            - For metric depth models: Sets align_gt=False and uses configured depth bounds
        """
        self.model_type = self.model_config["model_type"]
        if "relative" in (self.model_type).lower():
            self.align_gt = True
            self.max_depth = None
            self.min_depth = None
        else:
            self.align_gt = False
            self.max_depth = self.dataset_config["max_depth"]
            self.min_depth = self.dataset_config["min_depth"]

        ModelClass, LossClass = get_model_loss_class(self.model_type)
        self.model = ModelClass(self.model_config, self.max_depth, export=export)
        self.criterion = LossClass()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for training.

        Returns:
            dict: Dictionary containing optimizer and scheduler configuration.
                Keys include 'optimizer', 'lr_scheduler', and 'monitor'.

        Raises:
            NotImplementedError: If the specified optimizer is not supported.

        Note:
            - Pretrained parameters use base learning rate
            - Non-pretrained parameters use 10x base learning rate
            - Supports FSDP (Fully Sharded Data Parallel) strategy
        """
        self.train_config = self.experiment_spec.train
        self.lr = self.train_config.optim.lr

        # increase lr for DTP Head
        param_dicts = [
            {
                "params": [param for name, param in self.model.named_parameters() if 'pretrained' in name],
                "lr": self.lr,
            },
            {
                "params": [param for name, param in self.model.named_parameters() if 'pretrained' not in name],
                "lr": self.lr * 10,
            }
        ]

        if self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(
                params=param_dicts,
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=self.train_config.optim.weight_decay  # 1e-4
            )
        else:
            raise NotImplementedError(f"Optimizer {self.train_config.optim.optimizer} is not implemented")

        if self.train_config.distributed_strategy == "fsdp":
            # Override force_broadcast_object=False in PTL
            optim = OSS(params=base_optimizer.param_groups, optim=type(base_optimizer), force_broadcast_object=True, **base_optimizer.defaults)
        else:
            optim = base_optimizer

        optim_dict = {}
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config['optim']['lr_scheduler']
        lr_scheduler = build_lr_scheduler(optim, scheduler_type, self.train_config, self.trainer)

        optim_dict["lr_scheduler"] = {'scheduler': lr_scheduler, 'interval': 'step'}
        optim_dict['monitor'] = self.train_config['optim']['monitor_name']
        return optim_dict

    def on_train_start(self):
        """
        This method is called at the beginning of training. It sets up the datamodule
        and initializes training augmentation configuration.
        """
        self.trainer.datamodule.resume_step = self.trainer.global_step
        self.trainer.datamodule.setup('fit')
        self.train_aug_config = self.dataset_config["train_dataset"]["augmentation"]

    def training_step(self, batch, batch_idx):
        """
        This method handles the forward pass, loss computation, logging, and
        visualization for a single training batch.

        Args:
            batch (dict): Training batch containing:
                - image (torch.Tensor): Input images of shape (B, C, H, W)
                - disparity/depth (torch.Tensor): Ground truth depth/disparity
                - valid_mask (torch.Tensor): Valid pixel mask
                - image_path (list): Image file paths
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed training loss.

        Note:
            - Supports both disparity and depth ground truth
            - Logs training loss, learning rates, and step count
            - Generates visualization images at specified intervals
        """
        image1 = batch['image']
        if 'disparity' in batch:
            disp_gt = batch['disparity']
        else:
            disp_gt = batch['depth']
        valid = batch['valid_mask']
        batch_size = len(image1)

        # disp_pred (B, W', H'), dist_gt (B, 1, H, W), valid (B, H, W)
        disp_pred = self.model(image1.contiguous())
        losses = self.criterion(disp_pred, disp_gt.squeeze(1), (valid == 1))

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        lrs = [param_group['lr'] for param_group in self.optimizers().optimizer.param_groups]
        self.log("train/lr", lrs[0], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/lr1", lrs[1], on_step=True, on_epoch=False, prog_bar=False)
        self.log("steps", self.trainer.global_step, on_step=True, on_epoch=False, prog_bar=False)

        if self.trainer.global_step % self.train_config['vis_step_interval'] == 0 and self.train_config['dataloader_visualize']:
            image1 = F.interpolate(image1, disp_gt.shape[-2:], mode='bilinear', align_corners=True)
            canvas, canvas_caption = vis_mono(image1, pred_disp=disp_pred.detach(), disp_gt=disp_gt.detach()[:, 0],
                                              aug_config=self.train_aug_config, image_path=batch['image_path'],
                                              normalize_depth=self.dataset_config["normalize_depth"])

            if len(self.loggers) > 1:
                self.loggers[1].log_image(key="train/vis", images=canvas, step=self.trainer.global_step, caption=canvas_caption)

        return losses

    def on_train_epoch_end(self):
        """
        This method logs training metrics to the status logging system and
        updates the status logger with the average training loss for the epoch.
        """
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """
        This method initializes the validation evaluator and sets up validation
        augmentation configuration for the current epoch.
        """
        self.val_aug_config = self.dataset_config["val_dataset"]["augmentation"]
        self.val_evaluator = MonoDepthEvaluator(align_gt=self.align_gt,
                                                min_depth=self.min_depth,
                                                max_depth=self.max_depth,
                                                sync_on_compute=False).to(self.device)

    def validation_step(self, batch, batch_idx):
        """
        This method handles the forward pass, loss computation, evaluation metrics
        update, and visualization for a single validation batch.

        Args:
            batch (dict): Validation batch containing:
                - image (torch.Tensor): Input images of shape (B, C, H, W)
                - disparity/depth (torch.Tensor): Ground truth depth/disparity
                - valid_mask (torch.Tensor): Valid pixel mask
                - image_path (list): Image file paths
                - image_size (list): Original image sizes
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed validation loss.

        Note:
            - Performs post-processing on predictions
            - Updates validation metrics evaluator
            - Generates visualization images at specified intervals
        """
        image1 = batch['image']
        if 'disparity' in batch:
            disp_gt = batch['disparity']
        else:
            disp_gt = batch['depth']

        image_names = batch['image_path']
        valid = batch['valid_mask']
        image_size = batch['image_size']

        batch_size = len(image1)
        # disp_pred (B, W', H'), dist_gt (B, 1, H, W), valid (B, H, W)
        disp_pred = self.model(image1.contiguous())
        post_processed_results = self.post_processors(image1, disp_pred, image_size, valid,
                                                      resized_size=None, gt_depth=disp_gt, image_names=image_names)
        self.val_evaluator.update(post_processed_results=post_processed_results)

        # disp_pred (B, 1, W, H), dist_gt (B, 1, H, W)
        disp_pred = F.interpolate(disp_pred[:, None], disp_gt.shape[-2:], mode='bilinear', align_corners=True)

        # valid (B, 1, H, W)
        valid = valid.unsqueeze(1)
        valid_mask = (valid == 1)
        losses = self.criterion(disp_pred.squeeze(1), disp_gt.squeeze(1), valid_mask.squeeze(1))

        if self.trainer.global_step % self.train_config['vis_step_interval'] == 0:
            image1 = F.interpolate(image1, disp_gt.shape[-2:], mode='bilinear', align_corners=True)
            canvas, canvas_caption = vis_mono(image1, pred_disp=disp_pred[:, 0], disp_gt=disp_gt[:, 0],
                                              aug_config=self.val_aug_config, image_path=batch['image_path'],
                                              normalize_depth=self.dataset_config["normalize_depth"])
            if len(self.loggers) > 1:
                self.loggers[1].log_image(key="val/vis", images=canvas, step=self.trainer.global_step, caption=canvas_caption)

        self.log("val/loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return losses

    def on_validation_epoch_end(self):
        """
        This method computes and logs validation metrics, updates the status logger
        with validation results, and handles distributed training synchronization.
        """
        average_val_loss = self.trainer.logged_metrics["val/loss"].item()
        results_metric = self.val_evaluator.compute()
        for name, metric in results_metric.items():
            self.log(f'val/{name}', metric, sync_dist=True)

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val/loss"] = average_val_loss
            for name, metric in results_metric.items():
                self.status_logging_dict[f"val/{name}"] = metric
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    def predict_step(self, batch, batch_idx):
        """
        This method handles the forward pass and post-processing for inference
        on a single batch of images.

        Args:
            batch (dict): Inference batch containing:
                - image (torch.Tensor): Input images of shape (B, C, H, W)
                - disparity/depth (torch.Tensor, optional): Ground truth depth/disparity
                - valid_mask (torch.Tensor): Valid pixel mask
                - image_path (list): Image file paths
                - image_size (list): Original image sizes
                - resized_size (list): Resized image sizes
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Post-processed inference results containing depth predictions
                and evaluation metrics.

        Note:
            - Ground truth is optional for pure inference
            - Performs post-processing to match original image dimensions
        """
        image1 = batch['image']
        if 'disparity' in batch:
            disp_gt = batch['disparity']
        elif 'depth' in batch:
            disp_gt = batch['depth']
        else:
            disp_gt = None

        image_names = batch['image_path']
        valid_mask = batch['valid_mask']

        disp_pred = self.model(image1.contiguous())
        # disp_pred (B, W', H'), dist_gt (B, 1, H, W)

        image_size = batch['image_size']
        resized_size = batch['resized_size']
        image_size = torch.stack(image_size, dim=0)
        resized_size = torch.stack(resized_size, dim=0)
        post_processed_results = self.post_processors(image1, disp_pred, image_size, valid_mask, resized_size=resized_size, gt_depth=disp_gt, image_names=image_names)

        return post_processed_results

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        This method saves inference results and visualization images to disk
        after processing each prediction batch.

        Args:
            outputs (dict): Post-processed inference results from predict_step.
            batch (dict): Original input batch.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Note:
            - Creates output directory for inference images
            - Saves visualization images with proper augmentation
        """
        output_dir = os.path.join(self.experiment_spec.results_dir, 'inference_images')
        os.makedirs(output_dir, exist_ok=True)
        self.infer_aug_config = self.dataset_config["infer_dataset"]["augmentation"]
        save_inference_batch(outputs, output_dir, aug_config=self.infer_aug_config,
                             normalize_depth=self.dataset_config["normalize_depth"],
                             save_raw_pfm=self.experiment_spec["inference"]["save_raw_pfm"])

    def on_test_epoch_start(self) -> None:
        """
        This method initializes the test evaluator and sets up test augmentation
        configuration for the current epoch.
        """
        self.test_aug_config = self.dataset_config["test_dataset"]["augmentation"]
        self.test_evaluator = MonoDepthEvaluator(align_gt=self.align_gt,
                                                 min_depth=self.min_depth,
                                                 max_depth=self.max_depth,
                                                 sync_on_compute=False).to(self.device)

    def test_step(self, batch, batch_idx):
        """
        This method handles the forward pass, post-processing, and evaluation metrics
        update for a single test batch.

        Args:
            batch (dict): Test batch containing:
                - image (torch.Tensor): Input images of shape (B, C, H, W)
                - disparity/depth (torch.Tensor, optional): Ground truth depth/disparity
                - valid_mask (torch.Tensor): Valid pixel mask
                - image_path (list): Image file paths
                - image_size (list): Original image sizes
                - resized_size (list): Resized image sizes
            batch_idx (int): Index of the current batch.

        Note:
            - Performs post-processing on predictions
            - Updates test metrics evaluator
            - Ground truth is optional for test evaluation
        """
        image1 = batch['image']
        if 'disparity' in batch:
            disp_gt = batch['disparity']
        elif 'depth' in batch:
            disp_gt = batch['depth']
        else:
            disp_gt = None

        image_names = batch['image_path']
        valid_mask = batch['valid_mask']

        disp_pred = self.model(image1.contiguous())
        # disp_pred (B, W', H'), dist_gt (B, 1, H, W)

        image_size = batch['image_size']
        resized_size = batch['resized_size']
        image_size = torch.stack(image_size, dim=0)
        resized_size = torch.stack(resized_size, dim=0)
        post_processed_results = self.post_processors(image1, disp_pred, image_size, valid_mask, resized_size=resized_size, gt_depth=disp_gt, image_names=image_names)

        self.test_evaluator.update(post_processed_results=post_processed_results)

    def on_test_epoch_end(self):
        """
        This method computes and logs test metrics, updates the status logger
        with test results, and handles distributed training synchronization.
        """
        results_metric = self.test_evaluator.compute()
        for name, metric in results_metric.items():
            self.log(f'test/{name}', metric, sync_dist=True)

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            for name, metric in results_metric.items():
                self.status_logging_dict[f"val/{name}"] = metric
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    def forward(self, x):
        """Forward pass through the depth prediction model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Model predictions of shape (B, H', W').
        """
        outputs = self.model(x.contiguous())
        return outputs
