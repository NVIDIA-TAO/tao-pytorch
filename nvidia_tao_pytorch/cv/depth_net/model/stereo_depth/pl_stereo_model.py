# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for Stereo (FoundationStereo) Depthnet."""

import os

import matplotlib.cm as cm
import cv2
from fairscale.optim import OSS
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth import StereoDepthNet
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.loss import SequenceLoss
from nvidia_tao_pytorch.cv.depth_net.evaluation.stereo_evaluator import StereoDepthEvaluator
from nvidia_tao_pytorch.cv.depth_net.utils.frame_utils import write_pfm
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import InputPadder
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import write_image, unnormalize
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import get_filename_from_path, get_dataset_index
from nvidia_tao_pytorch.cv.depth_net.model.lr_scheduler import build_lr_scheduler


class StereoDepthNetPlModel(TAOLightningModule):
    """
    Lightning module for the StereoDepthNet model.

    This class encapsulates the training, validation, testing, and prediction
    logic for a stereo depth estimation neural network, built upon the
    TAOLightningModule. It handles model initialization, optimizer
    configuration, loss calculation, logging, and metric computation.
    """

    def __init__(self, experiment_spec, export=False):
        """
        Initializes the StereoDepthNetPlModel.

        Args:
            experiment_spec (omegaconf.dictconfig.DictConfig): A dictionary-like
                object containing the experiment configuration, including
                model, dataset, and training parameters.
            export (bool): A flag indicating whether the model is being
                initialized for export. Defaults to False.
        """
        super().__init__(experiment_spec)
        # init the model
        self.checkpoint_filename = "dn_model"
        self.min_depth = self.dataset_config["min_depth"]
        self.max_disparity = self.dataset_config["max_disparity"]
        self.model_type = self.model_config["model_type"]
        self._build_model_criterion(export)
        self.criterion = SequenceLoss(max_disparity=self.max_disparity)
        self.vis_step_interval = self.experiment_spec.train["vis_step_interval"]
        self.count = 0
        self.is_valid_disparity_gt = True

    def _build_model_criterion(self, export=False):
        """
        Internal function to build the stereo depth estimation model.

        This method retrieves the model class based on the 'model_type'
        specified in the model configuration and initializes the model.
        """
        self.model_class, self.loss_class = StereoDepthNet.get_model()[self.model_type.lower()]
        self.model = self.model_class(self.model_config, export=export)
        self._build_criterion()

    def _build_criterion(self):
        """
        Internal function to build the loss criterion.

        This method retrieves the loss class based on the 'loss_type'
        specified in the model configuration and initializes the criterion.
        """
        self.criterion = self.loss_class(max_disparity=self.max_disparity)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for training.

        This method sets up the optimizer (AdamW or SGD) and optionally
        a learning rate scheduler based on the training configuration
        provided in `experiment_spec`. It supports FSDP for distributed training.

        Returns:
            dict: A dictionary containing the optimizer and, if configured,
                  the learning rate scheduler.
        Raises:
            NotImplementedError: If an unsupported optimizer type is specified.
        """
        self.train_config = self.experiment_spec.train
        param_dicts = [
            {
                "params": [param for param in self.model.named_parameters()],
                "lr": self.train_config.optim.lr,
                "weight_decay": self.train_config.optim.weight_decay
            }]

        if self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(
                params=param_dicts, lr=self.train_config.optim.lr
            )
        elif self.train_config.optim.optimizer == 'SGD':
            base_optimizer = torch.optim.SGD(
                params=param_dicts, lr=self.train_config.optim.lr, momentum=0.9
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.train_config.optim.optimizer} is not implemented"
            )

        if self.train_config.distributed_strategy == "fsdp":
            # Override force_broadcast_object=False in PTL
            optim = OSS(
                params=base_optimizer.param_groups,
                optim=type(base_optimizer),
                force_broadcast_object=True,
                **base_optimizer.defaults,
            )
        else:
            optim = base_optimizer

        optim_dict = {}
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config['optim']['lr_scheduler']

        lr_scheduler = build_lr_scheduler(
            optim, scheduler_type, self.train_config, self.trainer)

        if scheduler_type is not None:
            optim_dict["lr_scheduler"] = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                "frequency": 1,
            }
        optim_dict['monitor'] = self.train_config['optim']['monitor_name']
        return optim_dict

    def on_train_start(self):
        """
        Hook called at the beginning of training.

        Resumes the data module setup and prepares for training.
        """
        self.trainer.datamodule.resume_step = self.trainer.global_step
        self.trainer.datamodule.setup('fit')
        self.train_aug_config = self.dataset_config["train_dataset"]["augmentation"]

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): A dictionary containing the input data for the batch,
                          including 'image', 'disparity', 'valid_mask', and 'right_image'.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the current training step.
        """
        image1 = batch['image']
        disp_gt = batch['disparity']
        valid = batch['valid_mask']
        batch_size = image1.shape[0]
        image2 = batch['right_image']
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        disp_pred, disp_preds = self.model(
            image1, image2, iters=self.model_config['train_iters'], test_mode=False
        )
        disp_pred = padder.unpad(disp_pred.cuda().float())
        # loss
        losses, _ = self.criterion(disp_preds, disp_pred, disp_gt, valid)

        self.log(
            "train_loss",
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        lrs = [param_group["lr"] for param_group in self.optimizers().optimizer.param_groups]
        self.log("lr", lrs[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("global_step", self.trainer.global_step, on_step=True, on_epoch=False, prog_bar=True)

        if (batch_idx % self.vis_step_interval) == 0:
            concat_images, list_images = self.build_disparity_viz(
                (image1, disp_gt, disp_preds, self.trainer.global_step)
            )
            if len(self.loggers) > 1:
                self.loggers[1].log_image(key="train/vis", images=list_images, step=self.global_step)
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            tb_logger.add_image('train/images', concat_images, batch_idx, dataformats='HWC')
        return losses

    def on_train_epoch_end(self):
        """
        Hook called at the end of each training epoch.

        Logs the average training loss to `status.json`.
        """
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.", status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """
        Hook called at the beginning of each validation epoch.

        Resets the depth metric evaluator for accurate epoch-level evaluation.
        """
        self.val_aug_config = self.dataset_config["val_dataset"]["augmentation"]
        self.val_evaluator = StereoDepthEvaluator(sync_on_compute=False,
                                                  max_disparity=self.dataset_config.max_disparity).to(self.device)

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (dict): A dictionary containing the input data for the batch,
                          including 'image', 'disparity', 'valid_mask', and 'right_image'.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated loss for the current validation step.
        """
        image1 = batch['image']
        disp_gt = batch['disparity']

        valid = batch['valid_mask']
        batch_size = image1.shape[0]
        valid_mask = valid
        image2 = batch['right_image']
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        disp_pred = self.model(
            image1, image2, iters=self.model_config['valid_iters'], test_mode=True
        )
        disp_pred = padder.unpad(disp_pred.cuda().float())
        losses, _ = self.criterion([disp_pred], disp_pred, disp_gt, valid_mask, is_interpolated=False)

        self.log(
            "val_loss",
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        lrs = [param_group["lr"] for param_group in self.optimizers().optimizer.param_groups]
        self.log("val/lr", lrs[0], on_step=True, on_epoch=False, prog_bar=True)

        concat_images, list_images = self.build_disparity_viz(
            (image1, disp_gt, [disp_pred], self.trainer.global_step)
        )
        if len(self.loggers) > 1:
            self.loggers[1].log_image(key="val/vis", images=list_images, step=self.global_step)
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        tb_logger.add_image('val/images', concat_images, batch_idx, dataformats='HWC')
        return losses

    def on_validation_epoch_end(self):
        """
        Hook called at the end of each validation epoch.

        Computes and logs validation metrics, and updates `status.json`.
        """
        average_val_loss = self.trainer.logged_metrics["val_loss"].item()
        results_metric = self.val_evaluator.compute()
        for name, metric in results_metric.items():
            self.log(f"val/{name}", metric, sync_dist=True)

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.", status_level=status_logging.Status.RUNNING
            )

    def build_disparity_viz(self, viz_batch) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Generates and prepares images for logging to Weights & Biases or TensorBoard.

        Args:
            viz_batch (tuple): A tuple containing the image, ground truth disparity,
                               predicted disparities, and global step.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: A tuple containing a concatenated
                                                 image for display and a list of
                                                 individual images.
        """

        def process_disp_gt(disp_gt, image_shape):
            """
            """
            if not self.is_valid_disparity_gt:
                return None
            disp_gt = F.interpolate(disp_gt, image_shape, mode="nearest")
            invalid = disp_gt == torch.inf
            disp_gt[invalid] = 0
            disp_gt[invalid] = 0
            return disp_gt

        def process_predicted_disparity(disp_pred, gt):
            """
            """
            if isinstance(disp_pred, list):
                disp_pred_last = disp_pred[-1][0].squeeze().detach().cpu().numpy()
            else:
                disp_pred_last = disp_pred[0].squeeze().detach().cpu().numpy()

            if gt is None:
                disp_pred_last = (cm.turbo(disp_pred_last / disp_pred_last.max()) * 255.0).astype(np.uint8)
                return disp_pred_last, None, None

            disp_pred_diff = torch.abs(gt.detach().cpu() - disp_pred_last)[0].squeeze()
            disp_pred_last = (cm.turbo(disp_pred_last / disp_pred_last.max()) * 255.0).astype(np.uint8)
            disp_pred_diff = (cm.turbo(disp_pred_diff / disp_pred_diff.max()) * 255.0).astype(np.uint8)
            gt_single_batch = gt[0].squeeze().detach().cpu()
            gt_single_batch = (cm.turbo(gt_single_batch / gt_single_batch.max()) * 255.0).astype(np.uint8)

            return disp_pred_last, disp_pred_diff, gt_single_batch

        # Log the images (Give them different names)
        (image, disp_gt, disp_pred, _) = viz_batch
        if isinstance(disp_pred, list):
            image_shape = disp_pred[-1].shape[-2:]
        else:
            image_shape = disp_pred.shape[-2:]
        # Resize the images accordingly. During training, the sizes are 1/4
        # the original size

        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        if (disp_gt is not None) and len(disp_gt.shape) < 4:
            disp_gt = disp_gt.unsqueeze(0)

        image = F.interpolate(image, image_shape, mode="bilinear")
        disp_gt = process_disp_gt(disp_gt, image_shape)
        disp_pred_last, disp_pred_diff, gt_single_batch = process_predicted_disparity(disp_pred, disp_gt)
        image = np.array(unnormalize(image[0].detach().cpu()) * 255.0).astype(np.uint8)
        image_transposed = np.transpose(np.array(image), (1, 2, 0))
        image_transposed = cv2.cvtColor(image_transposed, cv2.COLOR_BGR2RGB)

        if (disp_pred_diff is None) or (gt_single_batch is None):
            images = [image_transposed,
                      disp_pred_last[..., 0:3]]
        else:
            images = [image_transposed,
                      gt_single_batch[..., 0:3],
                      disp_pred_last[..., 0:3],
                      disp_pred_diff[..., 0:3]]

        concat_images = np.concatenate(images, axis=1)
        return concat_images.astype(np.uint8), images

    def on_predict_epoch_start(self):
        """
        Hook called at the beginning of each prediction epoch.

        Initializes the depth metric evaluator and a dictionary to
        collect prediction results.
        """
        self.pred_evaluator = StereoDepthEvaluator(
            sync_on_compute=False,
            max_disparity=self.dataset_config.max_disparity).to(
                self.device)
        self.collate_results = {}

    def predict_step(self, batch, batch_idx):
        """
        Performs a single prediction step.

        Processes an input batch to generate disparity predictions,
        updates evaluation metrics, and prepares results for saving.

        Args:
            batch (dict): A dictionary containing the input data for the batch,
                          including 'image', 'image_path', 'disparity', and 'right_image'.
            batch_idx (int): The index of the current batch.
        """
        image1 = batch["image"]
        image_names = batch["image_path"]
        disp_gt = batch["disparity"]
        if torch.sum(disp_gt) < 0:
            self.is_valid_disparity_gt = False
        batch_size = image1.shape[0]
        image2 = batch["right_image"]
        padder = InputPadder(image1.shape, divis_by=32)
        pad_image1, pad_image2 = padder.pad(image1, image2)
        disp_pred = self.model(
            pad_image1, pad_image2, iters=self.model_config["valid_iters"], test_mode=True
        )
        disp_pred = padder.unpad(disp_pred.cuda().float())
        pred_results = []
        invalid = disp_gt == torch.inf
        disp_pred[invalid] = 0
        self.pred_evaluator.update(disp_pred, disp_gt)
        for i in range(batch_size):
            viz_batch = (image1[i], disp_gt[i], disp_pred[i], batch_idx)
            concat_images, _ = self.build_disparity_viz(viz_batch)
            index = get_dataset_index(image_names[i], self.dataset_config["infer_dataset"])

            file_name = get_filename_from_path(
                image_names[i],
                self.dataset_config["infer_dataset"]["data_sources"][index]['dataset_name'])

            # Emit raw .pfm when save_raw_pfm=True (mirrors test_step).
            if self.experiment_spec["inference"]["save_raw_pfm"]:
                pfm_name = '.'.join(file_name.split('.')[:-1]) + '.pfm'
                target_pfm = os.path.join(self.experiment_spec.results_dir, pfm_name)
                os.makedirs(os.path.dirname(target_pfm), exist_ok=True)
                write_pfm(target_pfm, disp_pred[i].squeeze().detach().cpu().numpy())

            pred_results.append({"output_pred": concat_images, "image_name": file_name})
        return pred_results

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Hook called at the end of each prediction batch.

        Saves the generated prediction images to the specified output directory.

        Args:
            outputs (list): The output of `predict_step` for the current batch.
            batch (dict): The input batch data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader. Defaults to 0.
        """
        output_dir = self.experiment_spec.results_dir
        for i in range(len(outputs)):
            target = os.path.join(output_dir, outputs[i]["image_name"])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            write_image(outputs[i]["output_pred"], target)

    def on_predict_epoch_end(self):
        """
        Hook called at the end of each prediction epoch.

        Computes and prints the aggregated prediction metrics.
        """
        results_metric = self.pred_evaluator.compute()
        print("{:<8} {:<15}".format("Metric", "Value"), "\n", flush=True)
        for k, v in results_metric.items():
            print("{:<8} {:<15}".format(k, v), "\n", flush=True)

    def on_test_epoch_start(self) -> None:
        """
        Hook called at the beginning of each test epoch.

        Initializes the depth metric evaluator for testing.
        """
        self.test_evaluator = StereoDepthEvaluator(sync_on_compute=False,
                                                   max_disparity=self.dataset_config.max_disparity).to(self.device)
        self.collate_results = {}

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.

        Calculates disparity predictions and updates the test evaluator
        with the results.

        Args:
            batch (dict): A dictionary containing the input data for the batch,
                          including 'image', 'disparity', and 'right_image'.
            batch_idx (int): The index of the current batch.
        """
        image1 = batch["image"]
        image_names = batch["image_path"]
        disp_gt = batch["disparity"]
        if torch.sum(disp_gt) < 0:
            self.is_valid_disparity_gt = False
        image2 = batch["right_image"]
        padder = InputPadder(image1.shape, divis_by=32)
        pad_image1, pad_image2 = padder.pad(image1, image2)
        disp_pred = self.model(
            pad_image1, pad_image2, iters=self.model_config["valid_iters"], test_mode=True
        )
        disp_pred = padder.unpad(disp_pred.cuda().float())
        invalid = disp_gt == torch.inf
        disp_gt[invalid] = 0
        disp_pred[invalid] = 0
        if self.is_valid_disparity_gt:
            self.test_evaluator.update(disp_pred, disp_gt)

        index = get_dataset_index(image_names[0], self.dataset_config["test_dataset"])
        if self.experiment_spec["inference"]["save_raw_pfm"]:
            filename = get_filename_from_path(
                image_names[0],
                self.dataset_config["infer_dataset"]["data_sources"][index]['dataset_name'])
            filename = '.'.join(filename.split('.')[:-1]) + '.pfm'
            target = os.path.join(self.experiment_spec["results_dir"], filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            write_pfm(target, disp_pred[0].squeeze().detach().cpu().numpy())

    def on_test_epoch_end(self):
        """
        Hook called at the end of each test epoch.

        Computes and logs test metrics, and updates `status.json`.
        """
        results_metric = self.test_evaluator.compute()
        for name, metric in results_metric.items():
            self.log(f"test/{name}", metric, sync_dist=True)

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
        """
        Performs a forward pass through the DepthNet model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output of the DepthNet model.
        """
        outputs = self.model(x)
        return outputs
