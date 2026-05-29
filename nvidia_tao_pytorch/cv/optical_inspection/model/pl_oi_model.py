# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for optical inspection"""
import logging
import os
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import (build_oi_model, ContrastiveLoss1, AOIMetrics)


class OpticalInspectionModel(TAOLightningModule):
    """Pytorch Lighting for Optical Inspection

    Args:
        experiment_config (OmegaConf.DictConf): The experiment configuration.
        export (bool): Flag indicating whether to export the model.
    """

    def __init__(self, experiment_spec, dm, export=False, **kwargs):
        """Initialize"""
        super().__init__(experiment_spec, **kwargs)
        # init the model
        self._build_model(export)
        self.tensorboard = self.experiment_spec.train.tensorboard
        self.status_logging_dict = {}
        self.train_metrics = AOIMetrics()
        self.val_metrics = AOIMetrics()
        self.dm = dm

        self.checkpoint_filename = 'oi_model'

    def _build_model(self, export=False):
        self.model = build_oi_model(
            experiment_config=self.experiment_spec, export=export
        )
        print(self.model)

    def configure_optimizers(self):
        """Configure optimizers for training"""
        optim_dict = {}
        train_config = self.experiment_spec["train"]
        if train_config['optim']['type'] == 'Adam':
            optim = torch.optim.Adam(self.parameters(), train_config['optim']['lr'])
        else:
            optim = torch.optim.SGD(params=self.parameters(),
                                    lr=train_config['optim']['lr'],
                                    momentum=train_config['optim']['momentum'],
                                    weight_decay=train_config['optim']['weight_decay'])

        optim_dict["optimizer"] = optim
        return optim_dict

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: The input batch containing img0, img1, and label tensors.
            batch_idx: Index of the current batch.

        Returns:
            The computed loss value for the training step.
        """
        margin = self.model_config["margin"]
        img0, img1, label = batch
        batch_size = img0.shape[0]
        self.visualize_image(
            "compare_sample", img0,
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        self.visualize_image(
            "golden_sample", img1,
            logging_frequency=self.tensorboard.infrequent_logging_frequency
        )
        criterion = ContrastiveLoss1(margin)
        output1, output2 = self.model(img0, img1)
        loss = criterion(output1, output2, label)
        euclidean_distance = F.pairwise_distance(output1, output2)
        self.train_metrics.update(euclidean_distance, label)
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
        """Validation step.

        Args:
            batch: The input batch containing img0, img1, and label tensors.
            batch_idx: Index of the current batch.

        Returns:
            loss: Validation loss value.
        """
        margin = self.model_config["margin"]
        img0, img1, label = batch
        batch_size = img0.shape[0]
        criterion = ContrastiveLoss1(margin)
        output1, output2 = self.model(img0, img1)
        loss = criterion(output1, output2, label)
        euclidean_distance = F.pairwise_distance(output1, output2)

        self.val_metrics.update(euclidean_distance, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Log Validation metrics to status.json

        Args:
            validation_step_outputs: List of outputs from validation steps in the epoch.
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
        self.margin = self.model_config["margin"]
        self.valid_AOIMetrics = AOIMetrics(self.margin)
        self.num_comp = 0

    def test_step(self, batch, batch_idx):
        """Test step"""
        img0, img1, label = batch
        output1, output2 = self.model(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2)

        self.valid_AOIMetrics.update(euclidean_distance, label)
        self.num_comp += len(euclidean_distance)

    def on_test_epoch_end(self):
        """Test epoch end"""
        total_accuracy = self.valid_AOIMetrics.compute()['total_accuracy'].item()
        false_alarm = self.valid_AOIMetrics.compute()['false_alarm'].item()
        defect_accuracy = self.valid_AOIMetrics.compute()['defect_accuracy'].item()
        false_negative = self.valid_AOIMetrics.compute()['false_negative'].item()
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
        """Predict step"""
        img0, img1, _ = batch
        output1, output2 = self.model(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2)

        # TODO: is this logic correct?
        if batch_idx == 0:
            self.euclid = euclidean_distance
        else:
            self.euclid = torch.cat((self.euclid, euclidean_distance), 0)

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
            self.dm.df_infer['siamese_score'] = combined_results.cpu().numpy()

            self.dm.df_infer.to_csv(
                os.path.join(self.experiment_spec.results_dir, "inference.csv"),
                header=True,
                index=False
            )
            logging.info("Completed")

        # Clear the list for the next epoch
        self.euclid = []

    def forward(self, x1, x2):
        """Forward of the Optical Inspection model.

        Args:
            x1 (torch.Tensor): The first input image.
            x2 (torch.Tensor): The second input image.

        Returns:
            output (torch.Tensor): Output of the model.
        """
        output = self.model(x1, x2)
        return output

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

    # TODO @seanf: cc says this isn't used
    def load_final_model(self) -> None:
        """Loading a pre-trained network weights"""
        model_path = self.experiment_spec['inference']['checkpoint']
        gpu_device = self.experiment_spec['inference']['gpu_id']
        siamese_inf = self.model.cuda().load_state_dict(torch.load(model_path, map_location='cuda:' + str(gpu_device)))
        return siamese_inf
