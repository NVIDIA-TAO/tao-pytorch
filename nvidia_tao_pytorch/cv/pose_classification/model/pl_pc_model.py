# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Main PTL model file for pose classification."""

import numpy as np
import pytorch_lightning as pl
from tabulate import tabulate
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.pose_classification.model.build_nn_model import build_pc_model
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


# pylint:disable=too-many-ancestors
class PoseClassificationModel(TAOLightningModule):
    """PyTorch Lightning module for single stream pose classification."""

    def __init__(self, experiment_spec, export=False):
        """
        Initialize the training for pose classification model.

        Args:
            experiment_spec (dict): The experiment specifications.
            export (bool, optional): If set to True, the model is prepared for export. Defaults to False.
        """
        super().__init__(experiment_spec)

        # init the model
        self._build_model(export)

        self.label_map = self.dataset_config["label_map"]
        self.num_classes = len(self.label_map.keys())
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.status_logging_dict = {}

        self.checkpoint_filename = 'pc_model'

    def _build_model(self, export):
        """
        Internal function to build the model.

        Args:
            experiment_spec (dict): The experiment specifications.
            export (bool): If set to True, the model is prepared for export.
        """
        self.model = build_pc_model(experiment_config=self.experiment_spec,
                                    export=export)
        print(self.model)

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            dict: A dictionary containing the optimizer, learning rate scheduler, and the metric to monitor.

        Raises:
            NotImplementedError: If an unsupported scheduler type is provided.
        """
        self.train_config = self.experiment_spec["train"]
        optim_dict = {}
        optim = torch.optim.SGD(params=self.parameters(),
                                lr=self.train_config['optim']['lr'],
                                momentum=self.train_config['optim']['momentum'],
                                nesterov=self.train_config['optim']['nesterov'],
                                weight_decay=self.train_config['optim']['weight_decay'])
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config['optim']['lr_scheduler']
        if scheduler_type == "AutoReduce":
            lr_scheduler = ReduceLROnPlateau(optim, 'min',
                                             patience=self.train_config['optim']['patience'],
                                             min_lr=self.train_config['optim']['min_lr'],
                                             factor=self.train_config['optim']["lr_decay"])
        elif scheduler_type == "MultiStep":
            lr_scheduler = MultiStepLR(optimizer=optim,
                                       milestones=self.train_config['optim']["lr_steps"],
                                       gamma=self.train_config['optim']["lr_decay"])
        else:
            raise NotImplementedError("Only [AutoReduce, MultiStep] schedulers are supported")

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['lr_monitor']

        return optim_dict

    def training_step(self, batch, batch_idx):
        """
        Define a training step.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        data, label = batch
        batch_size = data.shape[0]
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.train_accuracy.update(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_acc_1", self.train_accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json at the end of the epoch."""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss
        self.status_logging_dict["train_acc"] = self.train_accuracy.compute().item()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def validation_step(self, batch, batch_idx):
        """
        Define a validation step.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss.
        """
        data, label = batch
        batch_size = data.shape[0]
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.val_accuracy.update(output, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc@1", self.val_accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Log Validation metrics to status.json at the end of the epoch.

        Args:
            validation_step_outputs (list): List of outputs from each validation step.
        """
        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["val_acc"] = self.val_accuracy.compute().item()
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

    def test_step(self, batch, batch_idx):
        """Test step"""
        data, label = batch
        batch_size = len(data)
        cls_scores = self.model(data)
        pred_id = torch.argmax(cls_scores, dim=1).cpu().numpy()
        for idx in range(batch_size):
            self.confusion_matrix[label[idx].item(), pred_id[idx]] += 1

    def on_test_epoch_end(self):
        """Test epoch end"""
        percentage_confusion_matrix, accuracy, m_accuracy = self.compute_metrics(self.confusion_matrix)
        table = []
        id2name = {v: k for k, v in self.label_map.items()}
        for idx in range(len(self.label_map)):
            cls_acc = percentage_confusion_matrix[idx][idx]
            table.append(["Class accuracy: " + id2name[idx], cls_acc])
        table.append(["Total accuracy", accuracy])
        table.append(["Average class accuracy", m_accuracy])
        status_logging.get_status_logger().kpi = {"accuracy": round(accuracy, 2), "avg_accuracy": round(m_accuracy, 2)}
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
        print(tabulate(table, headers=["Name", "Score"], floatfmt=".4f", tablefmt="fancy_grid"))

        self.confusion_matrix = []

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        label_map = self.dataset_config["label_map"]
        self.id2name = {v: k for k, v in label_map.items()}
        self.results = []

    def predict_step(self, batch, batch_idx):
        """Predict step"""
        data, _ = batch
        cls_scores = self.model(data)
        pred_id = torch.argmax(cls_scores, dim=1).cpu().numpy()
        pred_name = []
        for label_idx in pred_id:
            pred_name.append(self.id2name[label_idx])
        self.results.extend(pred_name)

    def on_predict_epoch_end(self):
        """Predict epoch end"""
        # save the output
        output_file = open(self.experiment_spec["inference"]["output_file"], "w")
        for idx in range(len(self.results)):
            output_file.write("{}\n".format(self.results[idx]))
        output_file.close()

        self.results = []

    def forward(self, x):
        """
        Define the forward pass of the pose classification model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        output = self.model(x)
        return output

    def compute_metrics(self, confusion_matrix):
        """
        Compute evaluation metrics based on the confusion matrix.

        This function computes the percentage confusion matrix, accuracy, and average class accuracy
        from the provided confusion matrix.

        Args:
            confusion_matrix (np.ndarray): The confusion matrix of shape (num_classes, num_classes).

        Returns:
            np.ndarray: The percentage confusion matrix of the same shape as the input matrix.
            float: The overall accuracy.
            float: The average class accuracy.
        """
        row_sum = np.sum(confusion_matrix, axis=1)
        _shape = confusion_matrix.shape
        percentage_confusion_matrix = np.zeros(
            _shape, dtype=np.float32)
        for x in range(_shape[0]):
            for y in range(_shape[1]):
                if not row_sum[x] == 0:
                    percentage_confusion_matrix[x][y] = np.float32(confusion_matrix[x][y]) / \
                        row_sum[x] * 100.0

        trace = np.trace(confusion_matrix)
        percent_trace = np.trace(percentage_confusion_matrix)

        accuracy = float(trace) / np.sum(confusion_matrix) * 100.0
        m_accuracy = percent_trace / _shape[0]

        return percentage_confusion_matrix, accuracy, m_accuracy
