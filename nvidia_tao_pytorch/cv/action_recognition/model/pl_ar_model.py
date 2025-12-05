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

""" Main PTL model file for action recognition """

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchmetrics

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.action_recognition.model.build_nn_model import build_ar_model


# pylint:disable=too-many-ancestors
class ActionRecognitionModel(TAOLightningModule):
    """ PTL module for action recognition model."""

    def __init__(self, experiment_spec, dm, export=False):
        """Init training for 2D/3D action recognition model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool, optional): Whether to build the model that can be exported to ONNX format. Defaults to False.
        """
        super().__init__(experiment_spec)

        # init the model
        self._build_model(export)
        self.dm = dm

        self.label_map = self.dataset_config["label_map"]
        self.num_classes = len(self.label_map.keys())
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.status_logging_dict = {}

        self.checkpoint_filename = 'ar_model'

    def _build_model(self, export):
        """Internal function to build the model.

        This method constructs a model using the specified experiment specification and export flag. It returns the model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool): Whether to build the model that can be exported to ONNX format.
        """
        self.model = build_ar_model(experiment_config=self.experiment_spec,
                                    export=export)
        print(self.model)

    def configure_optimizers(self):
        """Configure optimizers for training"""
        self.train_config = self.experiment_spec["train"]
        optim_dict = {}
        optim = torch.optim.SGD(params=self.parameters(),
                                lr=self.train_config['optim']['lr'],
                                momentum=self.train_config['optim']['momentum'],
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
            raise ValueError("Only [AutoReduce, MultiStep] scheduler is supported")

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['lr_monitor']

        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, label = batch
        batch_size = data.shape[0]
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.train_accuracy.update(output, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_acc_1", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
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
        """Validation step."""
        _, data, label = batch
        batch_size = data.shape[0]
        output = self.model(data)
        loss = F.cross_entropy(output, label)
        self.val_accuracy.update(output, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc_1", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Log Validation metrics to status.json"""
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

    def forward(self, x):
        """Forward of the action recognition model."""
        output = self.model(x)
        return output

    def on_test_epoch_start(self) -> None:
        """ Test epoch start."""
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int32)
        self.eval_mode_flag = self.experiment_spec["evaluate"]["video_eval_mode"] == "conv"

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate """
        sample_pred_dict = {}
        sample_path, data, action_label = batch
        batch_size = len(sample_path)
        cls_scores = self.model(data)

        # TODO @seanf: cc says eval_mode_flag = True is never tested
        if self.eval_mode_flag:
            prob = torch.softmax(cls_scores, dim=1)
            for idx in range(batch_size):
                if sample_path[idx] not in sample_pred_dict:
                    sample_pred_dict[sample_path[idx]] = prob[idx]
                sample_pred_dict[sample_path[idx]] += prob[idx]

            for k, v in sample_pred_dict.items():
                pred_id = np.argmax(v)
                action = self.dm.sample_dict[k]
                self.confusion_matrix[self.label_map[action], pred_id] += 1

        else:
            pred_id = torch.argmax(cls_scores, dim=1)
            for idx in range(batch_size):
                self.confusion_matrix[action_label[idx], pred_id[idx]] += 1

    def on_test_epoch_end(self):
        """Test epoch end."""
        confusion_matrix_gathered = self.all_gather(self.confusion_matrix)
        if self.trainer.local_rank == 0:
            confusion_matrix_gathered = confusion_matrix_gathered.cpu().numpy()
            if len(confusion_matrix_gathered.shape) == 3:
                confusion_matrix_gathered = np.sum(confusion_matrix_gathered, axis=0)
            percentage_confusion_matrix, accuracy, m_accuracy = self.compute_metrics(confusion_matrix_gathered)

            id2name = {v: k for k, v in self.label_map.items()}
            print("*******************************")
            for idx in range(len(self.label_map)):
                cls_acc = percentage_confusion_matrix[idx][idx]
                print("{:<14}{:.4}".format(
                    id2name[idx], cls_acc))

            print("*******************************")
            print("Total accuracy: {}".format(round(accuracy, 3)))
            print("Average class accuracy: {}".format(round(m_accuracy, 3)))

            status_logging.get_status_logger().kpi = {"accuracy": round(accuracy, 3),
                                                      "m_accuracy": round(m_accuracy, 3)}
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        self.confusion_matrix = []

    def compute_metrics(self, confusion_matrix):
        """Computes evaluation metrics.

        Args:
            confusion_matrix (numpy.ndarray): The confusion matrix.

        Returns:
            dict: A dictionary containing the evaluation metrics.
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

    def on_predict_epoch_start(self):
        """ Inference epoch start"""
        label_map = self.dataset_config["label_map"]
        self.id2name = {v: k for k, v in label_map.items()}
        self.sample_result_dict = {}

    def predict_step(self, batch, batch_idx):
        """ Inference step"""
        sample_path, data = batch
        batch_size = len(sample_path)
        cls_scores = self.model(data)
        pred_id = torch.argmax(cls_scores, dim=1).cpu().numpy()
        pred_name = []
        for label_idx in pred_id:
            pred_name.append(self.id2name[label_idx])
        for idx in range(batch_size):
            if sample_path[idx] not in self.sample_result_dict:
                self.sample_result_dict[sample_path[idx]] = [pred_name[idx]]
            else:
                self.sample_result_dict[sample_path[idx]].append(pred_name[idx])

    def on_predict_epoch_end(self) -> None:
        """ Inference epoch end"""
        # save the output and visualize
        for k, v in self.sample_result_dict.items():
            print("{} : {}".format(k, v))
