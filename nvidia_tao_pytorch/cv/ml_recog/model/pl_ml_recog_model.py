# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main PTL model file for Metric Learning Recognition."""

from collections import defaultdict
import os

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.ml_recog.utils.match_finder import EmbeddingKNN
from pytorch_metric_learning import losses, miners, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel

from nvidia_tao_pytorch.cv.ml_recog.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.re_identification.lr_schedulers.warmup_multi_step_lr import WarmupMultiStepLR
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class MLRecogModel(TAOLightningModule):
    """PTL module for single stream Metric Learning Recognition. The training
    process is to minimize the distances between the embeddings of the same class
    and maximize the distances between the embeddings of different classes. The
    validation process is to evaluate the similarity search performance of the
    model on the validation reference and query datasets. The validation process
    only supports running on a single GPU.
    """

    def __init__(self, experiment_spec, dm, subtask="train"):
        """Initializes training for Metric Learning Recognition model.

        Args:
            experiment_spec (DictConfig): Configuration File
            subtask (String): The purpose of the model. Can be "train", "evaluate", "export", "inference" only
        """
        super().__init__(experiment_spec)
        self.subtask = subtask
        self.dm = dm

        self.status_logging_dict = {}

        if subtask == "train":
            checkpoint = self.experiment_spec["train"]["resume_training_checkpoint_path"]
        elif subtask in ("evaluate", "export", "inference"):
            checkpoint = self.experiment_spec[subtask]["checkpoint"]
            self.topk = self.experiment_spec["inference"]["topk"]
        if checkpoint:
            # Failure should always be caught before or after this warning
            if not os.path.exists(checkpoint):
                checkpoint_to_load = False
            else:
                checkpoint_to_load = True
        else:
            checkpoint_to_load = False
        self._build_model(checkpoint_to_load)
        # Activates manual optimization
        self.automatic_optimization = False
        if self.subtask == "train":
            status_logging.get_status_logger().write(
                message="Preparing for training",
                status_level=status_logging.Status.RUNNING)
            self.my_loss_func = self.__make_loss(experiment_spec)
            self.load_tester()

        self.checkpoint_filename = 'ml_model'

    def load_tester(self):
        """Loads a `pytorch_metric_learning.testers.GlobalTwoStreamEmbeddingSpaceTester` to prepare for gallery-query similarity search evaluation."""
        # suppress end test print results
        def end_test_hook(tester):
            pass
        self.tester = testers.GlobalEmbeddingSpaceTester(
            batch_size=self.experiment_spec["train"]["val_batch_size"],
            end_of_testing_hook=end_test_hook,
            dataloader_num_workers=self.dataset_config["workers"],
            accuracy_calculator=AccuracyCalculator(
                k="max_bin_count",
                return_per_class=self.experiment_spec[self.subtask]["report_accuracy_per_class"]),
        )

    def _build_model(self, checkpoint_to_load=False):
        self.model = build_model(
            self.experiment_spec,
            checkpoint_to_load=checkpoint_to_load)
        if self.subtask != "train":
            self.model.eval()

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns:
            optim_dict1 (Dict[String, Object]): a map for trunk's optimizer, monitor and lr scheduler
            optim_dict2 ( Dict[String, Object]): a map for embedder's optimizer, monitor and lr scheduler
        """
        self.train_config = self.experiment_spec["train"]
        self.optim_config = self.train_config["optim"]
        optimizers = self.__make_optimizer()
        self.schedulers = {}
        for k, opt in optimizers.items():
            sch = WarmupMultiStepLR(
                opt, self.optim_config["steps"],
                gamma=self.optim_config["gamma"],
                warmup_factor=self.optim_config["warmup_factor"],
                warmup_iters=self.optim_config["warmup_iters"],
                warmup_method=self.optim_config["warmup_method"],
                last_epoch=self.current_epoch - 1,
            )
            self.schedulers[k] = sch
        optim_dict1 = {
            "optimizer": optimizers["trunk"],
            'monitor': None,
            "lr_scheduler": self.schedulers["trunk"]
        }
        optim_dict2 = {
            "optimizer": optimizers["embedder"],
            'monitor': None,
            "lr_scheduler": self.schedulers["embedder"]
        }
        return (optim_dict1, optim_dict2)

    def __make_module_optimizer(self, model_name):
        if model_name == "embedder":
            model = self.model.embedder
        elif model_name == "trunk":
            model = self.model.trunk
        params = []
        for key, value in model.named_parameters():
            lr = self.optim_config[model_name]["base_lr"]
            weight_decay = self.optim_config[model_name]["weight_decay"]
            if "bias" in key:
                lr = self.optim_config[model_name]["base_lr"] * self.optim_config[model_name]["bias_lr_factor"]
                weight_decay = self.optim_config[model_name]["weight_decay_bias"]
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if self.optim_config["name"] == 'SGD':
            optimizer = getattr(torch.optim, self.optim_config["name"])(params, momentum=self.optim_config[model_name]["momentum"])
        else:
            optimizer = getattr(torch.optim, self.optim_config["name"])(params)
        return optimizer

    def __make_optimizer(self):
        embedder_optimizer = self.__make_module_optimizer("embedder")
        trunk_optimizer = self.__make_module_optimizer("trunk")
        optimizers = {
            "embedder": embedder_optimizer,
            "trunk": trunk_optimizer
        }
        return optimizers

    def training_step(self, batch):
        """Training step.

        Args:
            batch (torch.Tensor): Batch of data

        Returns:
            loss (torch.float32): Loss value for each step in training
        """
        data, labels = batch
        data = data.float()
        batch_size = data.shape[0]
        opt1, opt2 = self.optimizers()
        self.optimizer_dict = {'trunk': opt1, "embedder": opt2}
        self._zero_grad()
        outputs = self.model(data)
        loss = self.my_loss_func(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("trunk_base_lr", self.schedulers["trunk"].get_lr()[0],
                 on_step=True, on_epoch=False, prog_bar=True)
        self.log("embedder_base_lr", self.schedulers["embedder"].get_lr()[0],
                 on_step=True, on_epoch=False, prog_bar=True)
        self.manual_backward(loss)
        # clip gradients
        for opt in self.optimizer_dict.values():
            self.clip_gradients(
                opt,
                gradient_clip_val=self.experiment_spec['train']['clip_grad_norm'],
                gradient_clip_algorithm="norm"
            )
        self._step_optimizers()
        self.current_loss = loss

        return loss

    def get_query_accuracy(self):
        """Obtains the metric results of gallery-query similarity search evaluation.

        Returns:
            all_accuracies (Dict[str, float]): a map of default accuracy metrics from
                `pytorch_metric_learning.utils.accuracy_calculator.AccuracyCalculator`.
                Explanations see
                https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/#explanations-of-the-default-accuracy-metrics
        """
        if self.subtask == "train":
            epoch_log = self.current_epoch
        else:
            epoch_log = "eval mode"
        all_accuracies = self.tester.test(self.dm.dataset_dict,
                                          epoch_log,  # used for logging
                                          self.model,  # your model
                                          splits_to_eval=[('query', ['gallery'])]
                                          )
        return all_accuracies

    @rank_zero_only
    def update_validation_metrics(self, report_acc_per_class, test=False):
        """Updates the validation metrics at the end of each validation epoch."""
        all_accuracies = self.get_query_accuracy()
        # df = self.report_accuracies(all_acc, save_results=True)
        ami = all_accuracies['query']['AMI_level0']
        nmi = all_accuracies['query']['NMI_level0']
        mean_avg_prec = all_accuracies['query']['mean_average_precision_level0']
        mean_reciprocal_rank = all_accuracies['query']['mean_reciprocal_rank_level0']
        mean_r_precision = all_accuracies['query']['r_precision_level0']
        val_accuracy = all_accuracies['query']['precision_at_1_level0']

        if test:
            prefix = "test"
        else:
            prefix = "val"

        self.status_logging_dict = {}
        self.status_logging_dict[f'{prefix}_AMI'] = ami
        self.status_logging_dict[f'{prefix}_NMI'] = nmi
        if report_acc_per_class:
            self.status_logging_dict[f'{prefix} Mean Average Precision'] = sum(mean_avg_prec) / len(mean_avg_prec)
            self.status_logging_dict[f'{prefix} Mean Reciprocal Rank'] = sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
            self.status_logging_dict[f'{prefix} r-Precision'] = sum(mean_r_precision) / len(mean_r_precision)
            self.status_logging_dict[f'{prefix} Precision at Rank 1'] = sum(val_accuracy) / len(val_accuracy)

        else:
            self.status_logging_dict[f'{prefix} Mean Average Precision'] = mean_avg_prec
            self.status_logging_dict[f'{prefix} Mean Reciprocal Rank'] = mean_reciprocal_rank
            self.status_logging_dict[f'{prefix} r-Precision'] = mean_r_precision
            self.status_logging_dict[f'{prefix} Precision at Rank 1'] = val_accuracy

        if not self.trainer.sanity_checking:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            if test:
                status_logging.get_status_logger().write(
                    message="Test metrics generated.",
                    status_level=status_logging.Status.RUNNING
                )
            else:
                status_logging.get_status_logger().write(
                    message="Eval metrics generated.",
                    status_level=status_logging.Status.RUNNING
                )

        # print out validation results
        print("******************* Evaluation results **********************")
        print(f"Results at epoch {self.current_epoch}:")
        for k, v in self.status_logging_dict.items():
            if "val" in k or "test" in k:
                print(f"{k}: {v:.4f}")

        if report_acc_per_class:
            print("\n******************* Accuracy per class **********************")
            for k, v in all_accuracies['query'].items():
                if "level" in k:
                    if isinstance(v, list):
                        print(f"  {k[:-7]}:")
                        for i, vv in enumerate(v):
                            print(f"    {self.dm.class_dict[i]}: {vv:.4f}")
        print("============================================")

        pl.utilities.memory.garbage_collection_cuda()

    def on_train_epoch_end(self):
        """Generates train and validation metrics and the end of training epoch."""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()
        report_loss = np.around(average_train_loss, decimals=4)
        report_trunk_lr = '{:.4e}'.format(self.schedulers['trunk'].get_lr()[0])
        report_embedder_lr = '{:.4e}'.format(self.schedulers['embedder'].get_lr()[0])

        # validation_epoch_end does not work here. Manually set it up.
        # add one to match the checkpoint saving epochs
        if (self.current_epoch + 1) % self.experiment_spec['train']['checkpoint_interval'] == 0:
            self.update_validation_metrics(self.experiment_spec['train']['report_accuracy_per_class'])

        self.status_logging_dict = {}
        self.status_logging_dict['train_loss'] = report_loss
        self.status_logging_dict['trunk_base_lr'] = report_trunk_lr
        self.status_logging_dict['embedder_base_lr'] = report_embedder_lr

        # status loggings are rank zero only
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.load_tester()

    def test_step(self, batch):
        """Test step"""
        # Although this function does nothing, it is necessary for Trainer.test() to not error
        return

    def on_test_epoch_end(self):
        """Test epoch end"""
        self.update_validation_metrics(self.experiment_spec["evaluate"]["report_accuracy_per_class"], test=True)

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        infernce_knn_func = EmbeddingKNN(reset_before=False,
                                         reset_after=False)
        self.inference_model = InferenceModel(self.model,
                                              knn_func=infernce_knn_func)
        self.inference_model.train_knn(self.dm.dataset_dict["gallery"])
        self.csv_f = os.path.join(self.trainer.default_root_dir, 'result.csv')

        if os.path.exists(self.csv_f):
            os.remove(self.csv_f)

    def predict_step(self, batch):
        """Predict step"""
        distances, indices = self.inference_model.get_nearest_neighbors(
            batch[0], k=self.topk)
        class_indices = [self.dm.dataset_dict["gallery"][i][1] for i in indices.flatten()]
        class_labels = np.array([self.dm.class_dict[idx] for idx in
                                class_indices]).reshape(len(batch[0]), -1)
        df = pd.DataFrame(zip(list(batch[1]), class_labels.tolist(), distances.tolist()))

        return df

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Writes results to a csv file
        """
        outputs.to_csv(self.csv_f, header=False, index=False, mode='a')

    def on_predict_epoch_end(self):
        """Infers all images in an image folder or a classification folder.

        Returns:
            final_df (pd.DataFrame): a table displaying image file path,
                top k predicted classes, topk distances
        """
        status_logging.get_status_logger().write(
            message=f"result saved at {self.csv_f}",
            status_level=status_logging.Status.RUNNING)

    def _step_optimizers(self):
        for v in self.optimizer_dict.values():
            v.step()

    def _step_schedulers(self):
        for v in self.schedulers.values():
            v.step()

    def _zero_grad(self):
        self.model.zero_grad()
        for v in self.optimizer_dict.values():
            v.zero_grad()

    def forward(self, x):
        """Forward of the Metric Learning Recognition model.

        Args:
            x (torch.Tensor): Batch of data

        Returns:
            output (torch.Tensor): Output of the model (class score, feats)
        """
        output = self.model(x)
        return output

    def __make_loss(self, cfg):
        self.optim_config = cfg["train"]["optim"]
        loss_func = losses.TripletMarginLoss(
            margin=self.optim_config["triplet_loss_margin"],
            smooth_loss=self.experiment_spec["train"]["smooth_loss"])
        mining_func = miners.MultiSimilarityMiner(
            epsilon=self.optim_config["miner_function_margin"])

        def calculate_loss(embeddings, labels):
            indices_tuple = mining_func(embeddings, labels)
            metric_loss = loss_func(embeddings, labels, indices_tuple)
            return metric_loss

        return calculate_loss

    # TODO @seanf: cc reports this is never used
    def report_accuracies(self, acc_dict, save_results=False):
        """Converts the metrics results map to a pd.DataFrame table to display
        top1 precisions of all classes.

        Args:
            acc_dict (Dict[str, float]): A map of metrics and results obtained from
            `self.get_query_accuracies()`
            save_results (Boolean): If True, the derived dataframe would be saved
                to a csv file at output_dir/accuracy_per_class.csv

        Returns:
            df (pd.DataFrame): A table of top1 precision  of all classes.
        """
        output = defaultdict(dict)
        count = 0
        for idx in self.dm.class_dict:
            class_name = self.dm.class_dict[idx]
            if class_name in self.dm.dataset_dict["query"].empty_classes:
                output[class_name]['top1_acc'] = None
                count += 1
            else:
                output[class_name]['top1_acc'] = \
                    acc_dict['query']['precision_at_1_level0'][idx - count]
        df = pd.DataFrame.from_dict(output, orient='index')
        if save_results:
            df.to_csv(os.path.join(
                self.trainer.default_root_dir,
                "accuracy_per_class.csv"))
        return df
