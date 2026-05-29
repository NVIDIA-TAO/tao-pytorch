# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classifier_pl Model PyTorch Lightning Module"""

import re
import os
import csv
from typing import Sequence, Union
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics.classification import Accuracy
from torchmetrics import MetricCollection
from transformers.optimization import get_cosine_schedule_with_warmup

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.core.path_utils import expand_path
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier import build_model
from nvidia_tao_pytorch.cv.classification_pyt.utils.loss import Cross_Entropy
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint
from nvidia_tao_pytorch.cv.classification_pyt.utils.utils_vis import (
    save_with_text_overlay, sync_tensor
)
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.augmentation import apply_mixup_cutmix


class ClassifierPlModel(TAOLightningModule):
    """
    PTL Model for Classifier
    """

    def __init__(self, experiment_spec, export=False):
        """pl_model initialization

        Args:
            experiment_spec (OmegaConf.DictConfig): Experiment configuration contains all the configurations. Default define in tao-core and user specify in yaml.
            export (bool, optional): No use in current Classifier repo because the model will not change the forward/architecture. Defaults to False.
        """
        super().__init__(experiment_spec)
        # Overriding what's done in super()
        self.checkpoint_filename = "classifier_model"
        self.dataset_config = self.experiment_spec.dataset
        self.model_config = self.experiment_spec.model
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.lr_policy_params = self.optimizer.policy_params
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name
        self.num_classes = self.dataset_config.num_classes

        # construct prediction id 2 class name mapping for visualization
        self.id_2_class_names = {}
        self.class_names = []
        classes_file = (self.dataset_config.classes_file
                        if os.path.exists(self.dataset_config.classes_file)
                        else os.path.join(self.dataset_config.root_dir, "classes.txt"))
        with open(classes_file) as f:
            for idx, line in enumerate(f):
                self.id_2_class_names[idx] = line.strip()
                self.class_names.append(line.strip())

        train_acc = {}
        val_acc = {}
        for topk in self.model_config.head.topk:
            train_acc[f"train_acc_{topk}"] = Accuracy(
                task="multiclass", num_classes=self.num_classes, top_k=topk
            )
            val_acc[f"val_acc_{topk}"] = Accuracy(
                task="multiclass", num_classes=self.num_classes, top_k=topk
            )
        self.train_acc = MetricCollection(train_acc)
        self.valid_acc = MetricCollection(val_acc)
        self.batch_size = self.dataset_config.batch_size

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.batch = None
        self.vis_dir = self.experiment_spec.results_dir
        self.optimizer_G = None

        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches

        # init the model
        self._build_model(export)
        self._build_criterion()

    def configure_callbacks(self) -> Union[Sequence[Callback], pl.Callback]:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(results_dir, append=True)

        resume_ckpt = self.experiment_spec["train"][
            "resume_training_checkpoint_path"
        ] or get_latest_checkpoint(results_dir)
        resumed_epoch = 0
        if resume_ckpt:
            resumed_epoch = re.search("epoch_(\\d+)", resume_ckpt)
            if resumed_epoch is not None:
                resumed_epoch = int(resumed_epoch.group(1))
            else:
                resumed_epoch = 0

        status_logger_callback.epoch_counter = resumed_epoch + 1
        callbacks.append(status_logger_callback)

        if self.experiment_spec["train"]["enable_ema"]:
            # Apply Exponential Moving Average Callback
            ema_callback = EMA(decay=self.experiment_spec["train"]["ema_decay"])
            ckpt_func = EMAModelCheckpoint
            callbacks.append(ema_callback)
        else:
            ckpt_func = ModelCheckpoint

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError(
                "checkpoint_filename not set in __init__() of model"
            )
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = ckpt_func(
            every_n_epochs=checkpoint_interval,
            dirpath=results_dir,
            save_on_train_epoch_end=True,
            monitor=None,
            save_top_k=-1,
            save_last="link",
            filename="model_{epoch:03d}",
            enable_version_counter=False,
        )
        callbacks.append(checkpoint_callback)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        return callbacks

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.criterion = Cross_Entropy(
            label_smoothing=self.model_config.head.loss.label_smooth_val,
        )

    @staticmethod
    def _get_parameter_groups(model, weight_decay, skip_names=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(s in name for s in skip_names):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def configure_optimizers(self):
        """Configure optimizers for training"""
        parameters = self._get_parameter_groups(
            self.model, self.optimizer.weight_decay, self.optimizer.skip_names
        )
        # define optimizers
        if self.optimizer.optim == "sgd":
            self.optimizer_G = optim.SGD(
                parameters,
                lr=self.lr,
                momentum=self.optimizer.momentum,  # 0.9
                weight_decay=self.optimizer.weight_decay,
            )  # 5e-4
        elif self.optimizer.optim == "adam":
            self.optimizer_G = optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.optimizer.weight_decay,
            )  # 0
        elif self.optimizer.optim == "adamw":
            self.optimizer_G = optim.AdamW(
                parameters,
                lr=self.lr,
                betas=self.optimizer.betas,
                weight_decay=self.optimizer.weight_decay,
            )
        else:
            raise NotImplementedError(
                "Optimizer {} is not implemented".format(self.optimizer.optim)
            )

        # Create main scheduler based on policy
        lr_policy = self.lr_policy.lower()
        if lr_policy == "linear":
            interval = "epoch"

            def lambda_rule(epoch):
                # gradually decay learning rate from epoch 0 to max_epochs
                lr_l = 1 - (epoch) / float(self.max_epochs + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif lr_policy == "step":
            interval = "epoch"
            if self.lr_policy_params is not None:
                step_size = self.lr_policy_params.step_size
                gamma = self.lr_policy_params.gamma
            else:   # default values
                step_size = self.max_epochs // 4
                gamma = 0.1
            # args.lr_decay_iters
            scheduler = lr_scheduler.StepLR(
                self.optimizer_G, step_size=step_size, gamma=gamma
            )
        elif lr_policy == "multistep":
            interval = "epoch"
            if self.lr_policy_params is not None:
                milestones = self.lr_policy_params.milestones
                gamma = self.lr_policy_params.gamma
            else:
                milestones = [self.max_epochs // 2]
                gamma = 0.1
            scheduler = lr_scheduler.MultiStepLR(self.optimizer_G, milestones, gamma=gamma)
        elif lr_policy == "cosine":
            interval = "step"
            epoch_steps = self.trainer.estimated_stepping_batches // (self.trainer.max_epochs * self.trainer.accumulate_grad_batches)
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer_G,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=epoch_steps * self.optimizer.warmup_epochs,
            )
        else:
            raise NotImplementedError('learning rate policy [{}] is not implemented'.format(self.lr_policy))

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = {
            "scheduler": self.lr_scheduler,
            "interval": interval,
            "frequency": 1
        }
        optim_dict["monitor"] = self.monitor_name
        return optim_dict

    def _initialize_csv(self, file_name, class_names, with_gt_label):
        """Initializes the CSV file with the appropriate header."""
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            heading = ["img_name"]
            if with_gt_label:
                heading = heading + class_names
            heading = heading + ["pred_label", "pred_score"]
            if with_gt_label:
                heading.append("gt_label")
            writer.writerow(heading)

    def _write_prediction_row(
        self,
        writer,
        name,
        class_scores,
        pred_label,
        pred_score,
        with_gt_label,
        gt_label=None,
    ):
        """Writes a single prediction row to the CSV."""
        # also add the visualization folder for the inference
        if "inference" in self.vis_dir:
            # check "visualize" folder exists in vis_dir, if not, create one
            if not os.path.exists(os.path.join(self.vis_dir, "visualize")):
                os.makedirs(os.path.join(self.vis_dir, "visualize"))
        row = [name]
        if with_gt_label and gt_label is not None:
            row = row + class_scores
        row = row + [pred_label, pred_score]
        if with_gt_label and gt_label is not None:
            row.append(gt_label)
        writer.writerow(row)

    def _visualize_predictions(self, out, batch, batch_idx, with_gt_label=False):
        """
        Save the predictions and ground truth in csv format. With the csv keys as ["img_name", "pred_label", "pred_score", "gt_label"] and also each class's score.
        """
        out = torch.nn.functional.softmax(
            out, dim=1
        )  # Apply softmax to get probabilities
        file_name = expand_path(
            os.path.join(self.vis_dir, "result.csv")
        )  # Get the file path

        # Initialize the CSV file header on the first batch
        if batch_idx == 0:
            self._initialize_csv(file_name, self.class_names, with_gt_label)

        write_out_argument = []
        with open(file_name, "a") as f:  # Open the file in append mode
            writer = csv.writer(f)
            for i in range(len(batch["name"])):  # Iterate over the batch size
                # Convert each class score to a list
                each_class_score = out[i].cpu().detach().tolist()
                pred_score, pred_label_idx = torch.max(
                    out[i], 0
                )  # Get the max score and corresponding label index
                pred_label = self.class_names[
                    pred_label_idx.item()
                ]  # Map index to class name

                if with_gt_label:  # Include ground truth label if required
                    gt_label = self.class_names[batch["class"][i].item()]
                    self._write_prediction_row(
                        writer,
                        batch["name"][i],
                        each_class_score,
                        pred_label,
                        pred_score.item(),
                        with_gt_label,
                        gt_label,
                    )
                else:
                    self._write_prediction_row(
                        writer,
                        batch["name"][i],
                        each_class_score,
                        pred_label,
                        pred_score.item(),
                        with_gt_label,
                    )
                    # for pred_score, only keep 4 decimal points
                    write_out_argument.append(
                        [pred_label_idx.item(), round(pred_score.item(), 3), pred_label]
                    )

        if not with_gt_label:
            # imgs = de_norm(batch['img'], self.dataset_config["augmentation"]["mean"], self.dataset_config["augmentation"]["std"])
            # imgs_size = np.transpose(batch["size"].cpu().numpy())
            # for i, img in enumerate(imgs):
            for i in range(len(batch["name"])):
                filename = batch["name"][i].split("/")[-1]
                img_path = expand_path(
                    os.path.join(self.vis_dir, "visualize", filename)
                )
                # save_with_text_overlay(img, write_out_argument[i], img_path, imgs_size[i])
                save_with_text_overlay(
                    batch["name"][i], write_out_argument[i], img_path
                )

    def _forward_pass(self, batch, split="train"):
        """Forward pass for training, validation and testing."""
        out = self.model(batch["img"])
        if split == "predict":
            self.loss = None
        else:
            # if score in batch it means using mixup or cutmix
            if "score" in batch:
                self.loss = self.criterion(out, batch['score'].float())
            elif "class" in batch:
                self.loss = self.criterion(out, batch['class'].long())
            else:
                raise ValueError("No label key 'class' or 'score' in batch")
        return out, self.loss

    def _apply_mixup_cutmix(self, batch):
        """Apply mixup or cutmix augmentation."""
        # convert to one_hot as mixup need one hot label
        score = torch.nn.functional.one_hot(batch["class"].clone().detach().long(), len(self.class_names)).float()
        # random select "mixup" or "cutmix"
        mixup_type = "mixup" if torch.rand(1) < 0.5 else "cutmix"

        lam = float(torch.distributions.beta.Beta(self.dataset_config["augmentation"]["mixup_alpha"], self.dataset_config["augmentation"]["mixup_alpha"]).sample())
        lam = float(np.clip(lam, 0, 1))
        lam = float(sync_tensor(lam, reduce_method="root"))

        img, score = apply_mixup_cutmix(batch["img"], score, mix_type=mixup_type, lam=lam)
        batch["img"] = img
        batch["score"] = score
        return batch

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.dataset_config.augmentation.mixup_cutmix:
            batch = self._apply_mixup_cutmix(batch)

        out, loss = self._forward_pass(batch, "train")
        acc = self.train_acc(out, batch["class"].long())
        self.log_dict(acc, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[-1],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()
        # self.log('train_acc_epoch', self.train_acc.compute())
        self.train_acc.reset()
        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        out, loss = self._forward_pass(batch, "val")
        self.valid_acc.update(out, batch["class"].long())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        # FLUSHING VALIDATION EPOCH METRICS
        # scores, mean_scores = self._collect_epoch_states()  # logs all evaluation metrics
        # self.log("val_acc", scores['acc'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        acc = self.valid_acc.compute()
        self.log_dict(acc, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()
        # self._clear_cache()

        average_val_loss = self.trainer.logged_metrics["val_loss"].item()
        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            for acc_key in acc.keys():
                self.status_logging_dict[acc_key] = acc[acc_key].item()
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING,
            )

        pl.utilities.memory.garbage_collection_cuda()

    def test_step(self, batch, batch_idx):
        """Test step."""
        out, loss = self._forward_pass(batch, "test")

        # Calculate running metrics
        self.valid_acc.update(out, batch["class"].long())

        self._visualize_predictions(out, batch, batch_idx, with_gt_label=True)

        self.log("test_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    def on_test_epoch_end(self):
        """Test epoch end."""
        # scores, mean_scores = self._collect_epoch_states()  # needed for update metrics
        acc = self.valid_acc.compute()
        self.log_dict(acc, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()
        self.status_logging_dict = {}
        for acc_key in acc.keys():
            self.status_logging_dict[acc_key] = acc[acc_key].item()
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING,
        )

    def predict_step(self, batch, batch_idx):
        """Predict step."""
        out, _ = self._forward_pass(batch, "predict")

        self._visualize_predictions(out, batch, batch_idx, with_gt_label=False)

        return out

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "classification"
