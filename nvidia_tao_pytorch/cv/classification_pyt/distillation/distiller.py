# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distiller module for classification model"""
import os
import logging
import re
import copy
from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics.classification import Accuracy
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from transformers.optimization import get_cosine_schedule_with_warmup

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint

from nvidia_tao_pytorch.core.distillation.distiller import Distiller

from nvidia_tao_pytorch.cv.classification_pyt.distillation.loss import DistillationLoss
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier import build_model
from nvidia_tao_pytorch.cv.classification_pyt.utils.loss import Cross_Entropy
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.dataset import NOCLASS_IDX
logger = logging.getLogger(__name__)


class ClassDistiller(Distiller):
    """Classification Distiller"""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        # Init local params
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = "classifier_model"
        self.dataset_config = self.experiment_spec.dataset
        self.model_config = self.experiment_spec.model
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference
        self.distill_config = self.experiment_spec.distill

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.lr_policy_params = self.optimizer.policy_params
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.num_classes = self.dataset_config.num_classes
        self.distill_weight = self.distill_config.loss_lambda
        self.distill_loss = self.distill_config.loss_type
        if self.distill_loss == "FD" or self.distill_loss == "CS":
            assert self.num_classes == 0, "Number of classes must be 0 when using `FD` or `CS` as the distillation loss type"

        # construct prediction id 2 class name mapping for visualization
        self.id_2_class_names = {}
        self.class_names = []
        with open(os.path.join(self.dataset_config.root_dir, "classes.txt")) as f:
            for idx, line in enumerate(f):
                self.id_2_class_names[idx] = line.strip()
                self.class_names.append(line.strip())

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.batch = None
        self.vis_dir = self.experiment_spec.results_dir
        self.optimizer_G = None

        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches
        # init the model
        super().__init__(experiment_spec, export)

        train_acc = {}
        val_acc = {}
        if self.num_classes > 0:
            for topk in self.model_config.head.topk:
                train_acc[f"train_acc_{topk}"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=topk,
                    ignore_index=NOCLASS_IDX,
                )
                val_acc[f"val_acc_{topk}"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=topk,
                    ignore_index=NOCLASS_IDX,
                )
        self.train_acc = MetricCollection(train_acc)
        self.valid_acc = MetricCollection(val_acc)
        self.batch_size = self.dataset_config.batch_size

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        self.checkpoint_filename = "classifier_model"
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
        )

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
            ema_callback = EMA(**self.experiment_spec["train"]["ema"])
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
        return callbacks

    def _setup_bindings(self):
        """Setup bindings to be captured during training for distillation."""
        pass

    def _build_model(self, export=False):
        """Internal function to build the model."""
        # Build the teacher config
        teacher_cfg = copy.deepcopy(self.experiment_spec)
        teacher_cfg.model = self.experiment_spec.distill.teacher
        teacher_cfg.model.backbone.pretrained_backbone_path = self.experiment_spec.distill.pretrained_teacher_model_path

        if 'radio' in teacher_cfg.model.backbone.type:
            assert self.num_classes == 0, "Number of classes must be 0 when using radio as the teacher"
        if self.num_classes == 0:
            assert self.distill_loss in ["FD", "CS", "balanced", "MSE"], \
                "Only FD (`smooth L1`), CS (`cosine similarity`), balanced (`cosine similarity` + `smooth L1`, MSE) are supported when the number of classes is 0"

        # Build the teacher model
        self.teacher = build_model(experiment_config=teacher_cfg, export=export)
        # Build the student model
        self.model = build_model(experiment_config=self.experiment_spec, export=export)
        self.teacher.eval()
        self.model.train()

        # Freeze teacher
        for _, param in self.teacher.named_parameters():
            param.requires_grad = False

        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.LayerNorm):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def _build_criterion(self):
        """Internal function to build the loss function."""
        assert self.model_config.head.loss.type in [
            "CrossEntropyLoss"
        ], "Only CrossEntropyLoss is supported."
        if self.model_config.head.loss.type == "CrossEntropyLoss":
            self.criterion = Cross_Entropy(
                label_smoothing=self.model_config.head.loss.label_smooth_val,
            )
        else:
            raise NotImplementedError(self.train_config["loss"])

        # Create the distillation loss module
        self.distillation_loss_fn = DistillationLoss(
            loss_type=self.distill_loss,
            student_model=self.model,
            teacher_model=self.teacher,
            distillation_mode=self.distill_config.mode or "auto",  # Auto-detect based on loss type
            num_classes=self.num_classes,
            temperature=getattr(self.distill_config, 'temperature', 1.0),
            use_mlp=getattr(self.distill_config, 'use_mlp', True),
            mlp_hidden_size=getattr(self.distill_config, 'mlp_hidden_size', 1024),
            mlp_num_inner=getattr(self.distill_config, 'mlp_num_inner', 0),
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

    def training_step(self, batch, batch_idx):
        """Training step"""
        out = self.model(batch["img"])
        if self.num_classes > 0:
            acc = self.train_acc(out, batch["class"].long())
            self.log_dict(acc, sync_dist=False, on_step=True, on_epoch=False, prog_bar=True)
            loss = self.criterion(out, batch["class"].long())
        else:
            loss = torch.tensor(0.0)
        # compute distillation loss
        distillation_loss = self.distillation_loss_fn(batch["img"])

        supervised_loss = (1 - self.distill_weight) * loss
        distill_loss = self.distill_weight * distillation_loss * 100

        if torch.isnan(supervised_loss):
            supervised_loss = torch.tensor(0.0)

        if torch.isnan(distill_loss):
            distill_loss = torch.tensor(0.0)

        total_loss = supervised_loss + distill_loss
        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[-1],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            "supervised_loss",
            supervised_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "distillation_loss",
            distill_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        return {"loss": total_loss}

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["total_loss_epoch"].item()
        self.train_acc.reset()
        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING,
        )

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        out = self.model(batch["img"])
        if self.num_classes > 0:
            loss = self.criterion(out, batch["class"].long())
            self.valid_acc.update(out, batch["class"].long())
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )
        else:
            loss = torch.tensor(0.0).to(out.device)

        # compute distillation loss
        distillation_loss = self.distillation_loss_fn(batch["img"])
        self.log(
            "distillation_loss",
            distillation_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        loss += distillation_loss
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        # FLUSHING VALIDATION EPOCH METRICS
        # scores, mean_scores = self._collect_epoch_states()  # logs all evaluation metrics
        # self.log("val_acc", scores['acc'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.num_classes > 0:
            acc = self.valid_acc.compute()
            self.log_dict(acc, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)
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

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint but ignore the teacher weights."""
        keys_to_pop = [
            key for key in checkpoint["state_dict"].keys() if key.startswith("teacher")
        ]
        for key in keys_to_pop:
            checkpoint["state_dict"].pop(key)
        checkpoint["tao_model"] = "classification"
