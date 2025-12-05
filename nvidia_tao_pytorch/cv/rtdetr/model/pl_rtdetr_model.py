# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

""" Main PTL model file for rt-detr. """

import re
from typing import Sequence
import omegaconf
from omegaconf import OmegaConf
import random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from fairscale.optim import OSS
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.rtdetr.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.rtdetr.model.postprocess import RTDETRPostProcess
from nvidia_tao_pytorch.cv.rtdetr.model.criterion import SetCriterion

from nvidia_tao_pytorch.cv.deformable_detr.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import save_inference_prediction, threshold_predictions
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator


# pylint:disable=too-many-ancestors
class RTDETRPlModel(TAOLightningModule):
    """PTL module for RT-DETR Object Detection Model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for RT-DETR Model."""
        super().__init__(experiment_spec)
        self.eval_class_ids = self.dataset_config["eval_class_ids"]
        self.dataset_type = self.dataset_config["dataset_type"]
        if self.dataset_type not in ("serialized", "default"):
            raise ValueError(f"{self.dataset_type} is not supported. Only serialized and default are supported.")

        # init the model
        self._build_model(export)
        self._build_criterion()

        self.checkpoint_filename = 'rtdetr_model'

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True
        )

        resume_ckpt = self.experiment_spec["train"]["resume_training_checkpoint_path"] or get_latest_checkpoint(results_dir)
        resumed_epoch = 0
        if resume_ckpt:
            resumed_epoch = re.search('epoch_(\\d+)', resume_ckpt)
            if resumed_epoch:
                resumed_epoch = int(resumed_epoch.group(1))
        else:
            resumed_epoch = 0
        status_logger_callback.epoch_counter = resumed_epoch + 1
        callbacks.append(status_logger_callback)

        if self.experiment_spec["train"]["enable_ema"]:
            # Apply Exponential Moving Average Callback
            ema_callback = EMA(
                **self.experiment_spec["train"]["ema"]
            )
            ckpt_func = EMAModelCheckpoint
            callbacks.append(ema_callback)
        else:
            ckpt_func = ModelCheckpoint

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError("checkpoint_filename not set in __init__() of model")
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = ckpt_func(every_n_epochs=checkpoint_interval,
                                        dirpath=results_dir,
                                        save_on_train_epoch_end=True,
                                        monitor=None,
                                        save_top_k=-1,
                                        save_last='link',
                                        filename='model_{epoch:03d}',
                                        enable_version_counter=False)
        callbacks.append(checkpoint_callback)
        return callbacks

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

        # freeze modules
        if self.experiment_spec["train"]["freeze"]:
            freezed_modules = []
            skipped_modules = []
            for module in self.experiment_spec["train"]["freeze"]:
                try:
                    module_to_freeze = rgetattr(self.model.model, module)
                    for p in module_to_freeze.parameters():
                        p.requires_grad = False
                    freezed_modules.append(module)
                except AttributeError:
                    skipped_modules.append(module)
            if freezed_modules:
                status_logging.get_status_logger().write(
                    message=f"Freezed module {freezed_modules}",
                    status_level=status_logging.Status.RUNNING,
                    verbosity_level=status_logging.Verbosity.INFO)
            if skipped_modules:
                status_logging.get_status_logger().write(
                    message=f"module {skipped_modules} not found. Skipped freezing",
                    status_level=status_logging.Status.SKIPPED,
                    verbosity_level=status_logging.Verbosity.WARNING)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.matcher = HungarianMatcher(cost_class=self.model_config["class_cost"],
                                        cost_bbox=self.model_config["bbox_cost"],
                                        cost_giou=self.model_config["giou_cost"])

        self.weight_dict = {'loss_vfl': self.model_config["vfl_loss_coef"],
                            'loss_bbox': self.model_config["bbox_loss_coef"],
                            'loss_giou': self.model_config["giou_loss_coef"]}

        self.criterion = SetCriterion(self.matcher,
                                      weight_dict=self.weight_dict,
                                      losses=self.model_config["loss_types"],
                                      alpha=self.model_config["alpha"], gamma=self.model_config["gamma"],
                                      num_classes=self.dataset_config["num_classes"])
        self.box_processors = RTDETRPostProcess(num_select=self.model_config["num_select"],
                                                remap_mscoco_category=self.dataset_config["remap_mscoco_category"])

    def configure_optimizers(self):
        """Configure optimizers for training."""
        self.train_config = self.experiment_spec.train
        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if not re.match('^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', n) and
                     not re.match('^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', n) and
                     not re.match('backbone', n) and
                     p.requires_grad],
                "lr": self.train_config.optim.lr,
                "weight_decay": self.train_config.optim.weight_decay
            },
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if re.match('^(?=.*encoder(?=.*bias|.*norm.*weight)).*$', n) and
                     p.requires_grad],
                "lr": self.train_config.optim.lr,
                "weight_decay": 0
            },
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if re.match('^(?=.*decoder(?=.*bias|.*norm.*weight)).*$', n) and
                     p.requires_grad],
                "lr": self.train_config.optim.lr,
                "weight_decay": 0
            },
            {
                "params": [p for n, p in self.model.named_parameters() if re.match('backbone', n) and p.requires_grad],
                "lr": self.train_config.optim.lr_backbone,
                "weight_decay": self.train_config.optim.weight_decay
            }
        ]

        if self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(params=param_dicts,
                                               lr=self.train_config.optim.lr)  # 1e-4
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

        if scheduler_type == "MultiStep":
            lr_scheduler = MultiStepLR(optimizer=optim,
                                       milestones=self.train_config['optim']["lr_steps"],
                                       gamma=self.train_config['optim']["lr_decay"])
        elif scheduler_type == "StepLR":
            lr_scheduler = StepLR(optimizer=optim,
                                  step_size=self.train_config['optim']["lr_step_size"],
                                  gamma=self.train_config['optim']["lr_decay"])
        else:
            raise NotImplementedError("LR Scheduler {} is not implemented".format(scheduler_type))

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['monitor_name']
        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, targets, _ = batch
        batch_size = data.shape[0]
        multi_scales = self.experiment_spec.dataset.augmentation.multi_scales
        if not self.experiment_spec.model.frozen_fm.enabled and multi_scales:
            sz = random.choice(multi_scales)
            # Convert omegaconf listconfig when reading lists from the experiment config.
            if isinstance(sz, omegaconf.listconfig.ListConfig):
                sz = OmegaConf.to_object(sz)
            if isinstance(sz, int):
                # square resize
                data = F.interpolate(data, size=[sz, sz])
            elif isinstance(sz, (list, tuple)):
                data = F.interpolate(data, size=sz)
            else:
                raise TypeError(f"{sz} is {type(sz)}. Need to pass int / list / tuple for multi_scale")

        outputs = self.model(data, targets)
        # loss
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict.values())

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_vfl", loss_dict['loss_vfl'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=False)
        lrs = [param_group['lr'] for param_group in self.optimizers().optimizer.param_groups]
        self.log("lr", lrs[0], on_step=True, on_epoch=False, prog_bar=True)

        return losses

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
        """
        Validation epoch start.
        Reset coco evaluator for each epoch.
        """
        self.val_coco_evaluator = CocoEvaluator(self.trainer.datamodule.val_dataset.coco, iou_types=['bbox'], eval_class_ids=None)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, targets, image_names = batch
        outputs = self.model(data, targets)
        batch_size = data.shape[0]

        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict.values())

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.val_coco_evaluator.update(res)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_loss_vfl", loss_dict['loss_vfl'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=False)

        return losses

    def on_validation_epoch_end(self):
        """
        Validation epoch end.
        Compute mAP at the end of epoch.
        """
        self.val_coco_evaluator.synchronize_between_processes()
        self.val_coco_evaluator.overall_accumulate()
        self.val_coco_evaluator.overall_summarize(is_print=False)
        mAP = self.val_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.val_coco_evaluator.coco_eval['bbox'].stats[1]
        if self.trainer.is_global_zero:
            logging.info("Validation mAP : {}".format(mAP))
            logging.info("Validation mAP50 : {}".format(mAP50))

        self.log("current_epoch", self.current_epoch, sync_dist=True)
        self.log("val_mAP", mAP, sync_dist=True)
        self.log("val_mAP50", mAP50, sync_dist=True)

        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_mAP"] = str(mAP)
            self.status_logging_dict["val_mAP50"] = str(mAP50)
            self.status_logging_dict["val_loss"] = average_val_loss
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
        # Clear memory
        self.val_coco_evaluator = None
        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self) -> None:
        """
        Test epoch start.
        Reset coco evaluator at start.
        """
        self.test_coco_evaluator = CocoEvaluator(self.trainer.datamodule.test_dataset.coco, iou_types=['bbox'], eval_class_ids=None)

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate."""
        data, targets, image_names = batch
        outputs = self.model(data)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)

        if self.experiment_spec.evaluate.conf_threshold > 0:
            filtered_res = threshold_predictions(results, self.experiment_spec.evaluate.conf_threshold)
        else:
            filtered_res = results
        res = {target['image_id'].item(): output for target, output in zip(targets, filtered_res)}
        self.test_coco_evaluator.update(res)

    def on_test_epoch_end(self):
        """
        Test epoch end.
        Compute mAP at the end of epoch.
        """
        self.test_coco_evaluator.synchronize_between_processes()
        self.test_coco_evaluator.overall_accumulate()
        self.test_coco_evaluator.overall_summarize(is_print=False)
        mAP = self.test_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.test_coco_evaluator.coco_eval['bbox'].stats[1]
        self.log("test_mAP", mAP, rank_zero_only=True)
        self.log("test_mAP50", mAP50, rank_zero_only=True)

        self.status_logging_dict = {}
        self.status_logging_dict["test_mAP"] = str(mAP)
        self.status_logging_dict["test_mAP50"] = str(mAP50)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference."""
        data, targets, image_names = batch
        outputs = self.model(data)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        pred_results = self.box_processors(outputs, orig_target_sizes, image_names)
        return pred_results

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Predict batch end.
        Save the result inferences at the end of batch.
        """
        output_dir = self.experiment_spec.results_dir
        label_map = self.trainer.datamodule.pred_dataset.label_map
        color_map = self.experiment_spec.inference.color_map
        conf_threshold = self.experiment_spec.inference.conf_threshold
        is_internal = self.experiment_spec.inference.is_internal
        save_inference_prediction(outputs, output_dir, conf_threshold, label_map, color_map, is_internal)

    def forward(self, x):
        """Forward of the deformable detr model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "rtdetr"
