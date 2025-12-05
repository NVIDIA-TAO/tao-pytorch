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

"""Distiller module for RTDETR model"""
import re
import copy
from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR, StepLR
from fairscale.optim import OSS
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.distillation.distiller import Distiller
from nvidia_tao_pytorch.core.distillation.utils import Binding, CaptureModule
from nvidia_tao_pytorch.core.distillation.losses import WeightedCriterion, LPCriterion, KLDivCriterion, FeatureMapCriterion

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.deformable_detr.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr
from nvidia_tao_pytorch.cv.deformable_detr.utils.box_ops import box_cxcywh_to_xyxy

from nvidia_tao_pytorch.cv.rtdetr.model.build_nn_model import build_model
# from nvidia_tao_pytorch.cv.rtdetr.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.rtdetr.model.postprocess import RTDETRPostProcess
from nvidia_tao_pytorch.cv.rtdetr.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.rtdetr.model.utils import rtdetr_parser, ptm_adapter
from nvidia_tao_pytorch.cv.rtdetr.utils.misc import bbox_overlaps


class RtdetrDistiller(Distiller):
    """Rtdetr Distiller class"""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        super().__init__(experiment_spec, export)
        self.checkpoint_filename = 'rtdetr_model'

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
        )

        resume_ckpt = self.experiment_spec["train"]["resume_training_checkpoint_path"] or get_latest_checkpoint(results_dir)
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

    def _setup_bindings(self):
        """Setup bindings to be captured during training for distillation."""
        pass

    def _build_model(self, export=False):
        """Internal function to build the model."""
        # Build the teacher config
        teacher_cfg = copy.deepcopy(self.experiment_spec)
        teacher_cfg.model = self.experiment_spec.distill.teacher

        # Build the teacher model
        self.teacher = build_model(experiment_config=teacher_cfg, export=export)
        # Build the student model
        self.model = build_model(experiment_config=self.experiment_spec, export=export)
        # Load teacher's encoder and/or decoder weight in student
        if self.experiment_spec.distill.pretrained_teacher_model_path:
            teacher_model_dict = self.teacher.model.state_dict()
            student_model_dict = self.model.model.state_dict()
            checkpoint = load_pretrained_weights(
                self.experiment_spec.distill.pretrained_teacher_model_path,
                ptm_adapter=ptm_adapter,
                parser=rtdetr_parser
            )
            new_checkpoint = {}
            kv_for_student = {}

            for k in sorted(teacher_model_dict.keys()):

                if "feature_criterion" in k:
                    continue
                k_ckpt = "model." + k
                v = checkpoint.get(k_ckpt, None)
                assert v is not None, f"{k_ckpt} doesn't exist in the pretrained teacher model."
                # Handle PTL format
                # k = k.replace("model.model.", "model.")
                if v.size() == teacher_model_dict[k].size():
                    new_checkpoint[k] = v
                else:
                    # Skip layers that mismatch
                    logging.info(
                        "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                        k, list(v.size()), list(teacher_model_dict[k].size())
                    )
                    new_checkpoint[k] = teacher_model_dict[k]

                if 'backbone' not in k:
                    if v.size() == student_model_dict[k].size():
                        if 'encoder' in k or 'decoder' in k:
                            kv_for_student[k] = v
                        else:
                            logging.info(f"skip layer in the backbone: {k}")
                    else:
                        # Skip layers that mismatch
                        logging.info(
                            "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                            k, list(v.size()), list(teacher_model_dict[k].size())
                        )
                        kv_for_student[k] = student_model_dict[k]
            # Load pretrained weights
            self.teacher.model.load_state_dict(new_checkpoint, strict=False)
            if self.experiment_spec.model.load_teacher_enc_dec:
                self.model.model.load_state_dict(kv_for_student, strict=False)
        # end of loading
        if self.experiment_spec.model.backbone.startswith('resnet'):
            self.experiment_spec.model.backbone = self.experiment_spec.model.backbone.replace('_', '')

        self.teacher.eval()
        self.model.train()

        # freeze student decoder
        # for n, param in self.model.named_parameters():
        #     if 'decoder' in n:
        #         param.requires_grad = False

        # Freeze teacher
        for n, param in self.teacher.named_parameters():
            if 'feature_criterion' not in n:
                param.requires_grad = False

        for module in self.teacher.modules():
            if any([isinstance(module, nn.BatchNorm2d),
                    isinstance(module, nn.LayerNorm),
                    isinstance(module, nn.Dropout)]):
                module.eval()

        # Setup bindings to be captured during training for distillation
        self._setup_bindings()

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
        self.matcher = HungarianMatcher(cost_class=self.model_config["vfl_loss_coef"],
                                        cost_bbox=self.model_config["bbox_loss_coef"],
                                        cost_giou=self.model_config["giou_loss_coef"])

        self.weight_dict = {'loss_vfl': self.model_config["vfl_loss_coef"],
                            'loss_bbox': self.model_config["bbox_loss_coef"],
                            'loss_giou': self.model_config["giou_loss_coef"],
                            'loss_distillation': self.model_config['distillation_loss_coef']}

        self.criterion = SetCriterion(self.matcher,
                                      weight_dict=self.weight_dict,
                                      losses=self.model_config["loss_types"],
                                      alpha=self.model_config["alpha"], gamma=self.model_config["gamma"],
                                      num_classes=self.dataset_config["num_classes"])

        self.box_processors = RTDETRPostProcess(num_select=self.model_config["num_select"],
                                                remap_mscoco_category=self.dataset_config["remap_mscoco_category"])

        self.student_binding_captures = list()
        self.teacher_binding_captures = list()

        self.criterions = {
            'L1': LPCriterion(p=1),
            'L2': LPCriterion(p=2),
            'KL': KLDivCriterion(),
            'IOU': 'IOU',
        }
        self.bindings = []
        self.binding_coef = dict()
        for binding in self.experiment_spec.distill.bindings:
            if binding.criterion in self.criterions:
                criterion = self.criterions[binding.criterion]
                self.binding_coef[binding.student_module_name] = binding.weight
                if criterion == 'IOU':
                    assert binding.student_module_name in ['srcs', 'dsrcs'], \
                        "module name must be `srcs` or `dsrcs` when criterion is `IOU`."
                    assert binding.student_module_name == binding.teacher_module_name, \
                        "module name must be the same from teacher and student."
                    self.bindings.append(
                        Binding(student=binding.student_module_name,
                                teacher=binding.teacher_module_name,
                                criterion=criterion,
                                loss_coef=binding.weight
                                ))
                else:
                    self.bindings.append(
                        Binding(student=binding.student_module_name,
                                teacher=binding.teacher_module_name,
                                criterion=WeightedCriterion(binding.weight, FeatureMapCriterion(criterion)),
                                loss_coef=binding.weight
                                ))
            else:
                raise NotImplementedError(f"Criterion {binding.criterion} not implemented, supported criterions: {self.criterions.keys()}")

        for binding in self.bindings:
            if binding.criterion == 'IOU':
                continue
            student_module = next(module for name, module in self.model.named_modules() if name == binding.student)
            teacher_module = next(module for name, module in self.teacher.named_modules() if name == binding.teacher)

            self.student_binding_captures.append(CaptureModule(student_module))
            self.teacher_binding_captures.append(CaptureModule(teacher_module))

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
        """Training step"""
        data, targets, _ = batch
        batch_size, _, input_height, input_width = data.shape

        gt_boxes = [t['boxes'] for t in targets]

        teacher_outputs = self.teacher(data, targets)
        outputs = self.model(data, targets)

        # Compute the loss
        loss_dict = self.criterion(outputs, targets)

        # Compute the distillation loss
        distillation_loss = 0.0
        for binding_index, binding in enumerate(self.bindings):
            if binding.criterion == 'IOU':
                key = binding.student
                factor = torch.Tensor([input_width, input_height, input_width, input_height])
                enc_topk_bboxes = teacher_outputs['enc_topk_bboxes']
                scores_per_img = teacher_outputs['scores_per_img']
                factor = factor.to(enc_topk_bboxes.device)
                cls_iou_scores = []
                for i in range(batch_size):
                    bbox_pred_per_img = enc_topk_bboxes[i]
                    bbox_pred_per_img = box_cxcywh_to_xyxy(bbox_pred_per_img) * factor
                    max_cls_score_per_img = scores_per_img[i]
                    if len(gt_boxes[i]) == 0:
                        max_cls_iou_score_per_img = max_cls_score_per_img
                    else:
                        max_iou_score_per_img = torch.max(bbox_overlaps(bbox_pred_per_img, gt_boxes[i]), dim=-1)[0]
                        max_cls_iou_score_per_img = max_cls_score_per_img * max_iou_score_per_img
                    cls_iou_scores.append(max_cls_iou_score_per_img)
                cls_iou_scores = torch.stack(cls_iou_scores, dim=0)
                loss_mse = nn.MSELoss(reduction='mean')
                c_querys = torch.nn.functional.normalize(teacher_outputs['obj_queries'], dim=-1)

                # option 2:
                # cls_iou_scores = teacher_outputs['scores_per_img']  # b, 300
                binding_loss = 0
                for feat_t, feat_s in zip(teacher_outputs[key], outputs[key]):
                    c_feats = torch.nn.functional.normalize(feat_t, dim=1)
                    mat = torch.einsum('bnc,bchw->bnhw', [c_querys, c_feats])
                    mask = torch.einsum('bnhw,bn->bhw', [mat, cls_iou_scores]).clamp(min=1e-2)

                    max_shu = torch.max(mask.flatten(1, 2), dim=-1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)  # [2, 1, 1]
                    mask = (mask / max_shu).unsqueeze(dim=1)
                    binding_loss += loss_mse(mask * feat_t, mask * feat_s) * self.binding_coef[binding.student]
            else:
                binding_loss = binding.criterion(
                    self.student_binding_captures[binding_index].output,
                    self.teacher_binding_captures[binding_index].output
                ) * self.binding_coef[binding.student]
            distillation_loss += binding_loss

        loss_dict['loss_distillation'] = distillation_loss
        losses = sum(loss_dict.values())

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_vfl", loss_dict['loss_vfl'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_distillation", loss_dict['loss_distillation'], on_step=True, on_epoch=False, prog_bar=True)
        lrs = [param_group['lr'] for param_group in self.optimizers().optimizer.param_groups]
        self.log("lr", lrs[0], on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': losses}

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train and Val metrics generated.",
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
        self.log("val_loss_vfl", loss_dict['loss_vfl'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=True)

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
            logging.info("\n Validation mAP : %s\n" % mAP)
            logging.info("\n Validation mAP50 : %s\n" % mAP50)

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

    def forward(self, x):
        """Forward of the rtdetr model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint but ignore the teacher weights."""
        keys_to_pop = [key for key in checkpoint['state_dict'].keys() if key.startswith('teacher')]
        for key in keys_to_pop:
            checkpoint['state_dict'].pop(key)
        checkpoint["tao_model"] = "rtdetr"
