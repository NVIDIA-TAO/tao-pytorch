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

"""Distiller module for DINO model"""

import copy
from contextlib import ExitStack
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR, StepLR
from fairscale.optim import OSS

from nvidia_tao_pytorch.core.distillation.distiller import Distiller
from nvidia_tao_pytorch.core.distillation.utils import Binding, CaptureModule
from nvidia_tao_pytorch.core.distillation.losses import WeightedCriterion, LPCriterion, KLDivCriterion, FeatureMapCriterion
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import CoCoDataMerge
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import PostProcess
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco import COCO
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr

from nvidia_tao_pytorch.cv.dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.dino.model.fan import fan_model_dict
from nvidia_tao_pytorch.cv.dino.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.dino.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.dino.model.utils import dino_parser, ptm_adapter
from nvidia_tao_pytorch.cv.dino.model.vision_transformer.transformer_modules import get_vit_lr_decay_rate


class DINODistiller(Distiller):
    """DINO Distiller class"""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        self.supported_teacher_arch = list(fan_model_dict.keys())

        # Restricting students to only some
        self.supported_student_arch = ['fan_small', 'fan_tiny',
                                       'resnet_34', 'resnet_50',
                                       'efficientvit_b0', 'efficientvit_b1',
                                       'efficientvit_b2', 'efficientvit_b3']

        # init the model
        super().__init__(experiment_spec, export)

        self.checkpoint_filename = 'dino_model'

    def _setup_bindings(self):
        """Setup bindings to be captured during training for distillation."""
        pass

    def prepare_channel_mapper(self, num_channels, num_feature_levels, hidden_dim, two_stage_type):
        """Create Channel Mapper style for DETR-based model.

        Args:
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            two_stage_type (str): type of two stage in DINO.
            hidden_dim (int): size of the hidden dimension.

        Returns:
            nn.ModuleList of input projection.
        """
        if num_feature_levels > 1:
            num_backbone_outs = len(num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            return nn.ModuleList(input_proj_list)

        assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels[-1], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])

    def _build_model(self, export=False):
        """Internal function to build the model."""
        # Build the teacher config
        teacher_cfg = copy.deepcopy(self.experiment_spec)
        teacher_cfg.model = self.experiment_spec.distill.teacher

        # Check if supported teacher arch
        assert teacher_cfg.model.backbone in self.supported_teacher_arch, f"Teacher arch {teacher_cfg.model.backbone} not supported.\
            Supported archs: {self.supported_teacher_arch}"

        # Check if supported student arch
        assert self.experiment_spec.model.backbone in self.supported_student_arch, f"Student arch {self.experiment_spec.model.backbone} not supported.\
            Supported archs: {self.supported_student_arch}"

        # Build the teacher model
        self.teacher = build_model(experiment_config=teacher_cfg, export=export)

        if self.experiment_spec.distill.pretrained_teacher_model_path:
            current_model_dict = self.teacher.model.state_dict()
            checkpoint = load_pretrained_weights(
                self.experiment_spec.distill.pretrained_teacher_model_path,
                parser=dino_parser,
                ptm_adapter=ptm_adapter
            )
            new_checkpoint = {}
            for k, k_ckpt in zip(sorted(current_model_dict.keys()), sorted(checkpoint.keys())):

                v = checkpoint[k_ckpt]
                # Handle PTL format
                k = k.replace("model.model.", "model.")
                if v.size() == current_model_dict[k].size():
                    new_checkpoint[k] = v
                else:
                    # Skip layers that mismatch
                    logging.info(
                        "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                        k, list(v.size()), list(current_model_dict[k].size())
                    )
                    new_checkpoint[k] = current_model_dict[k]
            # Load pretrained weights
            m = self.teacher.model.load_state_dict(new_checkpoint, strict=True)
            logging.info("Loaded teacher weights from %s \n %s", self.experiment_spec.distill.pretrained_teacher_model_path, m)

        # Build the student model
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

        if self.experiment_spec.model.backbone.startswith('resnet'):
            self.experiment_spec.model.backbone = self.experiment_spec.model.backbone.replace('_', '')

        self.teacher.eval()
        self.model.train()

        # Freeze teacher #TODO: In future we can add options to freeze/unfreeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

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
        self.matcher = HungarianMatcher(cost_class=self.model_config["cls_loss_coef"],
                                        cost_bbox=self.model_config["bbox_loss_coef"],
                                        cost_giou=self.model_config["giou_loss_coef"])
        weight_dict = {'loss_ce': self.model_config["cls_loss_coef"],
                       'loss_bbox': self.model_config["bbox_loss_coef"],
                       'loss_giou': self.model_config["giou_loss_coef"]}
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        # for de-noising training
        if self.model_config['use_dn']:
            weight_dict['loss_ce_dn'] = self.model_config["cls_loss_coef"]
            weight_dict['loss_bbox_dn'] = self.model_config["bbox_loss_coef"]
            weight_dict['loss_giou_dn'] = self.model_config["giou_loss_coef"]
        clean_weight_dict = copy.deepcopy(weight_dict)

        if self.model_config["aux_loss"]:
            aux_weight_dict = {}
            for i in range(self.model_config["dec_layers"] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if self.model_config['two_stage_type'] != 'no':
            interm_weight_dict = {}
            _coeff_weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 1.0 if not self.model_config['no_interm_box_loss'] else 0.0,
                'loss_giou': 1.0 if not self.model_config['no_interm_box_loss'] else 0.0,
            }
            interm_weight_dict.update({f'{k}_interm': v * self.model_config['interm_loss_coef'] * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)

        weight_dict['loss_distillation'] = self.model_config['distillation_loss_coef']

        self.student_binding_captures = list()
        self.teacher_binding_captures = list()

        self.criterions = {
            'L1': LPCriterion(p=1),
            'L2': LPCriterion(p=2),
            'KL': KLDivCriterion(),
        }
        self.bindings = []
        self.binding_coef = dict()
        for binding in self.experiment_spec.distill.bindings:
            if binding.criterion in self.criterions:
                criterion = self.criterions[binding.criterion]
                self.binding_coef[binding.student_module_name] = binding.weight
                if 'model.backbone.0.body' in binding.student_module_name:
                    self.bindings.append(Binding(student=binding.student_module_name,
                                                 teacher=binding.teacher_module_name,
                                                 criterion=WeightedCriterion(binding.weight, FeatureMapCriterion(criterion)),
                                                 loss_coef=binding.weight
                                                 )
                                         )
                elif 'pred_logits' in binding.student_module_name:
                    self.bindings.append(Binding(student=binding.student_module_name,
                                                 teacher=binding.teacher_module_name,
                                                 criterion=WeightedCriterion(binding.weight, criterion),
                                                 loss_coef=binding.weight
                                                 )
                                         )
                elif 'pred_boxes' in binding.student_module_name:
                    self.bindings.append(Binding(student=binding.student_module_name,
                                                 teacher=binding.teacher_module_name,
                                                 criterion=WeightedCriterion(binding.weight, criterion),
                                                 loss_coef=binding.weight
                                                 )
                                         )
                else:
                    self.bindings.append(Binding(student=binding.student_module_name,
                                                 teacher=binding.teacher_module_name,
                                                 criterion=WeightedCriterion(binding.weight, criterion),
                                                 loss_coef=binding.weight
                                                 )
                                         )
            else:
                raise NotImplementedError(f"Criterion {binding.criterion} not implemented, supported criterions: {self.criterions.keys()}")

        for binding in self.bindings:
            if binding.student in ['pred_logits', 'pred_boxes']:
                continue
            student_module = next(module for name, module in self.model.named_modules() if name == binding.student)
            teacher_module = next(module for name, module in self.teacher.named_modules() if name == binding.teacher)

            self.student_binding_captures.append(CaptureModule(student_module))
            self.teacher_binding_captures.append(CaptureModule(teacher_module))

        self.weight_dict = copy.deepcopy(weight_dict)

        self.criterion = SetCriterion(self.dataset_config["num_classes"], matcher=self.matcher,
                                      losses=self.model_config['loss_types'], focal_alpha=self.model_config["focal_alpha"])

        # nms_iou_threshold is always 0 in original DINO
        self.box_processors = PostProcess(num_select=self.model_config['num_select'])

    def configure_optimizers(self):
        """Configure optimizers for training."""
        self.train_config = self.experiment_spec.train
        param_dicts = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" not in n:
                param_dicts.append({"params": [p]})
            if "backbone" in n and self.model_config.backbone.startswith("vit"):
                # ViT style layer-wise learning rate
                # Note that ViT is very sensitive to this layer-wise lr decay rate
                # https://github.com/czczup/ViT-Adapter/tree/dinov2/detection/configs/mask_rcnn/dinov2#results-and-models
                num_layers = self.model.model.backbone[0].body.depth
                scale = get_vit_lr_decay_rate(n, lr_decay_rate=self.train_config.optim.layer_decay_rate, num_layers=num_layers)
                scaled_lr = self.train_config.optim.lr * scale
                param_dicts.append({"params": [p], "lr": scaled_lr})
            if "backbone" in n and not self.model_config.backbone.startswith("vit"):
                param_dicts.append({"params": [p], "lr": self.train_config.optim.lr_backbone})

        if self.train_config.optim.optimizer == 'SGD':
            base_optimizer = torch.optim.SGD(params=param_dicts,
                                             lr=self.train_config.optim.lr,
                                             momentum=self.train_config.optim.momentum,
                                             weight_decay=self.train_config.optim.weight_decay)
        elif self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(params=param_dicts,
                                               lr=self.train_config.optim.lr,
                                               weight_decay=self.train_config.optim.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.train_config.optim.optimizer} is not implemented")

        if self.train_config.distributed_strategy == "fsdp":
            # Override force_broadcast_object=False in PTL
            optim = OSS(params=base_optimizer.param_groups, optim=type(base_optimizer), force_broadcast_object=True, **base_optimizer.defaults)
        else:
            optim = base_optimizer

        optim_dict = {}
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config.optim.lr_scheduler

        if scheduler_type == "MultiStep":
            lr_scheduler = MultiStepLR(optimizer=optim,
                                       milestones=self.train_config.optim.lr_steps,
                                       gamma=self.train_config.optim.lr_decay)
        elif scheduler_type == "StepLR":
            lr_scheduler = StepLR(optimizer=optim,
                                  step_size=self.train_config.optim.lr_step_size,
                                  gamma=self.train_config.optim.lr_decay)
        else:
            raise NotImplementedError("LR Scheduler {} is not implemented".format(scheduler_type))

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config.optim.monitor_name
        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step"""
        data, targets, _ = batch
        batch_size = data.shape[0]

        with ExitStack() as stack:
            for teacher_capture, student_capture in zip(self.teacher_binding_captures, self.student_binding_captures):
                stack.enter_context(teacher_capture)
                stack.enter_context(student_capture)

            # Distillation
            with torch.no_grad():
                if self.model_config['use_dn']:
                    teacher_outputs = self.teacher(data, targets)
                else:
                    teacher_outputs = self.teacher(data)

            if self.model_config['use_dn']:
                outputs = self.model(data, targets)
            else:
                outputs = self.model(data)

            # Compute the loss
            loss_dict = self.criterion(outputs, targets)

            # Compute the distillation loss
            distillation_loss = 0.0
            for binding_index, binding in enumerate(self.bindings):
                if binding.student == 'pred_logits':
                    binding_loss = binding.criterion(outputs['pred_logits'], teacher_outputs['pred_logits']) * self.binding_coef[binding.student]
                elif binding.student == 'pred_boxes':
                    binding_loss = binding.criterion(outputs['pred_boxes'], teacher_outputs['pred_boxes']) * self.binding_coef[binding.student]
                elif binding.student == 'model.backbone.0.body':
                    # Special case to use output of input_proj to distill feature maps
                    binding_loss = binding.criterion(outputs['srcs'], teacher_outputs['srcs']) * self.binding_coef[binding.student]
                else:
                    binding_loss = binding.criterion(self.student_binding_captures[binding_index].output, self.teacher_binding_captures[binding_index].output) * self.binding_coef[binding.student]
                distillation_loss += binding_loss

            loss_dict['loss_distillation'] = distillation_loss

            losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

            self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log("train_class_error", loss_dict['class_error'], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train_loss_distillation", loss_dict['loss_distillation'], on_step=True, on_epoch=False, prog_bar=False)

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
        if self.dataset_type == "serialized":
            # Load from scratch since COCO object is not instantiated for SerializedDatasetFromList
            coco_lists = []
            for source in self.dataset_config["val_data_sources"]:
                with open(source["json_file"], "r") as f:
                    tmp = json.load(f)
                coco_lists.append(COCO(tmp))
            coco = COCO(CoCoDataMerge(coco_lists))
            self.val_coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)
        else:
            self.val_coco_evaluator = CocoEvaluator(self.trainer.datamodule.val_dataset.coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, targets, image_names = batch
        batch_size = data.shape[0]
        if self.model_config['use_dn']:
            outputs = self.model(data, targets)
        else:
            outputs = self.model(data)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.val_coco_evaluator.update(res)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_class_error", loss_dict['class_error'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=False, prog_bar=False)
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
            logging.info("\n Validation mAP : {}\n".format(mAP))
            logging.info("\n Validation mAP50 : {}\n".format(mAP50))

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
        """Forward of the dino model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint but ignore the teacher weights."""
        keys_to_pop = [key for key in checkpoint['state_dict'].keys() if key.startswith('teacher')]
        for key in keys_to_pop:
            checkpoint['state_dict'].pop(key)
