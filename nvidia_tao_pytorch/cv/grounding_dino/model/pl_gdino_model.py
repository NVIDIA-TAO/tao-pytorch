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

""" Main PTL model file for Grounding DINO. """

import copy
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr, match_name_keywords
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import save_inference_prediction, threshold_predictions
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator

from nvidia_tao_pytorch.cv.grounding_dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.grounding_dino.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.grounding_dino.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.grounding_dino.model.post_process import PostProcess
from nvidia_tao_pytorch.cv.grounding_dino.utils.get_tokenlizer import get_tokenlizer
from nvidia_tao_pytorch.cv.grounding_dino.model.bertwraper import (
    generate_masks_with_special_tokens_and_transfer_map
)


# pylint:disable=too-many-ancestors
class GDINOPlModel(TAOLightningModule):
    """PTL module for GDINO Object Detection Model."""

    def __init__(self, experiment_spec, cap_lists, export=False):
        """Init training for DINO Model."""
        super().__init__(experiment_spec)
        self.cap_lists = cap_lists
        self.max_text_len = self.model_config.max_text_len
        self.eval_class_ids = self.dataset_config.eval_class_ids

        # To disable warnings from HF tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # init the model
        self._build_model(export)
        self._build_criterion()

        self.checkpoint_filename = 'gdino_model'

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)
        self.tokenizer = get_tokenlizer(self.model_config.text_encoder_type)

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
        self.matcher = HungarianMatcher(cost_class=self.model_config["set_cost_class"],
                                        cost_bbox=self.model_config["set_cost_bbox"],
                                        cost_giou=self.model_config["set_cost_giou"])
        weight_dict = {'loss_ce': self.model_config["cls_loss_coef"],
                       'loss_bbox': self.model_config["bbox_loss_coef"],
                       'loss_giou': self.model_config["giou_loss_coef"]}
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)
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

        self.weight_dict = copy.deepcopy(weight_dict)

        self.criterion = SetCriterion(matcher=self.matcher,
                                      losses=self.model_config['loss_types'],
                                      focal_alpha=self.model_config["focal_alpha"],
                                      focal_gamma=self.model_config["focal_gamma"])

        self.box_processors = PostProcess(self.model.model.tokenizer,
                                          cat_list=self.cap_lists,
                                          num_select=self.model_config['num_select'])

    def configure_optimizers(self):
        """Configure optimizers for training."""
        self.train_config = self.experiment_spec.train
        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if not match_name_keywords(n, self.model_config["backbone_names"]) and
                     not match_name_keywords(n, self.model_config["linear_proj_names"]) and
                     p.requires_grad],
                "lr": self.train_config['optim']['lr'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, self.model_config["backbone_names"]) and p.requires_grad],
                "lr": self.train_config['optim']['lr_backbone'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, self.model_config["linear_proj_names"]) and p.requires_grad],
                "lr": self.train_config['optim']['lr'] * self.train_config['optim']['lr_linear_proj_mult'],
            }
        ]

        if self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(params=param_dicts,
                                               lr=self.train_config.optim.lr,
                                               weight_decay=self.train_config.optim.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.train_config.optim.optimizer} is not implemented")

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

    def tokenize_captions(self, captions, pad_to_max=False):
        """Tokenize the captions through model tokeninzer."""
        if pad_to_max:
            padding = "max_length"
        else:
            padding = "longest"

        tokenized = self.tokenizer(captions, padding=padding, return_tensors="pt").to(
            self.device
        )
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.model.model.specical_tokens, self.tokenizer)

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len]

            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        return tokenized, one_hot_token, position_ids, text_self_attention_masks

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, targets = batch
        batch_size = data.shape[0]

        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]

        (
            tokenized,
            one_hot_token,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(samples=data,
                             targets=targets if self.model_config['use_dn'] else None,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks)

        # loss
        loss_dict = self.criterion(outputs, targets, cap_list, captions, one_hot_token)

        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=False, prog_bar=False)
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
        self.val_coco_evaluator = CocoEvaluator(self.trainer.datamodule.val_dataset.coco,
                                                iou_types=['bbox'],
                                                eval_class_ids=self.eval_class_ids)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, targets, image_names = batch
        batch_size = data.shape[0]

        # For logits calculation, the entire class names should be passed.
        captions = [self.trainer.datamodule.val_dataset.captions] * batch_size
        cap_list = [self.trainer.datamodule.val_dataset.cap_lists] * batch_size

        (
            tokenized,
            one_hot_token,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(samples=data,
                             targets=targets if self.model_config['use_dn'] else None,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.val_coco_evaluator.update(res)

        loss_dict = self.criterion(outputs, targets, cap_list, captions, one_hot_token)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
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
        self.test_coco_evaluator = CocoEvaluator(self.trainer.datamodule.test_dataset.coco,
                                                 iou_types=['bbox'],
                                                 eval_class_ids=self.eval_class_ids)

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate."""
        data, targets, image_names = batch
        batch_size = data.shape[0]

        # For logits calculation, the entire class names should be passed.
        captions = [self.trainer.datamodule.test_dataset.captions] * batch_size
        (
            tokenized,
            _,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(samples=data,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks)

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
        self.test_coco_evaluator.overall_summarize(is_print=True)
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
        batch_size = data.shape[0]
        captions = [' . '.join(self.cap_lists) + ' .'] * batch_size
        (
            tokenized,
            _,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(data,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks)
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
        """Forward of the groudning dino model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "grounding_dino"
