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

""" Main PTL model file for Mask Grounding DINO. """

import copy
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logger

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr, match_name_keywords
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator

from nvidia_tao_pytorch.cv.grounding_dino.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.grounding_dino.utils.get_tokenlizer import get_tokenlizer
from nvidia_tao_pytorch.cv.grounding_dino.model.bertwraper import generate_masks_with_special_tokens_and_transfer_map

from nvidia_tao_pytorch.cv.mask_grounding_dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.rescoco_eval import ReferStyleCocoEvaluator
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.vl_utils import create_positive_map
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.post_process import (
    PostProcess,
    get_phrase_from_expression,
    save_inference_prediction,
    threshold_predictions,
)


# pylint:disable=too-many-ancestors
class MaskGDINOPlModel(TAOLightningModule):
    """PTL module for MaskGDINO Object Detection and Segmentation Model."""

    def __init__(self, experiment_spec, cap_lists, export=False):
        """Init for MaskGDINO Model."""
        super().__init__(experiment_spec)
        self.cap_lists = cap_lists
        self.max_text_len = self.model_config.max_text_len

        # To disable warnings from HF tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # init the model
        self._build_model(export)
        self._build_criterion()
        if cap_lists is not None:
            caption = " . ".join(cap_lists) + ' .'
            tokenized = self.tokenizer([caption], padding="longest", return_tensors="pt")
            label_list = torch.arange(len(cap_lists))
            label_positive_map = create_positive_map(tokenized[0], label_list, cap_lists, caption)
            self.register_buffer("label_positive_map", label_positive_map, persistent=False)
        else:
            self.label_positive_map = None

        self.status_logging_dict = {}

        self.checkpoint_filename = 'mask_gdino_model'

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
                logger.info(f"Freezed module {freezed_modules}")
                status_logging.get_status_logger().write(
                    message=f"Freezed module {freezed_modules}",
                    status_level=status_logging.Status.RUNNING,
                    verbosity_level=status_logging.Verbosity.INFO)
            if skipped_modules:
                logger.info(f"module {skipped_modules} not found. Skipped freezing")
                status_logging.get_status_logger().write(
                    message=f"module {skipped_modules} not found. Skipped freezing",
                    status_level=status_logging.Status.SKIPPED,
                    verbosity_level=status_logging.Verbosity.WARNING)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.matcher = HungarianMatcher(cost_class=self.model_config["set_cost_class"],
                                        cost_bbox=self.model_config["set_cost_bbox"],
                                        cost_giou=self.model_config["set_cost_giou"])
        self.model.set_matcher(self.matcher)
        weight_dict = {'loss_ce': self.model_config["cls_loss_coef"],
                       'loss_bbox': self.model_config["bbox_loss_coef"],
                       'loss_giou': self.model_config["giou_loss_coef"],
                       'loss_mask': self.model_config["mask_loss_coef"],
                       'loss_dice': self.model_config["dice_loss_coef"],
                       'loss_rela_nt': self.model_config['rela_nt_loss_coef'],
                       'loss_rela_minimap': self.model_config['rela_minimap_loss_coef'],
                       'loss_rela_union_mask': self.model_config['rela_union_mask_loss_coef']}
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
            interm_weight_dict.update({f'{k}_interm': v * self.model_config['interm_loss_coef'] * _coeff_weight_dict.get(k, 0) for k, v in clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)

        self.weight_dict = copy.deepcopy(weight_dict)
        assert "masks" in self.model_config['loss_types'], "`masks` must be included in `loss_types`."
        self.criterion = SetCriterion(matcher=self.matcher,
                                      losses=self.model_config['loss_types'],
                                      focal_alpha=self.model_config["focal_alpha"],
                                      focal_gamma=self.model_config["focal_gamma"])

        self.box_processors = PostProcess(num_select=self.model_config['num_select'],
                                          has_mask=self.model_config['has_mask'])

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
        data, targets = batch[0], batch[1]
        batch_size = data.shape[0]

        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]

        (
            tokenized,
            one_hot_token,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(data,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks,
                             captions=captions,
                             cat_list=cap_list,
                             is_training=True,
                             one_hot_token=one_hot_token,
                             targets=targets)

        # loss
        loss_dict = self.criterion(outputs, targets, cap_list, captions, one_hot_token)

        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_rela_nt", loss_dict.get('loss_rela_nt', 0), on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_rela_minimap", loss_dict.get('loss_rela_minimap', 0), on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_rela_union_mask", loss_dict.get('loss_rela_union_mask', 0), on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_dice_loss", loss_dict.get('loss_dice', 0), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_mask_loss", loss_dict.get('loss_mask', 0), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

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
        if self.experiment_spec.dataset.val_data_sources.data_type == "OD":
            self.iou_types = ['bbox', 'segm'] if self.model_config['has_mask'] else ['bbox']
            self.val_coco_evaluator = CocoEvaluator(
                self.trainer.datamodule.val_dataset.coco,
                iou_types=self.iou_types,
                eval_class_ids=None)
        else:
            dataname = 'coco_variant'
            self.val_coco_evaluator = ReferStyleCocoEvaluator(dataname, self.device, output_dir=self.experiment_spec.results_dir)
            self.val_coco_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        image_names = None
        data, targets = batch[0], batch[1]
        if len(batch) == 3:
            image_names = batch[2]
        batch_size = data.shape[0]

        # Original MaskGDINO considers all class names as caption -> All samples have the same caption
        # For logits calculation, the entire class names should be passed.
        if self.experiment_spec.dataset.val_data_sources.data_type == "OD":
            captions = [self.trainer.datamodule.val_dataset.captions] * batch_size
            cap_list = [self.trainer.datamodule.val_dataset.cap_lists] * batch_size
        else:  # RES Style Dataset, using expression, not class name -> Each sample has a different caption, ODVG data module also assign caption for each sample
            captions = [t["caption"] for t in targets]
            cap_list = [t["cap_list"] for t in targets]

        (
            tokenized,
            one_hot_token,
            position_ids,
            text_self_attention_masks
        ) = self.tokenize_captions(captions)

        outputs = self.model(data,
                             input_ids=tokenized["input_ids"],
                             attention_mask=tokenized["attention_mask"],
                             position_ids=position_ids,
                             token_type_ids=tokenized["token_type_ids"],
                             text_self_attention_masks=text_self_attention_masks,
                             captions=captions,
                             cat_list=cap_list,
                             is_training=False,
                             one_hot_token=one_hot_token,
                             targets=targets)

        loss_dict = self.criterion(outputs, targets, cap_list, captions, one_hot_token)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_bbox", loss_dict['loss_bbox'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_giou", loss_dict['loss_giou'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_rela_nt", loss_dict.get('loss_rela_nt', 0), on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_rela_minimap", loss_dict.get('loss_rela_minimap', 0), on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_loss_rela_union_mask", loss_dict.get('loss_rela_union_mask', 0), on_step=True, on_epoch=False, prog_bar=False)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        if image_names is None:
            image_names = [t["img_name"] for t in targets]
        results, no_targets = self.box_processors(
            outputs,
            orig_target_sizes,
            image_names,
            input_sizes=target_sizes,
            label_positive_map=self.label_positive_map,
            text_threshold=self.experiment_spec.evaluate.text_threshold,
            ioi_threshold=self.experiment_spec.evaluate.ioi_threshold
        )

        if self.experiment_spec.dataset.val_data_sources.data_type == "OD":
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            self.val_coco_evaluator.update(res)
        else:
            self.val_coco_evaluator.update(results, targets, no_targets)

        return losses

    def on_validation_epoch_end(self):
        """
        Validation epoch end.
        Compute gIoU, cIoU, N-Acc, T-Acc, Pr@X, mAP at the end of epoch.
        """
        self.status_logging_dict = {}
        if self.experiment_spec.dataset.val_data_sources.data_type == "OD":
            self.val_coco_evaluator.synchronize_between_processes()
            self.val_coco_evaluator.overall_accumulate()
            self.val_coco_evaluator.overall_summarize(is_print=False)
            for iou_type in self.iou_types:
                mAP = self.val_coco_evaluator.coco_eval[iou_type].stats[0]
                mAP50 = self.val_coco_evaluator.coco_eval[iou_type].stats[1]
                if self.trainer.is_global_zero:
                    print(f"\n Validation mAP ({iou_type}): {mAP}\n")
                    print(f"\n Validation mAP50 ({iou_type}): {mAP50}\n")
                self.log(f"{iou_type}_val_mAP", mAP, sync_dist=True)
                self.log(f"{iou_type}_val_mAP50", mAP50, sync_dist=True)
                self.status_logging_dict[f"{iou_type}_val_mAP"] = str(mAP)
                self.status_logging_dict[f"{iou_type}_val_mAP50"] = str(mAP50)
        else:
            eval_results = self.val_coco_evaluator.evaluate()
            if self.trainer.is_global_zero:
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        self.log(f"val_{key}", value)
                    self.status_logging_dict[f"val_{key}"] = str(value)

        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict["val_loss"] = average_val_loss
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        self.val_coco_evaluator = None
        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self) -> None:
        """
        Test epoch start.
        Reset coco evaluator at start.
        """
        if self.experiment_spec.dataset.test_data_sources.data_type == "OD":   # Keep original mask grounding DINO test dataset, Use OD dataset
            self.iou_types = ['bbox', 'segm'] if self.model_config['has_mask'] else ['bbox']
            self.test_coco_evaluator = CocoEvaluator(
                self.trainer.datamodule.test_dataset.coco,
                iou_types=self.iou_types,
                eval_class_ids=None)
        else:
            dataname = Path(self.experiment_spec.dataset.test_data_sources['json_file']).stem
            self.test_coco_evaluator = ReferStyleCocoEvaluator(dataname, self.device, output_dir=self.experiment_spec.results_dir)
            self.test_coco_evaluator.reset()

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate."""
        data, targets = batch[0], batch[1]
        image_names = None
        if len(batch) == 3:
            image_names = batch[2]
        batch_size = data.shape[0]

        # For logits calculation, the entire class names should be passed.
        if self.experiment_spec.dataset.test_data_sources.data_type == "OD":
            captions = [self.trainer.datamodule.test_dataset.captions] * batch_size
        else:
            captions = [t["caption"] for t in targets]
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
                             text_self_attention_masks=text_self_attention_masks,
                             captions=captions,
                             cat_list=None,
                             is_training=False,
                             one_hot_token=None,
                             targets=targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        if image_names is None:
            image_names = [t["img_name"] for t in targets]
        results, no_targets = self.box_processors(
            outputs,
            orig_target_sizes,
            image_names,
            input_sizes=target_sizes,
            label_positive_map=self.label_positive_map,
            text_threshold=self.experiment_spec.evaluate.text_threshold,
            ioi_threshold=self.experiment_spec.evaluate.ioi_threshold
        )
        filtered_res = threshold_predictions(
            results,
            self.experiment_spec.evaluate.conf_threshold,
            self.experiment_spec.evaluate.nms_threshold
        )
        if self.experiment_spec.dataset.test_data_sources.data_type == "OD":
            res = {target['image_id'].item(): output for target, output in zip(targets, filtered_res)}
            self.test_coco_evaluator.update(res)
        else:
            self.test_coco_evaluator.update(filtered_res, targets, no_targets)

    def on_test_epoch_end(self):
        """
        Test epoch end.
        Compute gIoU, cIoU, N-Acc, T-Acc, Pr@X, mAP at the end of epoch.
        """
        self.status_logging_dict = {}
        if self.experiment_spec.dataset.test_data_sources.data_type == "OD":
            self.test_coco_evaluator.synchronize_between_processes()
            self.test_coco_evaluator.overall_accumulate()
            self.test_coco_evaluator.overall_summarize(is_print=bool(self.trainer.is_global_zero))

            for iou_type in self.iou_types:
                mAP = self.test_coco_evaluator.coco_eval[iou_type].stats[0]
                mAP50 = self.test_coco_evaluator.coco_eval[iou_type].stats[1]
                self.log(f"{iou_type}_test_mAP", mAP, rank_zero_only=True, sync_dist=True)
                self.log(f"{iou_type}_test_mAP50", mAP50, rank_zero_only=True, sync_dist=True)

                # Log the evaluation results to a file
                if self.trainer.is_global_zero:
                    logger.info('**********************Start logging Evaluation Results **********************')
                    logger.info('*************** %s mAP *****************' % iou_type)
                    logger.info('mAP : %2.2f' % mAP)
                    logger.info('*************** %s mAP50 *****************' % iou_type)
                    logger.info('mAP50 : %2.2f' % mAP50)
                self.status_logging_dict[f"{iou_type}_test_mAP"] = str(mAP)
                self.status_logging_dict[f"{iou_type}_test_mAP50"] = str(mAP50)
        else:
            eval_results = self.test_coco_evaluator.evaluate()
            if self.trainer.is_global_zero:
                logger.info(eval_results)
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        self.log(f"test_{key}", value)
                    self.status_logging_dict[f"test_{key}"] = str(value)

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference."""
        data, targets, image_names = batch[0], batch[1], batch[2]
        batch_size = data.shape[0]
        if self.cap_lists is not None:
            captions = [' . '.join(self.cap_lists) + ' .'] * batch_size
        else:
            captions = [t["caption"] for t in targets]
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
                             text_self_attention_masks=text_self_attention_masks,
                             captions=captions,
                             cat_list=None,
                             is_training=False,
                             one_hot_token=None,
                             targets=targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        pred_results, no_targets = self.box_processors(
            outputs,
            orig_target_sizes,
            image_names,
            input_sizes=target_sizes,
            label_positive_map=self.label_positive_map,
            text_threshold=self.experiment_spec.inference.text_threshold,
            ioi_threshold=self.experiment_spec.inference.ioi_threshold
        )
        filtered_res = threshold_predictions(
            pred_results,
            self.experiment_spec.inference.conf_threshold,
            self.experiment_spec.inference.nms_threshold,
            no_targets=no_targets
        )
        if self.experiment_spec.dataset.infer_data_sources.data_type == "VG":
            filtered_res = get_phrase_from_expression(filtered_res, self.tokenizer, tokenized)
        return filtered_res

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Predict batch end.
        Save the result inferences at the end of batch.
        """
        output_dir = self.experiment_spec.results_dir
        color_map = self.experiment_spec.inference.get('color_map', None)
        is_internal = self.experiment_spec.inference.is_internal
        if self.experiment_spec.dataset.infer_data_sources.data_type == "VG":
            use_phrases = True
            label_map = None
        else:
            label_map = self.trainer.datamodule.pred_dataset.label_map
            use_phrases = False
        save_inference_prediction(predictions=outputs,
                                  output_dir=output_dir,
                                  label_map=label_map,
                                  color_map=color_map,
                                  is_internal=is_internal,
                                  use_phrases=use_phrases,
                                  save_masks=self.model_config['has_mask'])

    def forward(self, x):
        """Forward of the groudning dino model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "mask_grounding_dino"
