# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Lightning module for CoDETR."""

import copy
import json

import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import CoCoDataMerge
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import (
    PostProcess, save_inference_prediction, threshold_predictions
)
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco import COCO
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import rgetattr

from nvidia_tao_pytorch.cv.codetr.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.codetr.model.category_mapping import (
    apply_category_mapping_groupnms,
    build_default_color_map,
    build_output_label_map_and_remap,
    soft_nms_kwargs_from_model_config,
)
from nvidia_tao_pytorch.cv.codetr.model.criterion import CoDETRCriterion
from nvidia_tao_pytorch.cv.dino.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.dino.model.vision_transformer.transformer_modules import get_vit_lr_decay_rate

# stride mapping: backbone stage index → feature map stride
# Index 4 is the extra level created via stride-2 conv on the last backbone stage.
_STRIDE_MAP = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}


class CoDETRPlModel(TAOLightningModule):
    """PTL module for CoDETR object detection.

    Args:
        experiment_spec (OmegaConf): experiment configuration containing model,
            dataset, train, inference, and evaluation settings.
        export (bool): whether to initialize the module for export instead of
            training or evaluation.
    """

    def __init__(self, experiment_spec, export=False):
        """Initialize CoDETR training module.

        Args:
            experiment_spec (OmegaConf): experiment configuration containing
                model, dataset, train, inference, and evaluation settings.
            export (bool): whether to initialize the module for export instead
                of training or evaluation.
        """
        super().__init__(experiment_spec)
        self.eval_class_ids = self.dataset_config["eval_class_ids"]
        self.dataset_type = self.dataset_config["dataset_type"]
        if self.dataset_type not in ("serialized", "default"):
            raise ValueError(f"dataset_type '{self.dataset_type}' not supported.")

        self._build_model(export)
        self._build_criterion()
        self.checkpoint_filename = 'codetr_model'
        # Populated by on_predict_start when inference.category_mapping is set.
        self._output_label_map = None
        self._category_remap = None
        self._auto_color_map = None
        self._saved_box_processors_soft_nms = None

    def _build_model(self, export):
        """Build the CoDETR model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

        # Optional module freezing
        if self.experiment_spec["train"]["freeze"]:
            frozen, skipped = [], []
            for module_name in self.experiment_spec["train"]["freeze"]:
                try:
                    module = rgetattr(self.model.model.model, module_name)
                    for p in module.parameters():
                        p.requires_grad = False
                    frozen.append(module_name)
                except AttributeError:
                    skipped.append(module_name)
            if frozen:
                status_logging.get_status_logger().write(
                    message=f"Froze modules: {frozen}",
                    status_level=status_logging.Status.RUNNING,
                    verbosity_level=status_logging.Verbosity.INFO)
            if skipped:
                status_logging.get_status_logger().write(
                    message=f"Modules not found (skipped): {skipped}",
                    status_level=status_logging.Status.SKIPPED,
                    verbosity_level=status_logging.Verbosity.WARNING)

    def _build_criterion(self):
        """Build the CoDETR combined loss criterion."""
        mc = self.model_config
        return_interm_indices = list(mc["return_interm_indices"])
        strides = [_STRIDE_MAP[i] for i in return_interm_indices if i in _STRIDE_MAP]
        # Collab heads receive encoder features + one downsampled level (stride 2x last)
        collab_strides = strides + [strides[-1] * 2]

        matcher = HungarianMatcher(
            cost_class=mc["cls_loss_coef"],
            cost_bbox=mc["bbox_loss_coef"],
            cost_giou=mc["giou_loss_coef"],
        )

        weight_dict = {
            'loss_ce': mc["cls_loss_coef"],
            'loss_bbox': mc["bbox_loss_coef"],
            'loss_giou': mc["giou_loss_coef"],
        }
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        if mc['use_dn']:
            weight_dict['loss_ce_dn'] = mc["cls_loss_coef"]
            weight_dict['loss_bbox_dn'] = mc["bbox_loss_coef"]
            weight_dict['loss_giou_dn'] = mc["giou_loss_coef"]
        clean_weight_dict = copy.deepcopy(weight_dict)

        if mc["aux_loss"]:
            aux_weight_dict = {}
            for i in range(mc["dec_layers"] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if mc['two_stage_type'] != 'no':
            interm_weight_dict = {}
            _coeff = {
                'loss_ce': 1.0,
                'loss_bbox': 0.0 if mc['no_interm_box_loss'] else 1.0,
                'loss_giou': 0.0 if mc['no_interm_box_loss'] else 1.0,
            }
            interm_weight_dict.update({
                f'{k}_interm': v * mc['interm_loss_coef'] * _coeff[k]
                for k, v in clean_weight_dict_wo_dn.items()
            })
            weight_dict.update(interm_weight_dict)

        # Collaborative head losses — add to weight_dict
        co_w = mc['co_head_loss_weight']
        for head_idx in range(mc['num_co_heads']):
            suffix = '' if head_idx == 0 else f'_{head_idx}'
            weight_dict[f'collab_loss_cls{suffix}'] = co_w
            weight_dict[f'collab_loss_bbox{suffix}'] = co_w
            weight_dict[f'collab_loss_centerness{suffix}'] = co_w

        self.weight_dict = copy.deepcopy(weight_dict)

        self.criterion = CoDETRCriterion(
            num_classes=self.dataset_config["num_classes"],
            matcher=matcher,
            losses=mc['loss_types'],
            focal_alpha=mc["focal_alpha"],
            strides=collab_strides,
            co_head_loss_weight=mc['co_head_loss_weight'],
        )
        self.box_processors = PostProcess(
            num_select=mc['num_select'],
            soft_nms_enabled=mc.get('soft_nms_enabled', False),
            soft_nms_method=mc.get('soft_nms_method', 'linear'),
            soft_nms_iou_threshold=mc.get('soft_nms_iou_threshold', 0.8),
            soft_nms_sigma=mc.get('soft_nms_sigma', 0.5),
        )

    def configure_optimizers(self):
        """Configure AdamW / SGD optimizers with backbone LR separation."""
        train_config = self.experiment_spec.train
        param_dicts = []
        backbone_name = self.model_config.backbone

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" not in n:
                param_dicts.append({"params": [p]})
            elif backbone_name.startswith("vit"):
                num_layers = self.model.model.model.backbone[0].body.depth
                scale = get_vit_lr_decay_rate(n, lr_decay_rate=train_config.optim.layer_decay_rate,
                                              num_layers=num_layers)
                param_dicts.append({"params": [p], "lr": train_config.optim.lr * scale})
            else:
                param_dicts.append({"params": [p], "lr": train_config.optim.lr_backbone})

        if train_config.optim.optimizer == 'AdamW':
            optim = torch.optim.AdamW(params=param_dicts, lr=train_config.optim.lr,
                                      weight_decay=train_config.optim.weight_decay)
        elif train_config.optim.optimizer == 'SGD':
            optim = torch.optim.SGD(params=param_dicts, lr=train_config.optim.lr,
                                    momentum=train_config.optim.momentum,
                                    weight_decay=train_config.optim.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {train_config.optim.optimizer} not implemented")

        sched_type = train_config.optim.lr_scheduler
        if sched_type == "MultiStep":
            scheduler = MultiStepLR(optim, milestones=train_config.optim.lr_steps,
                                    gamma=train_config.optim.lr_decay)
        elif sched_type == "StepLR":
            scheduler = StepLR(optim, step_size=train_config.optim.lr_step_size,
                               gamma=train_config.optim.lr_decay)
        else:
            raise NotImplementedError(f"LR scheduler {sched_type} not implemented")

        return {"optimizer": optim, "lr_scheduler": scheduler,
                "monitor": train_config.optim.monitor_name}

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, targets, _ = batch
        batch_size = data.shape[0]

        outputs = self.model(data, targets=targets if self.model_config['use_dn'] else None)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k]
                     for k in loss_dict if k in self.weight_dict)

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("train_loss_ce", loss_dict.get('loss_ce', 0.),
                 on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_loss_bbox", loss_dict.get('loss_bbox', 0.),
                 on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_collab_cls", loss_dict.get('collab_loss_cls', 0.),
                 on_step=True, on_epoch=False, prog_bar=False)
        lrs = [pg['lr'] for pg in self.optimizers().optimizer.param_groups]
        self.log("lr", lrs[0], on_step=True, on_epoch=False, prog_bar=True)
        return losses

    def on_train_epoch_end(self):
        """Log training metrics to status.json."""
        avg_loss = self.trainer.logged_metrics.get("train_loss_epoch", 0.)
        if hasattr(avg_loss, 'item'):
            avg_loss = avg_loss.item()
        self.status_logging_dict = {"train_loss": avg_loss}
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING)

    def on_validation_epoch_start(self):
        """Reset COCO evaluator for this epoch."""
        if self.dataset_type == "serialized":
            coco_lists = []
            for source in self.dataset_config["val_data_sources"]:
                with open(source["json_file"], "r") as f:
                    tmp = json.load(f)
                coco_lists.append(COCO(tmp))
            coco = COCO(CoCoDataMerge(coco_lists))
        else:
            coco = self.trainer.datamodule.val_dataset.coco
        self.val_coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'],
                                                eval_class_ids=self.eval_class_ids)
        self._val_contiguous2cat = getattr(
            self.trainer.datamodule.val_dataset, 'contiguous2cat', None
        )

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, targets, image_names = batch
        batch_size = data.shape[0]

        outputs = self.model(data, targets=targets if self.model_config['use_dn'] else None)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k]
                     for k in loss_dict if k in self.weight_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        if self._val_contiguous2cat is not None:
            for r in results:
                r['labels'] = torch.tensor(
                    [self._val_contiguous2cat[lbl.item()] for lbl in r['labels']],
                    device=r['labels'].device,
                )
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.val_coco_evaluator.update(res)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        return losses

    def on_validation_epoch_end(self):
        """Compute validation mAP."""
        self.val_coco_evaluator.synchronize_between_processes()
        self.val_coco_evaluator.overall_accumulate()
        self.val_coco_evaluator.overall_summarize(is_print=False)
        mAP = self.val_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.val_coco_evaluator.coco_eval['bbox'].stats[1]
        if self.trainer.is_global_zero:
            logging.info(f"\n Validation mAP: {mAP}\n mAP50: {mAP50}")

        self.log("current_epoch", self.current_epoch, sync_dist=True)
        self.log("val_mAP", mAP, sync_dist=True)
        self.log("val_mAP50", mAP50, sync_dist=True)

        avg_val_loss = self.trainer.logged_metrics.get("val_loss", 0.)
        if hasattr(avg_val_loss, 'item'):
            avg_val_loss = avg_val_loss.item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {
                "val_mAP": str(mAP), "val_mAP50": str(mAP50), "val_loss": avg_val_loss
            }
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING)
        self.val_coco_evaluator = None
        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self):
        """Reset COCO evaluator for test."""
        if self.dataset_type == "serialized":
            with open(self.dataset_config["test_data_sources"]["json_file"], "r") as f:
                tmp = json.load(f)
            coco = COCO(tmp)
        else:
            coco = self.trainer.datamodule.test_dataset.coco
        self.test_coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'],
                                                 eval_class_ids=self.eval_class_ids)
        self._contiguous2cat = getattr(
            self.trainer.datamodule.test_dataset, 'contiguous2cat', None
        )

    def test_step(self, batch, batch_idx):
        """Test step."""
        data, targets, image_names = batch
        outputs = self.model(data)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        if self.experiment_spec.evaluate.conf_threshold > 0:
            results = threshold_predictions(results, self.experiment_spec.evaluate.conf_threshold)
        if self._contiguous2cat is not None:
            for r in results:
                r['labels'] = torch.tensor(
                    [self._contiguous2cat[lbl.item()] for lbl in r['labels']],
                    device=r['labels'].device,
                )
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.test_coco_evaluator.update(res)

    def on_test_epoch_end(self):
        """Compute test mAP."""
        self.test_coco_evaluator.synchronize_between_processes()
        self.test_coco_evaluator.overall_accumulate()
        self.test_coco_evaluator.overall_summarize(is_print=True)
        mAP = self.test_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.test_coco_evaluator.coco_eval['bbox'].stats[1]
        self.log("test_mAP", mAP, rank_zero_only=True)
        self.log("test_mAP50", mAP50, rank_zero_only=True)
        self.status_logging_dict = {"test_mAP": str(mAP), "test_mAP50": str(mAP50)}
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING)

    def on_predict_start(self):
        """Build optional category_mapping remap once, before any predict batch.

        The model's input ``label_map`` lives on the datamodule, so we can only
        derive the output category remap after ``setup('predict')``.

        When ``category_mapping`` is set, we also disable the per-original-class
        soft-NMS pass inside ``box_processors`` for the predict path only — the
        single per-output-category pass inside
        :func:`apply_category_mapping_groupnms` is the right place to do NMS,
        and running both would double-decay scores for any pair that ends up in
        the same group both times. The original setting is restored in
        ``on_predict_end`` so val/test paths are unaffected.
        """
        self._output_label_map = None
        self._category_remap = None
        self._auto_color_map = None
        self._saved_box_processors_soft_nms = None
        cat_map = getattr(self.experiment_spec.inference, "category_mapping", None)
        if cat_map:
            input_label_map = self.trainer.datamodule.pred_dataset.label_map
            self._output_label_map, self._category_remap = (
                build_output_label_map_and_remap(input_label_map, cat_map)
            )
            # Auto-generate a color_map covering the new output categories so the
            # writer doesn't silently skip drawing them. The default writer only
            # draws boxes whose class_name is a key in color_map; the COCO
            # METAINFO fallback doesn't contain merged names like "road_sign".
            self._auto_color_map = build_default_color_map(self._output_label_map)
            self._saved_box_processors_soft_nms = self.box_processors.soft_nms_enabled
            self.box_processors.soft_nms_enabled = False
            logging.info(
                "[category_mapping] %d original classes -> %d output categories. "
                "Disabling per-original-class soft-NMS in box_processors; "
                "group-NMS will run on output categories instead.",
                len(self._category_remap), len(self._output_label_map),
            )

    def on_predict_end(self):
        """Restore box_processors.soft_nms_enabled if we toggled it in on_predict_start."""
        if self._saved_box_processors_soft_nms is not None:
            self.box_processors.soft_nms_enabled = self._saved_box_processors_soft_nms
            self._saved_box_processors_soft_nms = None

    def predict_step(self, batch, batch_idx):
        """Predict step (with optional category-mapping post-processing)."""
        data, targets, image_names = batch
        outputs = self.model(data)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        if self._category_remap is not None:
            results = apply_category_mapping_groupnms(
                results,
                self._category_remap,
                soft_nms_kwargs=soft_nms_kwargs_from_model_config(self.model_config),
            )
        return results

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Save inference predictions."""
        output_dir = self.experiment_spec.results_dir
        # Use the output category label_map when category_mapping is enabled.
        label_map = (
            self._output_label_map
            if self._output_label_map is not None
            else self.trainer.datamodule.pred_dataset.label_map
        )
        # When category_mapping is active, fall back to an auto-generated color
        # for each output category if the user didn't specify one. User-provided
        # entries always win on overlapping keys.
        user_color_map = self.experiment_spec.inference.color_map
        if self._auto_color_map is not None:
            color_map = dict(self._auto_color_map)
            if user_color_map:
                color_map.update(user_color_map)
        else:
            color_map = user_color_map
        conf_threshold = self.experiment_spec.inference.conf_threshold
        is_internal = self.experiment_spec.inference.is_internal
        outline_width = self.experiment_spec.inference.outline_width
        save_annotated_images = getattr(
            self.experiment_spec.inference, "save_annotated_images", True
        )
        save_inference_prediction(outputs, output_dir, conf_threshold, label_map,
                                  color_map, is_internal, outline_width,
                                  save_annotated_images=save_annotated_images)

    def forward(self, x):
        """Forward pass for inference."""
        return self.model(x)

    def on_save_checkpoint(self, checkpoint):
        """Tag checkpoint with model identifier."""
        checkpoint["tao_model"] = "codetr"
