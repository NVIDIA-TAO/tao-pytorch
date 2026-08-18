# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for Mask2former. """

import copy
import logging
import cv2
import os
import itertools
import json
import functools
import pickle
import numpy as np

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.mask2former.model.mask2former import MaskFormerModel
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import set_precise_msda
from nvidia_tao_pytorch.cv.mask2former.utils.criterion import SetCriterion
from nvidia_tao_pytorch.cv.mask2former.utils.lr_scheduler import WarmupPolyLR
from nvidia_tao_pytorch.cv.mask2former.utils.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.mask2former.utils.metrics import total_intersect_over_union
from nvidia_tao_pytorch.cv.mask2former.utils.solver import maybe_add_gradient_clipping
from nvidia_tao_pytorch.cv.mask2former.utils.task_metrics import (
    create_instance_evaluator,
    finalize_instance_evaluator,
    normalize_task_mode,
    semantic_metric_names,
    validate_instance_category_contract,
)
from nvidia_tao_pytorch.cv.mask2former.utils.d2.structures import Instances
from nvidia_tao_pytorch.cv.mask2former.utils.d2.visualizer import ColorMode, Visualizer
from nvidia_tao_pytorch.cv.mask2former.utils.d2.catalog import MetadataCatalog
logger = logging.getLogger(__name__)


def rgetattr(obj, attr, *args):
    """Get object attribute recursively.
    Args:
        obj (object): object
        attr (str): attribute name, can be nested, e.g. "encoder.block.0"

    Returns:
        object (object): object attribute value
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class Mask2formerPlModule(TAOLightningModule):
    """Mask2former LightningModule."""

    def __init__(self, cfg) -> None:
        """Initialize Mask2former model.
        Args:
            cfg (OmegaConfig): Hydra config
        """
        super().__init__(cfg)
        self.n_bits = 8
        self.num_classes = self.model_config.sem_seg_head.num_classes
        self.num_queries = self.model_config.mask_former.num_object_queries
        self.mode = normalize_task_mode(self.model_config.mode)
        self.test_topk_per_image = self.model_config.test_topk_per_image
        self.overlap_threshold = self.model_config.overlap_threshold
        self.object_mask_threshold = self.model_config.object_mask_threshold

        self._build_model()
        self._build_criterion()
        self.status_logging_dict = {}
        self.val_coco_evaluator = None
        self.test_coco_evaluator = None
        self._instance_category_ids = {}
        if not self.model_config.export:
            metadata = self.get_metadata()
            self.metadata = MetadataCatalog.get("custom").set(
                thing_classes=metadata["thing_classes"],
                thing_colors=metadata["thing_colors"],
                stuff_classes=metadata["stuff_classes"],
                stuff_colors=metadata["stuff_colors"],
                thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
                stuff_dataset_id_to_contiguous_id=metadata["stuff_dataset_id_to_contiguous_id"],
            )

        self.checkpoint_filename = "mask2former_model"

    def get_metadata(self):
        """Prepare metadata for the dataset."""
        label_map = self.experiment_spec.dataset.label_map
        with open(label_map, 'r', encoding='utf-8') as f:
            categories = json.load(f)

        if not self.experiment_spec.dataset.contiguous_id:
            categories_full = [{'name': "nan", 'color': [0, 0, 0], 'isthing': 1, 'id': i + 1} for i in range(self.num_classes)]
            for cat in categories:
                categories_full[cat['id'] - 1] = cat
            categories = categories_full

        meta = {}
        thing_classes = [k["name"] for k in categories if k.get("isthing", 1)]
        thing_colors = [k.get("color", np.random.randint(0, 255, size=3).tolist()) for k in categories if k.get("isthing", 1)]
        stuff_classes = [k["name"] for k in categories]
        stuff_colors = [k.get("color", np.random.randint(0, 255, size=3).tolist()) for k in categories]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        if self.experiment_spec.dataset.contiguous_id:
            thing_dataset_id_to_contiguous_id = {}
            stuff_dataset_id_to_contiguous_id = {}

            for i, cat in enumerate(categories):
                if cat.get("isthing", 1):
                    thing_dataset_id_to_contiguous_id[cat["id"]] = i
                # in order to use sem_seg evaluator
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            thing_dataset_id_to_contiguous_id = {j: j for j in range(len(categories))}
            stuff_dataset_id_to_contiguous_id = {j: j for j in range(len(categories))}
        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
        return meta

    def load_backbone_weights(self, pm=None):
        """Load backbone weights."""
        # TODO(@yuw): only accommodate public weights for swin and efficientvit
        # Need to update for TAO cls
        if pm:
            logger.info(f"Loading backbone weights from: {pm}")
            state_dict = torch.load(pm, map_location='cpu')
            updated_state_dict = {}
            if 'model' in state_dict:
                for key, value in state_dict['model'].items():
                    if "head" in key:
                        continue
                    new_key = ".".join(['backbone', key])
                    updated_state_dict[new_key] = value
            else:
                for key, value in state_dict['state_dict'].items():
                    if "head" in key:
                        continue
                    new_key = key.replace('backbone', 'backbone.model')
                    updated_state_dict[new_key] = value
            self.model.load_state_dict(updated_state_dict, strict=False)
            logger.info("The backbone weights were loaded successfuly.")

    def load_pretrained_weights(self, pm=None):
        """Load TAO pretrained weights."""
        if pm:
            self.model_config.backbone.pretrained_weights = None
            logger.info(f"Loading the pretrained model from: {pm}")
            if pm.endswith('.pkl'):  # load D2 weights
                with open(pm, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                state_dict = data['model']
                updated_state_dict = {}
                for key, value in list(state_dict.items()):
                    if "static_query" in key:
                        key = key.replace("static_query", "query_feat")
                    # new_key = ".".join(['backbone', key])
                    updated_state_dict[key] = torch.from_numpy(value)
                self.model.load_state_dict(updated_state_dict, strict=False)
            elif pm.endswith('.pth'):
                state_dict = torch.load(pm, map_location='cpu')
                updated_state_dict = {}
                for key, value in list(state_dict['state_dict'].items()):
                    if "query_feat" in key or "query_embed" in key or "class_embed" in key:
                        continue
                    if key.startswith("model."):
                        key = key[len("model."):]
                    updated_state_dict[key] = value
                self.model.load_state_dict(updated_state_dict, strict=False)
            elif pm.endswith('.engine'):
                raise NotImplementedError(
                    "TensorRT inference is supported through tao-deploy. "
                    "Please use tao-deploy to generate TensorRT enigne and run inference.")
            else:
                raise NotImplementedError("Model path format is only supported for .pkl or .pth")
            logger.info("The pretrained model was loaded successfuly.")

    def _build_model(self):
        """Internal function to build the model."""
        # Opt-in deterministic MSDeformAttn path (no-op unless precise_msda=True).
        set_precise_msda(getattr(self.model_config, "precise_msda", False))
        self.model = MaskFormerModel(self.experiment_spec)

        # freeze modules
        if self.experiment_spec.train.freeze:
            freezed_modules = []
            skipped_modules = []
            for module in self.experiment_spec.train.freeze:
                try:
                    module_to_freeze = rgetattr(self.model, module)
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
        # Loss parameters:
        deep_supervision = self.model_config.mask_former.deep_supervision
        no_object_weight = self.model_config.mask_former.no_object_weight

        # loss weights
        class_weight = self.model_config.mask_former.class_weight
        dice_weight = self.model_config.mask_former.dice_weight
        mask_weight = self.model_config.mask_former.mask_weight
        # boundary_weight = self.model_config.mask_former.boundary_weight

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.model_config.mask_former.train_num_points,
            use_point_sample=False,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = self.model_config.mask_former.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=self.model_config.mask_former.train_num_points,
            oversample_ratio=self.model_config.mask_former.oversample_ratio,
            importance_sample_ratio=self.model_config.mask_former.importance_sample_ratio,
            use_point_sample=False,
        )

    def configure_optimizers(self):
        """Configure optimizers for training."""
        defaults = {}
        defaults["lr"] = self.experiment_spec.train.optim.lr
        defaults["weight_decay"] = self.experiment_spec.train.optim.weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params = []
        memo = set()
        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.experiment_spec.train.optim.backbone_multiplier
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = 0.0
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm = self.experiment_spec.train.clip_grad_norm
            enable = (
                self.experiment_spec.train.clip_grad_type == "full" and clip_norm > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optim_type = self.experiment_spec.train.optim.type.lower()
        if optim_type == "sgd":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, self.experiment_spec.train.optim.lr, momentum=self.experiment_spec.train.optim.momentum)
        elif optim_type == "adamw":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, self.experiment_spec.train.optim.lr)
        else:
            raise NotImplementedError(
                f"Optimizer type ({self.experiment_spec.train.optim.type}) not supported.")

        if self.experiment_spec.train.clip_grad_type != "full":
            optimizer = maybe_add_gradient_clipping(self.experiment_spec, optimizer)

        total_iters = self.experiment_spec.train.iters_per_epoch * self.experiment_spec.train.num_epochs + 1
        if self.experiment_spec.train.optim.lr_scheduler.lower() == "warmuppoly":  # Step based
            interval = 'step'
            lr_scheduler = WarmupPolyLR(optimizer, total_iters,
                                        warmup_factor=1.0,
                                        warmup_iters=0,
                                        warmup_method="linear",
                                        last_epoch=-1,
                                        power=0.9,
                                        constant_ending=0.0)
        elif self.experiment_spec.train.optim.lr_scheduler.lower() == "multistep":  # Epoch based
            interval = 'epoch'
            lr_scheduler = MultiStepLR(optimizer, self.experiment_spec.train.optim.milestones,
                                       gamma=self.experiment_spec.train.optim.gamma)
        else:
            raise NotImplementedError(f"{self.experiment_spec.train.optim.lr_scheduler} is not supported.")
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                "interval": interval,
                "frequency": 1},
            'monitor': self.experiment_spec.train.optim.monitor_name}

    def training_step(self, batch, batch_idx):
        """Training step."""
        inputs = batch['images']
        targets = batch['targets']
        batch_size = inputs.shape[0]
        outputs = self.model(inputs)

        # loss
        losses = self.criterion(outputs, targets)  # dict
        weight_dict = self.criterion.weight_dict

        loss_total = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_dice", losses['loss_dice'], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_ce", losses['loss_ce'], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_mask", losses['loss_mask'], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("lr_backbone", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return loss_total

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
        """Reset task-correct validation metric state for each epoch."""
        self.validation_outputs = []
        self.val_coco_evaluator = None
        if self.mode == "instance" and not self.trainer.sanity_checking:
            self.val_coco_evaluator = self._create_instance_evaluator("val")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inputs = batch['images']
        targets = batch['targets']

        batch_size = inputs.shape[0]
        if batch_size > 1:
            assert len(self.cfg.dataset.val.target_size) > 0, "target_size must be set for batch evaluation."
        outputs = self.model(inputs)

        # loss
        losses = self.criterion(outputs, targets)  # dict
        weight_dict = self.criterion.weight_dict
        val_loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.mode == "instance":
            if self.val_coco_evaluator is not None:
                self.val_coco_evaluator.update(
                    self._instance_predictions(outputs, batch, split="val")
                )
            return {'val_loss': val_loss}

        metrics = self._semantic_batch_metrics(outputs, batch)
        self.validation_outputs.append(metrics)
        return {'val_loss': val_loss, **metrics}

    def on_validation_epoch_end(self):
        """Finalize and publish the validation metric for the configured task."""
        if self.trainer.sanity_checking:
            self.validation_outputs.clear()
            self.val_coco_evaluator = None
            return

        if self.mode == "instance":
            self.status_logging_dict = self._finalize_instance_metrics(
                self.val_coco_evaluator,
                split="val",
            )
            self.val_coco_evaluator = None
        else:
            self.status_logging_dict = self._finalize_semantic_metrics(split="val")

        val_loss = self._logged_metric_value("val_loss")
        if val_loss is not None:
            self.status_logging_dict["val_loss"] = val_loss
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Eval metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_test_epoch_start(self):
        """Reset task-correct standalone evaluation state."""
        self.validation_outputs = []
        self.test_coco_evaluator = None
        if self.mode == "instance":
            self.test_coco_evaluator = self._create_instance_evaluator("test")

    def _test_eval_config(self):
        """Return the config for the split used by standalone evaluation."""
        test_cfg = self.cfg.dataset.test
        has_test_annotations = any(
            bool(getattr(test_cfg, field, ""))
            for field in ("annot_file", "instance_json", "panoptic_json")
        )
        return test_cfg if has_test_annotations else self.cfg.dataset.val

    def test_step(self, batch, batch_idx):
        """Evaluate a standalone test batch using the configured task metric."""
        inputs = batch['images']
        if inputs.shape[0] > 1:
            assert len(self._test_eval_config().target_size) > 0, \
                "target_size must be set for batch evaluation."
        outputs = self.model(inputs)
        if self.mode == "instance":
            self.test_coco_evaluator.update(
                self._instance_predictions(outputs, batch, split="test")
            )
            return None

        metrics = self._semantic_batch_metrics(outputs, batch)
        self.validation_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Finalize and publish standalone task-correct metrics."""
        if self.mode == "instance":
            self.status_logging_dict = self._finalize_instance_metrics(
                self.test_coco_evaluator,
                split="test",
                is_print=True,
            )
            self.test_coco_evaluator = None
        else:
            self.status_logging_dict = self._finalize_semantic_metrics(split="test")

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def _dataset_for_split(self, split):
        """Return the constructed validation/test dataset."""
        dataset = getattr(self.trainer.datamodule, f"{split}_dataset", None)
        if dataset is None:
            raise RuntimeError(
                f"Mask2Former {split} dataset is unavailable when metric "
                "evaluation starts."
            )
        return dataset

    def _create_instance_evaluator(self, split):
        """Create a COCO mask evaluator and freeze category-index routing."""
        dataset = self._dataset_for_split(split)
        coco = getattr(dataset, "coco", None)
        if coco is None:
            raise ValueError(
                "Mask2Former instance mode requires a COCO instance dataset "
                f"for the {split} split."
            )
        category_ids = validate_instance_category_contract(coco, self.num_classes)
        dataset_mapping = getattr(dataset, "contiguous_id_to_dataset_id", None)
        expected_mapping = {
            index: category_id for index, category_id in enumerate(category_ids)
        }
        if dataset_mapping != expected_mapping:
            raise ValueError(
                "Mask2Former dataset/model category mapping is missing or "
                "inconsistent with the COCO annotation order."
            )
        self._instance_category_ids[split] = category_ids
        return create_instance_evaluator(coco, self.num_classes)

    def _instance_predictions(self, outputs, batch, split):
        """Convert model masks to original-image COCO segmentation results."""
        batch_info = batch.get("info")
        if batch_info is None or len(batch_info) != batch['images'].shape[0]:
            raise ValueError(
                "Mask2Former instance evaluation requires per-image geometry "
                "metadata from COCODataset."
            )
        category_ids = self._instance_category_ids.get(split)
        if category_ids is None:
            raise RuntimeError(
                f"Mask2Former instance category mapping for {split} was not initialized."
            )

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = F.interpolate(
            outputs["pred_masks"],
            size=batch['images'].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        category_lookup = torch.as_tensor(
            category_ids,
            dtype=torch.long,
            device=mask_cls_results.device,
        )

        predictions = {}
        padded_height, padded_width = batch['images'].shape[-2:]
        for mask_cls, mask_pred, info in zip(
                mask_cls_results, mask_pred_results, batch_info):
            resized_height, resized_width = info["resized_size"]
            padding_height, padding_width = info["padding"]
            if (
                resized_height + padding_height != padded_height or
                resized_width + padding_width != padded_width
            ):
                raise ValueError(
                    "Mask2Former evaluation geometry metadata does not match "
                    "the padded input tensor."
                )
            mask_pred = mask_pred[:, :resized_height, :resized_width]
            mask_pred = F.interpolate(
                mask_pred.unsqueeze(0),
                size=tuple(info["original_size"]),
                mode="bilinear",
                align_corners=False,
            )[0]
            result = self.instance_inference(mask_cls, mask_pred)
            labels = category_lookup[result.pred_classes]
            predictions[int(info["image_id"])] = {
                "masks": result.pred_masks,
                "scores": result.scores,
                "labels": labels,
            }
        return predictions

    def _semantic_batch_metrics(self, outputs, batch):
        """Compute one batch of semantic metrics for semantic/diagnostic use."""
        pred_masks = self.batch_semantic_inference(
            outputs["pred_logits"],
            outputs["pred_masks"],
        )
        if self.num_classes == 1:
            pred_semseg = (pred_masks[:, 0] > 0.5).long().cpu().numpy()
        else:
            pred_semseg = torch.argmax(pred_masks, axis=1).cpu().numpy()

        area_intersect, area_union, area_pred_label, area_label = \
            total_intersect_over_union(
                pred_semseg,
                batch['segms'].cpu().numpy(),
                self.num_classes,
                ignore_index=2 ** self.n_bits - 1,
                reduce_zero_label=self.num_classes > 1,
            )
        return {
            'area_intersect': area_intersect,
            'area_union': area_union,
            'area_pred_label': area_pred_label,
            'area_label': area_label,
        }

    def _finalize_semantic_metrics(self, split):
        """Globally aggregate semantic metrics and apply task-correct names."""
        # UnrepeatedDistributedSampler can leave an individual rank without
        # batches.  That rank must still enter the collective below with zero
        # sufficient statistics; raising here would strand the other ranks in
        # all_reduce during distributed validation/test.
        aggregated = np.zeros((4, self.num_classes), dtype=np.float64)
        for index, key in enumerate(
            (
                "area_intersect",
                "area_union",
                "area_pred_label",
                "area_label",
            )
        ):
            values = [np.asarray(item[key]) for item in self.validation_outputs]
            if values:
                aggregated[index] = np.sum(values, axis=0)
        aggregate_tensor = torch.as_tensor(
            aggregated,
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                aggregate_tensor,
                op=torch.distributed.ReduceOp.SUM,
            )
        (
            total_area_intersect,
            total_area_union,
            _,
            total_area_label,
        ) = aggregate_tensor.cpu().numpy()

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = total_area_intersect / total_area_union
            miou = float(np.nanmean(iou))
            all_acc = float(
                total_area_intersect.sum() / total_area_label.sum()
            )
        if not np.isfinite(miou) or not np.isfinite(all_acc):
            raise RuntimeError(
                f"Mask2Former {split} semantic evaluation produced non-finite metrics."
            )

        miou_name, accuracy_name = semantic_metric_names(self.mode, split)
        self.log(miou_name, miou, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        self.log(accuracy_name, all_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        self.validation_outputs.clear()
        status_accuracy_name = (
            "ACC_all" if self.mode == "semantic" else accuracy_name
        )
        return {
            miou_name: miou,
            status_accuracy_name: all_acc,
        }

    def _finalize_instance_metrics(self, evaluator, split, is_print=False):
        """Globally aggregate and publish COCO mask AP for instance mode."""
        if evaluator is None:
            raise RuntimeError(
                f"Mask2Former {split} COCO segmentation evaluator is unavailable."
            )
        metrics = finalize_instance_evaluator(
            evaluator,
            split=split,
            is_print=is_print,
        )
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        if split == "val":
            self.log("current_epoch", self.current_epoch, sync_dist=True)
        return {name: float(value) for name, value in metrics.items()}

    def _logged_metric_value(self, name):
        """Return one finite logged scalar as a Python float when available."""
        value = self.trainer.logged_metrics.get(name)
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        return value if np.isfinite(value) else None

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference."""
        if self.experiment_spec.dataset.test.batch_size > 1:
            assert len(self.experiment_spec.dataset.test.target_size) > 0, "target_size must be set for batch inferencing."
        inputs = batch['images']
        outputs = self.model(inputs)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(inputs.shape[-2], inputs.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return (mask_cls_results, mask_pred_results)

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Predict batch end.
        Save the result inferences at the end of batch.
        """
        inputs = batch['images']
        raw_images = batch['raw_images']
        batch_info = batch['info']

        for i, (mask_cls, mask_pred) in enumerate(zip(*outputs)):
            visualizer = Visualizer(
                raw_images[i],
                MetadataCatalog.get("custom"),
                instance_mode=ColorMode.IMAGE)
            # preprocess prediction to return mask in the original size
            dh, dw = batch['info'][i]['padding']
            image_size = batch['info'][i]['image_size']
            mask_pred = mask_pred[:, :inputs.shape[-2] - dh, :inputs.shape[-1] - dw].expand(1, -1, -1, -1)
            mask_pred = F.interpolate(mask_pred, size=image_size, mode='bilinear', align_corners=False)[0]

            if self.mode == "panoptic":
                panoptic_seg, segments_info = self.panoptic_inference(mask_cls, mask_pred)
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(torch.device("cpu")), segments_info
                )
            elif self.mode == "instance":
                result = self.instance_inference(mask_cls, mask_pred)
                vis_output = visualizer.draw_instance_predictions(
                    predictions=result.to(torch.device("cpu")),
                    mask_threshold=self.object_mask_threshold
                )
            elif self.mode == "semantic":
                sem_seg = self.semantic_inference(mask_cls, mask_pred)
                # For binary segmentation (num_classes=1), threshold the single channel
                # instead of using argmax which always returns 0
                if self.num_classes == 1:
                    pred_seg = (sem_seg[0] > 0.5).long().cpu()
                else:
                    pred_seg = sem_seg.argmax(dim=0).cpu()
                vis_output = visualizer.draw_sem_seg(pred_seg)
            else:
                raise ValueError(f"The provided model.mode ({self.mode}) is not supported.")
            cv2.imwrite(
                os.path.join(
                    self.experiment_spec.inference.results_dir,
                    batch_info[i]["filename"] + ".jpg"),
                vis_output.get_image()
            )

    def forward(self, x):
        """Forward."""
        outputs = self.model(x)
        return outputs

    def _get_dice(self, predict, target):
        smooth = 1e-5
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1)
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        return score.mean()

    def semantic_inference(self, mask_cls, mask_pred):
        """Post process for semantic segmentation."""
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # original Mask2former
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        """Post process for panoptic segmentation."""
        scores, labels = F.softmax(mask_cls, dim=-1)[..., 1:].max(-1)
        # Original Mask2former
        # scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = scores > self.object_mask_threshold
        # original Mask2former:
        # keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info

        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        """Post process for instance segmentation."""
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, 1:]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def batch_semantic_inference(self, mask_cls, mask_pred):
        """Batched semantic inference."""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "mask2former"
