# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch Lightning module for OneFormer model training and inference."""

import copy
from nvidia_tao_pytorch.core.tlt_logging import logging
import itertools
import functools
import numpy as np
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
import json
import os
import cv2

from nvidia_tao_pytorch.cv.oneformer.model.oneformer_model import OneFormerModel
from nvidia_tao_pytorch.cv.oneformer.utils.criterion import SetCriterion
from nvidia_tao_pytorch.cv.mask2former.utils.lr_scheduler import WarmupPolyLR
from nvidia_tao_pytorch.cv.oneformer.utils.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.mask2former.utils.metrics import total_intersect_over_union
from nvidia_tao_pytorch.cv.mask2former.utils.d2.visualizer import ColorMode, Visualizer
from nvidia_tao_pytorch.cv.mask2former.utils.d2.catalog import MetadataCatalog
from nvidia_tao_pytorch.cv.mask2former.utils.d2.structures import Instances


def rgetattr(obj, attr, *args):
    """Recursively get attribute from nested object using dot notation.

    Args:
        obj: Object to get attribute from
        attr (str): Dot-separated attribute path
        *args: Default value if attribute not found

    Returns:
        Attribute value or default
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class OneformerPlModule(TAOLightningModule):
    """PyTorch Lightning module for OneFormer model.

    This class wraps the OneFormer model for training, validation, and testing
    using PyTorch Lightning framework.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_config = cfg.model
        self.n_bits = 8
        self.num_classes = self.model_config.sem_seg_head.num_classes
        self.object_mask_threshold = self.model_config.test.object_mask_threshold
        self.overlap_threshold = self.model_config.test.overlap_threshold
        self.test_topk_per_image = self.model_config.test.test_topk_per_image
        self.num_queries = self.model_config.one_former.num_object_queries
        self._build_model()
        self._build_criterion()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.checkpoint_filename = "oneformer_model"
        self.status_logging_dict = {}
        self.mode = self.cfg.inference.mode.lower()
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

    def get_metadata(self):
        """Prepare metadata for the dataset."""
        label_map = self.cfg.dataset.label_map
        with open(label_map, 'r', encoding='utf-8') as f:
            categories = json.load(f)

        if not self.cfg.dataset.contiguous_id:
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

        if self.cfg.dataset.contiguous_id:
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
        """Load pretrained backbone weights.

        Args:
            pm (str, optional): Path to pretrained model weights
        """
        if pm and self.cfg.model.backbone.name == "D2SwinTransformer":
            checkpoint = torch.load(pm, map_location="cuda", weights_only=False)
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                new_key = "backbone." + key
                new_state_dict[new_key] = value
            missing_keys, _ = self.model.load_state_dict(new_state_dict, strict=False)
            total_keys = len(self.model.state_dict())
            num_missing_keys = len(missing_keys)
            successful_keys_count = total_keys - num_missing_keys
            logging.info(
                f"The backbone weights were loaded successfully. {successful_keys_count} keys loaded out of {total_keys}."
            )
        elif pm and self.cfg.model.backbone.radio.backbone == "vit_large_patch16_reg4_dinov2":
            checkpoint = torch.load(pm, map_location="cuda", weights_only=False)
            new_state_dict = OrderedDict()
            expected_prefix = "backbone.radio."

            for key, value in checkpoint.items():
                if "grandma" in key:
                    # Keep your specific replacement for "grandma" keys
                    new_key = "backbone." + key.replace("grandma", "gamma")
                    new_state_dict[new_key] = value
                elif key.startswith("radio_model"):
                    key = key.replace("radio_model", "radio")
                    new_key = expected_prefix + key
                    new_state_dict[new_key] = value
                else:
                    # Copy any other keys that don't match the above conditions
                    new_state_dict[key] = value
            missing_keys, _ = self.model.load_state_dict(new_state_dict, strict=False)
            total_keys = len(self.model.state_dict())
            num_missing_keys = len(missing_keys)
            successful_keys_count = total_keys - num_missing_keys
            logging.info(
                f"The backbone weights were loaded successfully. {successful_keys_count} keys loaded out of {total_keys}."
            )

    def load_pretrained_weights(self, pm=None):
        """Load pretrained model weights.

        Args:
            pm (str, optional): Path to pretrained model file
        """
        if pm.endswith(".pth"):
            checkpoint = torch.load(pm, map_location="cuda", weights_only=False)
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.removeprefix("model.")
                new_state_dict[new_key] = value
            missing_keys, unexpected_keys = self.model.load_state_dict(
                new_state_dict, strict=False
            )
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
            logging.info("The pretrained model was loaded successfully.")
        else:
            logging.info("No pretrained model provided.")

    def _build_model(self):
        self.model = OneFormerModel(self.cfg)
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "freeze"):
            freezed_modules = []
            skipped_modules = []
            for module in self.cfg.train.freeze:
                try:
                    module_to_freeze = rgetattr(self.model, module)
                    protected_params = set()
                    for name, param in module_to_freeze.named_parameters():
                        if "query_feat" in name or "query_embed" in name:
                            protected_params.add(param)
                            logging.info(
                                f"Protecting query embedding from freezing: {name}"
                            )

                    for p in module_to_freeze.parameters():
                        if p not in protected_params:
                            p.requires_grad = False
                        else:
                            logging.info(
                                "Kept query embedding trainable despite freeze config"
                            )

                    freezed_modules.append(module)
                except AttributeError:
                    skipped_modules.append(module)
            if freezed_modules:
                logging.info(f"Freezed modules: {freezed_modules}")
            if skipped_modules:
                logging.warning(
                    f"Modules not found (skipped freezing): {skipped_modules}"
                )

    def _build_criterion(self):
        deep_supervision = self.model_config.one_former.deep_supervision
        no_object_weight = self.model_config.one_former.no_object_weight
        class_weight = self.model_config.one_former.class_weight
        dice_weight = self.model_config.one_former.dice_weight
        mask_weight = self.model_config.one_former.mask_weight
        contrastive_weight = self.model_config.one_former.contrastive_weight
        contrast_temperature = self.model_config.one_former.contrastive_temperature

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.model_config.one_former.train_num_points,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_contrastive": contrastive_weight,
        }

        if deep_supervision:
            dec_layers = self.model_config.one_former.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "contrastive"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=self.model_config.one_former.train_num_points,
            oversample_ratio=self.model_config.one_former.oversample_ratio,
            importance_sample_ratio=self.model_config.one_former.importance_sample_ratio,
            contrast_temperature=contrast_temperature,
        )

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        train_cfg = self.cfg.train.optim
        defaults = {}
        defaults["lr"] = train_cfg.lr
        defaults["weight_decay"] = train_cfg.weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
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
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * train_cfg.backbone_multiplier
                    )
                if (
                    "relative_position_bias_table" in module_param_name or
                    "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = 0.0

                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = 0.0
                    if (
                        "query_feat" in module_param_name or
                        "query_embed" in module_param_name
                    ):
                        hyperparams["lr"] = hyperparams["lr"] * 2.0
                        logging.info(
                            f"Setting higher LR for query embedding: {module_name}.{module_param_name}"
                        )

                params.append({"params": [value], **hyperparams})

        trainable_params = sum(
            p.numel() for group in params for p in group["params"] if p.requires_grad
        )
        logging.info(
            f"Total parameter groups: {len(params)}, Total trainable parameters: {trainable_params}"
        )

        def maybe_add_gradient_clipping(
            optimizer_cls, optimizer_kwargs, clip_norm, clip_type
        ):
            class OptimizerWithGradClip(optimizer_cls):
                def step(self, closure=None):
                    if clip_type == "full_model":
                        all_params = itertools.chain(
                            *[x["params"] for x in self.param_groups]
                        )
                        torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
                    elif clip_type == "value":
                        for group in self.param_groups:
                            torch.nn.utils.clip_grad_value_(group["params"], clip_norm)
                    super().step(closure=closure)

            if clip_norm > 0.0:
                return OptimizerWithGradClip, optimizer_kwargs
            return optimizer_cls, optimizer_kwargs

        clip_norm = 0.0
        clip_type = "none"
        if hasattr(train_cfg, "clip_gradients") and train_cfg.clip_gradients.enabled:
            clip_norm = train_cfg.clip_gradients.clip_value
            clip_type = train_cfg.clip_gradients.clip_type

        optim_type = train_cfg.type.lower()
        if optim_type == "sgd":
            momentum = (
                train_cfg.optim.momentum
                if hasattr(train_cfg, "optim") and hasattr(train_cfg.optim, "momentum")
                else 0.9
            )
            optimizer_cls, optimizer_kwargs = maybe_add_gradient_clipping(
                torch.optim.SGD, {"momentum": momentum}, clip_norm, clip_type
            )
            optimizer = optimizer_cls(params, **optimizer_kwargs)

        elif optim_type == "adamw":
            optimizer_cls, optimizer_kwargs = maybe_add_gradient_clipping(
                torch.optim.AdamW, {}, clip_norm, clip_type
            )
            optimizer = optimizer_cls(params, **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer type ({optim_type}) not supported.")

        iters_per_epoch = None
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "iters_per_epoch"):
            iters_per_epoch = self.cfg.train.iters_per_epoch

        if iters_per_epoch is None:
            if hasattr(self, "datamodule") and hasattr(
                self.datamodule, "train_dataloader"
            ):
                train_loader = self.datamodule.train_dataloader()
                iters_per_epoch = len(train_loader)
            else:
                iters_per_epoch = 1000
                logging.warning(
                    f"iters_per_epoch not specified, using default value of {iters_per_epoch}"
                )

        num_epochs = (
            self.cfg.train.num_epochs
            if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "num_epochs")
            else 50
        )
        total_iters = iters_per_epoch * num_epochs + 1

        if hasattr(train_cfg, "optim") and hasattr(train_cfg.optim, "lr_scheduler"):
            scheduler_type = train_cfg.optim.lr_scheduler.lower()
            if scheduler_type == "warmuppoly":
                interval = "step"
                warmup_iters = (
                    train_cfg.optim.warmup_iters
                    if hasattr(train_cfg.optim, "warmup_iters")
                    else train_cfg.warmup_iters
                )
                warmup_factor = (
                    train_cfg.optim.warmup_factor
                    if hasattr(train_cfg.optim, "warmup_factor")
                    else train_cfg.warmup_factor
                )
                lr_scheduler = WarmupPolyLR(
                    optimizer,
                    total_iters,
                    warmup_factor=warmup_factor,
                    warmup_iters=warmup_iters,
                    warmup_method="linear",
                    last_epoch=-1,
                    power=0.9,
                    constant_ending=0.0,
                )
            elif scheduler_type == "multistep":
                interval = "epoch"
                steps = (
                    train_cfg.optim.steps
                    if hasattr(train_cfg.optim, "steps")
                    else train_cfg.steps
                )
                gamma = (
                    train_cfg.optim.gamma if hasattr(train_cfg.optim, "gamma") else 0.1
                )
                lr_scheduler = MultiStepLR(optimizer, steps, gamma=gamma)
            else:
                raise NotImplementedError(f"{scheduler_type} is not supported.")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": interval,
                    "frequency": 1,
                    "monitor": None,
                    "strict": False,
                },
            }
        else:
            return optimizer

    def setup(self, stage=None):
        """Set up the model for training/testing."""
        if (
            self.trainer and
            self.trainer.strategy and
            getattr(self.trainer.strategy, "_ddp_strategy_initialized", False)
        ):

            def make_hook():
                def hook(grad):
                    return grad.contiguous() if grad is not None else grad

                return hook

            for _, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(make_hook())

            logging.info("Registered gradient contiguity hooks for DDP")

    def prepare_targets(self, targets, images):
        """Prepare targets for training."""
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        inputs = batch["images"]
        batch_size = inputs.shape[0]
        targets = batch["instances"]
        targets = self.prepare_targets(targets, inputs)
        tasks = batch["tasks"]
        texts = batch["texts"]
        outputs = self.model(inputs, tasks=tasks, texts=texts)

        losses = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict

        loss_total = sum(
            losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict
        )
        self.log(
            "train_loss",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_dice",
            losses["loss_dice"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_ce",
            losses["loss_ce"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_mask",
            losses["loss_mask"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_contrastive",
            losses["loss_contrastive"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

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
        """Initialize validation epoch."""
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        inputs = batch["images"]
        segms = batch["sem_segs"]
        tasks = batch["tasks"]
        outputs = self.model(inputs, tasks=tasks, texts=None)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        pred_masks = self.batch_semantic_inference(mask_cls_results, mask_pred_results)

        mask_pred_resize = F.interpolate(
            pred_masks,
            size=(inputs.shape[-2], inputs.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        pred_semseg = torch.argmax(mask_pred_resize, dim=1).cpu().numpy()
        area_intersect, area_union, area_pred_label, area_label = (
            total_intersect_over_union(
                pred_semseg,
                segms.cpu().numpy(),
                self.num_classes,
                ignore_index=2**self.n_bits - 1,
                reduce_zero_label=False,
            )
        )
        val_metrics = {
            "area_intersect": area_intersect,
            "area_union": area_union,
            "area_pred_label": area_pred_label,
            "area_label": area_label,
        }
        self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def on_validation_epoch_end(self):
        """Process validation epoch end."""
        total_area_intersect, total_area_union = 0, 0
        total_area_pred_label, total_area_label = 0, 0

        for out in self.validation_step_outputs:
            total_area_intersect += out["area_intersect"]
            total_area_union += out["area_union"]
            total_area_pred_label += out["area_pred_label"]
            total_area_label += out["area_label"]

        iou = total_area_intersect / total_area_union
        miou = np.nanmean(iou)
        all_acc = total_area_intersect.sum() / total_area_label.sum()
        self.log(
            "mIoU",
            miou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            "all_acc",
            all_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.status_logging_dict = {}
        self.status_logging_dict["mIoU"] = float(miou)
        self.status_logging_dict["ACC_all"] = float(all_acc)

        # class_names = self.metadata.stuff_classes
        # for i, class_iou in enumerate(iou):
        #     if i < len(class_names):
        #         class_name = class_names[i].replace(" ", "_")
        #         self.log(
        #             f"{class_name}",
        #             class_iou,
        #             on_step=False,
        #             on_epoch=True,
        #             prog_bar=False,  # Set to False to avoid cluttering the progress bar
        #             sync_dist=True
        #         )
        #         self.status_logging_dict[f"IoU_{class_name}"] = float(class_iou)

        self.validation_step_outputs.clear()

        if not self.trainer.sanity_checking:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Test step"""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Test epoch end"""
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        """Perform a prediction step."""
        inputs = batch["images"]
        tasks = batch["tasks"]
        outputs = self.model(inputs, tasks=tasks, texts=None)

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
        """Perform a prediction batch end."""
        inputs = batch["images"]
        raw_images = batch["raw_images"]
        batch_info = batch["info"]
        for i, (mask_cls, mask_pred) in enumerate(zip(*outputs)):
            visualizer = Visualizer(
                raw_images[i],
                MetadataCatalog.get("custom"),
                instance_mode=ColorMode.IMAGE
            )
            dh, dw = batch_info[i]['padding']
            image_size = batch_info[i]['image_size']
            mask_pred = mask_pred[:, :inputs.shape[-2] - dh, :inputs.shape[-1] - dw].expand(1, -1, -1, -1)
            mask_pred = F.interpolate(mask_pred, size=image_size, mode='bilinear', align_corners=False)[0]

            if self.mode == 'semantic':
                sem_seg = self.semantic_inference(mask_cls, mask_pred)
                vis_output = visualizer.draw_sem_seg(sem_seg.argmax(dim=0).cpu())
                vis_output.save(f"{self.cfg.results_dir}/pred_{batch_info[i]['filename']}.png")
            elif self.mode == "instance":
                result = self.instance_inference(mask_cls, mask_pred)
                vis_output = visualizer.draw_instance_predictions(
                    predictions=result.to(torch.device("cpu")),
                    mask_threshold=self.object_mask_threshold
                )
            elif self.mode == "panoptic":
                panoptic_seg, segments_info = self.panoptic_inference(mask_cls, mask_pred)
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(torch.device("cpu")), segments_info
                )
            else:
                raise ValueError(f"The provided model.mode ({self.mode}) is not supported.")
            cv2.imwrite(
                os.path.join(
                    self.cfg.inference.results_dir,
                    batch_info[i]["filename"] + ".jpg"),
                vis_output.get_image()
            )

    def forward(self, inputs, tasks=None, texts=None):
        """Forward pass through the model."""
        outputs = self.model(inputs, tasks=tasks, texts=texts)
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
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def instance_inference(self, mask_cls, mask_pred):
        """Post process for instance segmentation."""
        image_size = mask_pred.shape[-2:]
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, 1:]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

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

    def batch_semantic_inference(self, mask_cls, mask_pred):
        """Perform batch semantic inference."""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "oneformer"
