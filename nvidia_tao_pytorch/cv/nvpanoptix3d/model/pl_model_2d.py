# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Main PTL 2D stage model for NVPanoptix3D."""

import os
import json
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.mask2former.utils.d2.catalog import MetadataCatalog

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_2d import MaskFormerModel
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.helper import (
    get_metadata,
    freeze_modules,
    get_kept_mapping,
    configure_optimizers,
    visualize_2d_predictions
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.evaluation_2d import MetricsManager


class Mask2formerPlModule(TAOLightningModule):
    """Lightning wrapper for the 2D MaskFormer stage.

    Provides training/validation/test loops, loss setup, checkpoint loading, and
    evaluation utilities for the 2D panoptic + depth model. This module handles
    target preparation, occupancy target generation, and metric logging.
    """

    def __init__(self, cfg) -> None:
        """
        Initialize 2D module for 3D Panoptic Reconstruction model.

        Constructs 2D MaskFormer model, criterion, and metadata. Sets up logging.

        Args:
            cfg (OmegaConfig): Hydra config containing model/dataset parameters.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.checkpoint_filename = "panoptic_recon_3d_model"
        self.num_classes = self.cfg.model.sem_seg_head.num_classes
        self.num_queries = self.cfg.model.mask_former.num_object_queries
        self.mode = self.cfg.model.mode.lower()
        self.test_topk_per_image = self.cfg.model.test_topk_per_image
        self.overlap_threshold = self.cfg.model.overlap_threshold
        self.object_mask_threshold = self.cfg.model.object_mask_threshold
        # For loading flexibility between 2D & 3D, set strict loading to False
        self.strict_loading = False

        self._build_model()
        self._build_criterion()
        self.status_logging_dict = {}

        metadata = get_metadata(self.cfg)
        if "custom" in MetadataCatalog:
            MetadataCatalog.remove("custom")
        self.metadata = MetadataCatalog.get("custom").set(
            thing_classes=metadata["thing_classes"],
            thing_colors=metadata["thing_colors"],
            stuff_classes=metadata["stuff_classes"],
            stuff_colors=metadata["stuff_colors"],
            thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
            stuff_dataset_id_to_contiguous_id=metadata["stuff_dataset_id_to_contiguous_id"],
            class_info=metadata["class_info"],
        )

        self.dataset = self.cfg.dataset.name.lower()
        self.kept = None
        self.mapping = None
        self._init_metrics()

    def _build_model(self):
        """
        Internal function to build the model.

        Initializes MaskFormerModel with cfg and applies optional freezing
        specified in cfg.train.freeze.
        """
        self.model = MaskFormerModel(self.cfg)

        if self.cfg.train.checkpoint_2d:
            self.load_2d_checkpoint(self.cfg.train.checkpoint_2d)

        # freeze modules
        if self.cfg.train.freeze:
            freeze_modules(self.model, self.cfg.train.freeze, status_logging)

    def _build_criterion(self):
        """
        Internal function to build the criterion.

        Constructs a SetCriterion for MaskFormer training with classification,
        mask, dice, depth, and multiplane occupancy losses. Also builds the Hungarian
        matcher for bipartite assignment.
        """
        # Loss parameters:
        deep_supervision = self.cfg.model.mask_former.deep_supervision
        no_object_weight = self.cfg.model.mask_former.no_object_weight

        # loss weights
        class_weight = self.cfg.model.mask_former.class_weight
        dice_weight = self.cfg.model.mask_former.dice_weight
        mask_weight = self.cfg.model.mask_former.mask_weight
        depth_weight = self.cfg.model.mask_former.depth_weight
        mp_occ_weight = self.cfg.model.mask_former.mp_occ_weight

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.cfg.model.mask_former.train_num_points,
            use_point_sample=True,
        )

        weight_dict = {
            "loss_ce": class_weight, "loss_mask": mask_weight,
            "loss_dice": dice_weight, "loss_depth": depth_weight,
            "loss_mp_occ": mp_occ_weight
        }
        if deep_supervision:
            dec_layers = self.cfg.model.mask_former.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "depths"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=self.cfg.model.mask_former.train_num_points,
            oversample_ratio=self.cfg.model.mask_former.oversample_ratio,
            importance_sample_ratio=self.cfg.model.mask_former.importance_sample_ratio,
            use_point_sample=True,
        )

    def configure_optimizers(self):
        """Configure optimizers and LR schedulers for training.

        Returns:
            Optimizer or (optimizers, schedulers) as expected by PyTorch Lightning.
        """
        return configure_optimizers(self.cfg, self.model)

    def load_2d_checkpoint(self, checkpoint_path):
        """Load a 2D-stage checkpoint into the underlying model.

        Args:
            checkpoint_path: Path to a checkpoint file on disk.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning(
                f"2D checkpoint path {checkpoint_path} does not exist. Skipping 2D weight loading."
            )
            return

        logging.info(f"Loading 2D checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter and map the 2D model weights
        updated_state_dict = {}

        for key, value in state_dict.items():
            # Remove the 'model.' prefix from Lightning checkpoint keys
            if key.startswith("model."):
                new_key = key[len("model."):]

                # Only load 2D components (exclude 3D-specific components)
                if not any(exclude_key in new_key for exclude_key in [
                    "reprojection", "completion", "projector", "ol"  # 3D-specific modules
                ]):
                    updated_state_dict[new_key] = value

        # Load the filtered weights into the model
        missing_keys, _ = self.model.load_state_dict(updated_state_dict, strict=False)

        if missing_keys:
            logging.info(f"Missing keys (expected for 3D components): {len(missing_keys)} keys")
            logging.debug(f"Missing keys: {missing_keys[:10]}...")

        logging.info("2D checkpoint loaded successfully into 3D model.")

    def on_load_checkpoint(self, checkpoint):
        """Hook called by Lightning after loading a checkpoint.

        When resuming training, we update the scheduler state's max_iters to
        match the current config. This allows extending training beyond the
        original run's max steps without rebuilding the checkpoint.

        Args:
            checkpoint (dict): Lightning checkpoint dictionary. If present, this method updates
                checkpoint["lr_schedulers"][*]["max_iters"] in-place to match the current
                experiment config (self.cfg.train.optim.max_steps).
        """
        super().on_load_checkpoint(checkpoint)
        # When resuming, update scheduler's max_iters to match current config
        # This allows extending training beyond original max_steps
        if "lr_schedulers" in checkpoint:
            for sched_state in checkpoint["lr_schedulers"]:
                if "max_iters" in sched_state:
                    sched_state["max_iters"] = self.cfg.train.optim.max_steps

    def prepare_targets(self, batch, outputs):
        """
        Prepare targets for loss calculation.

        Converts per-image instance annotations into a list of targets, resizing
        masks/depths to (H_pad, W_pad) to align with model outputs.

        Args:
            batch (dict):
                - image: (B, C, H_pad, W_pad)
                - instances: list of dicts per image with keys gt_masks, gt_classes,
                  and optionally gt_depths.
            outputs (dict): Outputs from model.
                - orig_pad_shape: Padded shape (H_pad, W_pad)
                - resized_shape: Resized shape (H_resized, W_resized)

        Returns:
            list[dict]: per-image targets with keys 'labels', 'masks', optionally 'depths'.
        """
        height, width = outputs["outputs"]["orig_pad_shape"]  # Padded resolution (e.g., 256×320)
        targets = []
        for x in batch["instances"]:
            instance_dict = {}
            for key, value in x.items():
                if isinstance(value, torch.Tensor):
                    instance_dict[key] = value.to(self.device)
                else:
                    instance_dict[key] = value
            targets.append(instance_dict)

        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image["gt_masks"]  # Shape: (N, H_orig, W_orig)

            if gt_masks.shape[0] > 0:
                # Check if masks need resizing (different from padded resolution)
                if (gt_masks.shape[1] / gt_masks.shape[2]) != (height / width):
                    # Interpolate masks to padded resolution using nearest mode
                    # to preserve binary mask values
                    resized_masks = self.model.model_input_resize.apply_image(gt_masks)
                else:
                    resized_masks = gt_masks
            else:
                resized_masks = torch.zeros(
                    (0, height, width), dtype=gt_masks.dtype, device=gt_masks.device
                )

            target = {
                "labels": targets_per_image["gt_classes"],
                "masks": resized_masks,
                # Per-sample loss weight (e.g., down-weight augmented samples).
                "loss_weight": float(targets_per_image.get("loss_weight", 1.0)),
            }

            gt_depths = targets_per_image.get("gt_depths", None)

            if gt_depths is not None:
                if gt_depths.shape[0] > 0:
                    # Also interpolate depths to padded resolution for consistency
                    if (gt_depths.shape[1] != height) or (gt_depths.shape[2] != width):
                        # Pad depth GT with 0 (invalid depth), not 128.
                        resized_depths = self.model.model_input_resize.apply_segmentation(gt_depths, pad_value=0)
                    else:
                        resized_depths = gt_depths
                    # Use raw metric depth - batch-level scale alignment in loss handles
                    # the scale difference between relative predictions and metric targets
                    target["depths"] = resized_depths
                else:
                    target["depths"] = torch.zeros(
                        (1, height, width), dtype=gt_depths.dtype,
                        device=gt_depths.device
                    )
            new_targets.append(target)
        return new_targets

    def gen_occ2d_tg(self, batched_inputs, kept, mapping, occupancy_pred):
        """
        Generate occupancy targets for 2D panoptic segmentation.

        Uses back-projection results to rasterize 3D occupancy ground truth into
        multi-plane occupancy targets aligned with model predictions.

        Args:
            batched_inputs (dict): Batch dict containing 'mp_occ_256' and 'transformed_depth'.
            kept (Tensor): Boolean mask from back-projection, shape (B, 256, 256, 256),
                indicating which voxels are within the viewing frustum.
            mapping (Tensor): Voxel coordinate mapping from back-projection, shape
                (B, 256, 256, 256, 5) with channels [batch_idx, x, y, z, depth].
            occupancy_pred (Tensor): Predicted occupancy tensor (B, C_occ, Dz, Hy, Wx).

        Returns:
            dict: {'occupancy': (B, C_occ, Dz, Hy, Wx), 'depth_map': (B, 1, Hd, Wd)}
        """
        ret = {}
        batch_size = kept.shape[0]
        kept = kept.to(self.device)
        mapping = mapping.to(self.device)

        mapping_kept = mapping[kept]

        mapping_kept[:, -1] = mapping_kept[:, -1] * 100 / self.cfg.dataset.depth_max
        mapping_kept = mapping_kept.long()

        depth_occ_tg = torch.zeros(
            (batch_size, occupancy_pred.shape[1]) +
            tuple(self.cfg.dataset.reduced_target_size[::-1]), device=self.device
        )

        occupancy = batched_inputs["mp_occ_256"].to(self.device)

        depth_occ_tg[
            mapping_kept[:, 0], mapping_kept[:, -1],
            mapping_kept[:, 2], mapping_kept[:, 1]
        ] = occupancy[kept].float()

        depth_occ_tg = F.max_pool3d(depth_occ_tg, 3, 1, 1)
        ret["occupancy"] = depth_occ_tg

        depth_tg = batched_inputs["transformed_depth"].unsqueeze(1)
        ret["depth_map"] = depth_tg  # add depth_map to calculate multi-plane occupancy loss

        return ret

    def _init_metrics(self):
        """Initialize PQ/DPQ metrics and metadata.

        Loads the dataset label map and constructs per-class metadata needed by
        MetricsManager. Metrics are computed on CPU by default.
        """
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = self.device if hasattr(self, "device") else torch.device("cuda")

        with open(self.cfg.dataset.label_map, "r", encoding="utf-8") as f:
            categories = json.load(f)

        if not self.cfg.dataset.contiguous_id:
            categories_full = [
                {"name": "nan", "color": [0, 0, 0], "isthing": 1, "id": i + 1}
                for i in range(self.num_classes)
            ]
            for cat in categories:
                if "trainId" not in cat:
                    cat["trainId"] = cat.get("id", 0)
                categories_full[cat["id"] - 1] = cat
            categories = categories_full
        else:
            for cat in categories:
                if "trainId" not in cat:
                    cat["trainId"] = cat.get("id", 0)

        self.pq_meta = SimpleNamespace(class_info=categories)
        self.metrics = MetricsManager(self.pq_meta, device=device)

    def _update_metrics(self, batch, outputs, mode="train", log_step=False):
        """Update metric accumulators and optionally log per-step metrics.

        This is a thin wrapper around self.metrics.update(...) that:
        - updates internal metric state for the given mode (train/val/test),
        - if log_step=True, logs per-step metrics for each threshold bucket returned by the
          metrics manager.

        Args:
            batch (dict): Current batch. Must contain batch["image"] so we can pass the
                appropriate batch_size to Lightning logging.
            outputs (dict): Model outputs for the batch (passed through to the metrics manager).
            mode (str): One of "train", "val", or "test".
            log_step (bool): If True, log step-level metrics (in addition to updating state).

        Returns:
            None. (Step metrics are logged via self.log, and epoch metrics are produced in
            common_epoch_end().)
        """
        # Compute and log batch-level metrics for immediate feedback (all modes)
        metric_out = self.metrics.update(batch, outputs, mode=mode, return_stats=log_step)
        if log_step:
            prog_bar_metric = ["-1.0", "averages"]
            for threshold in metric_out.keys():
                threshold_str = str(threshold)
                is_prog_bar = threshold_str in prog_bar_metric
                for metric_key in metric_out[threshold].keys():
                    value = metric_out[threshold].get(metric_key, None)
                    if value is not None:
                        self.log(
                            f"{mode}_{metric_key}_{threshold_str}_step",
                            value,
                            prog_bar=is_prog_bar,
                            sync_dist=True,
                            batch_size=batch["image"].shape[0]
                        )

    def common_epoch_end(self, mode="train"):
        """Compute and log epoch-level metrics.

        Args:
            mode: One of "train", "val", or "test".
        """
        average_loss = self.trainer.logged_metrics.get(f"{mode}_loss_epoch", 0.0)
        if hasattr(average_loss, "item"):
            average_loss = average_loss.item()
        self.status_logging_dict = {}
        self.status_logging_dict[f"{mode}_loss"] = average_loss

        # Compute and log pq/dpq
        # Compute and log epoch-level metrics
        if self.metrics is not None:
            out = self.metrics.compute(mode)
        else:
            out = {}

        prog_bar_metric = ["-1.0", "averages"]
        for threshold in out.keys():
            threshold_str = str(threshold)
            is_prog_bar = threshold_str in prog_bar_metric
            for metric_key in out[threshold].keys():
                value = out[threshold].get(metric_key, 0.0)
                self.log(
                    f"{mode}_{metric_key}_{threshold_str}_epoch",
                    value,
                    prog_bar=is_prog_bar,
                    sync_dist=True
                )
                self.status_logging_dict[f"{mode}_{metric_key}_{threshold_str}_epoch"] = value.item()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message=f"{mode} metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

        # Reset metrics
        self.metrics.reset_epoch_metrics(mode)

    def common_step(self, batch, batch_idx, mode="train"):
        """
        Common step for train/val/test.

        Runs forward pass, prepares targets, computes losses and logs metrics.
        Uses back-projection helper to generate kept/mapping for occupancy supervision.

        Args:
            batch (dict): Input batch as prepared by dataset/preprocessor.
                Expected keys: 'image', 'instances', 'mp_occ_256', 'transformed_depth',
                'frustum_mask', 'intrinsic'.
            batch_idx (int): Batch index.
            mode (str): One of {'train','val','test'}.

        Returns:
            Tensor: Total loss tensor for optimization/logging.
        """
        inputs = batch["image"]
        batch_size = inputs.shape[0]

        outputs = self.model(
            inputs, post_process=True,
            nopad_image_shape=batch.get("nopad_image_shape", None),
            room_mask=batch.get("room_mask", None),
        )

        # Calculate validation loss
        # kept: (B, 256, 256, 256) boolean mask for valid frustum voxels
        # mapping: (B, 256, 256, 256, 5) with channels [batch_idx, x, y, z, depth]
        kept, mapping = get_kept_mapping(self.model, self.cfg, batch, self.device)

        # Gen mp occupancy targets
        occupancy_preds = outputs["outputs"]["occupancy_preds"]
        with torch.no_grad():
            occupancy_targets = self.gen_occ2d_tg(batch, kept, mapping, occupancy_preds)

        # Calculate & update loss
        targets = self.prepare_targets(batch, outputs)
        losses = self.criterion(outputs["outputs"], targets, occupancy_preds, occupancy_targets)
        weight_dict = self.criterion.weight_dict

        loss_total = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)
        self.log(
            f"{mode}_loss", loss_total, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True, batch_size=batch_size
        )

        # Only log detail loss for train
        if mode == "train":
            self.log(
                "loss_dice", losses["loss_dice"], on_step=True, on_epoch=False,
                prog_bar=True, sync_dist=True, batch_size=batch_size
            )
            self.log(
                "loss_ce", losses["loss_ce"], on_step=True, on_epoch=False,
                prog_bar=False, sync_dist=True, batch_size=batch_size
            )
            self.log(
                "loss_mask", losses["loss_mask"], on_step=True, on_epoch=False,
                prog_bar=False, sync_dist=True, batch_size=batch_size
            )
            self.log(
                "loss_mp_occ", losses["loss_mp_occ"], on_step=True, on_epoch=False,
                prog_bar=True, sync_dist=True, batch_size=batch_size
            )
            self.log(
                "loss_depth", losses["loss_depth"], on_step=True, on_epoch=False,
                prog_bar=True, sync_dist=True, batch_size=batch_size
            )
            self.log(
                "lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False,
                prog_bar=True, sync_dist=True
            )

        self._update_metrics(batch, outputs["processed_outputs"], mode=mode, log_step=True)

        return loss_total

    def training_step(self, batch, batch_idx):
        """Lightning training step (delegates to common_step)."""
        return self.common_step(batch, batch_idx, mode="train")

    def on_train_epoch_end(self):
        """Lightning hook to finalize training epoch metrics."""
        self.common_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        """Lightning validation step (delegates to common_step)."""
        return self.common_step(batch, batch_idx, mode="val")

    def on_validation_epoch_end(self):
        """Lightning hook to finalize validation epoch metrics."""
        self.common_epoch_end("val")

    def test_step(self, batch, batch_idx):
        """Lightning test step (delegates to common_step)."""
        return self.common_step(batch, batch_idx, mode="test")

    def on_test_epoch_end(self):
        """Lightning hook to finalize test epoch metrics."""
        return self.common_epoch_end("test")

    def forward(self, x):
        """Forward pass."""
        outputs = self.model(x)
        return outputs

    def predict_step(self, batch, batch_idx):
        """
        Predict step for inference.

        Runs forward pass with postprocessing and saves 2D predictions.

        Args:
            batch (dict): Input batch with 'image' and 'image_id' keys.
            batch_idx (int): Batch index.

        Returns:
            list: Processed outputs with panoptic_seg, depth, and semantic_seg (stored under "sem_seg").
        """
        inputs = batch["image"]

        with torch.no_grad():
            outputs = self.model(
                inputs,
                post_process=True,
                nopad_image_shape=batch.get("nopad_image_shape", None),
                room_mask=batch.get("room_mask", None),
            )
        processed_outputs = outputs["processed_outputs"]

        return processed_outputs

    def visualize_predictions(self, batch, processed_outputs):
        """
        Helper to save 2D predictions for each image in batch.

        Args:
            batch (dict): Batch containing 'image' and 'image_id'.
            processed_outputs (list): List of processed result dicts.
        """
        output_dir = getattr(self.cfg.inference, "results_dir", None)
        if output_dir is None:
            return

        batch_size = batch["image"].shape[0]
        for i in range(batch_size):
            frame_name = batch["image_id"][i] if "image_id" in batch else f"frame_{i}"
            visualize_2d_predictions(
                output_dir=output_dir,
                frame_name=frame_name,
                processed_output=processed_outputs[i],
                image=batch["image"][i],
            )
