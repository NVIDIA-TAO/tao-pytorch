# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main PTL 3D stage model for NVPanoptix3D."""

import os
import torch
from tabulate import tabulate

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.mask2former.utils.d2.catalog import MetadataCatalog

from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.helper import (
    get_metadata,
    freeze_modules,
    clear_cuda_cache,
    get_kept_mapping,
    configure_optimizers,
)
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_3d import NVPanoptix3DModel
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.evaluation_3d import PanopticReconstructionQuality
from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class NVPanoptix3DPlModule(TAOLightningModule):
    """Lightning wrapper for the 3D NVPanoptix3D stage.

    Orchestrates training and evaluation for the 3D frustum completion model,
    including criterion construction, checkpoint loading (2D/3D), and PRQ metric
    computation.
    """

    def __init__(self, cfg) -> None:
        """Initialize the 3D Lightning module.

        Args:
            cfg: Hydra/OmegaConfig configuration containing model, dataset, and
                training parameters.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.checkpoint_filename = "nvpanoptix3d_model"
        self.num_classes = self.cfg.model.sem_seg_head.num_classes
        self.num_queries = self.cfg.model.mask_former.num_object_queries
        self.mode = self.cfg.model.mode.lower()
        self.test_topk_per_image = self.cfg.model.test_topk_per_image
        self.overlap_threshold = self.cfg.model.overlap_threshold
        self.object_mask_threshold = self.cfg.model.object_mask_threshold

        self._build_model()
        self._build_criterion()

        self.dataset = self.cfg.dataset.name.lower()
        self.kept = None
        self.mapping = None

        # Initialize 3D evaluation metrics
        self.metadata = get_metadata(self.cfg)
        if "custom" in MetadataCatalog:
            MetadataCatalog.remove("custom")
        self.metadata = MetadataCatalog.get("custom").set(
            thing_classes=self.metadata["thing_classes"],
            thing_colors=self.metadata["thing_colors"],
            stuff_classes=self.metadata["stuff_classes"],
            stuff_colors=self.metadata["stuff_colors"],
            thing_dataset_id_to_contiguous_id=self.metadata["thing_dataset_id_to_contiguous_id"],
            stuff_dataset_id_to_contiguous_id=self.metadata["stuff_dataset_id_to_contiguous_id"],
            class_info=self.metadata["class_info"],
        )

        self.pq_evaluator = PanopticReconstructionQuality(
            self.metadata,
            matching_threshold=0.25,
            ignore_labels=[0, 12],
            reduction="mean"
        )

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

    def load_3d_checkpoint(self, checkpoint_path):
        """Load a 3D-stage checkpoint into the underlying model.

        Args:
            checkpoint_path: Path to a checkpoint file on disk.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning(
                f"3D checkpoint path {checkpoint_path} does not exist. Skipping 3D weight loading."
            )
            return

        logging.info(f"Loading 3D checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter and map the 3D model weights
        updated_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[len("model."):]
                if any(include_key in new_key for include_key in [
                    "reprojection", "completion", "projector", "ol"
                ]):
                    updated_state_dict[new_key] = value

        self.model.load_state_dict(updated_state_dict, strict=False)

    def _build_model(self):
        """Build the underlying NVPanoptix3DModel and optionally freeze modules."""
        self.model = NVPanoptix3DModel(self.cfg)

        if self.cfg.train.checkpoint_2d != "" and self.training:
            self.load_2d_checkpoint(self.cfg.train.checkpoint_2d)
        if self.cfg.train.checkpoint_3d != "" and self.training:
            self.load_3d_checkpoint(self.cfg.train.checkpoint_3d)

        # freeze modules
        if self.cfg.train.freeze:
            freeze_modules(self.model, self.cfg.train.freeze, status_logging)

    def _build_criterion(self):
        """Build criterion and matcher for 3D losses.

        Constructs a Hungarian matcher and a SetCriterion configured for the
        3D stage (geometry/occupancy/panoptic losses).
        """
        # loss weights:
        class_weight = self.cfg.model.mask_former.class_weight
        dice_weight = self.cfg.model.mask_former.dice_weight
        mask_weight = self.cfg.model.mask_former.mask_weight
        no_object_weight = self.cfg.model.mask_former.no_object_weight

        panoptic_weight = self.cfg.model.frustum3d.panoptic_weight
        occupancy_weights = self.cfg.model.frustum3d.completion_weights
        geometry_weight = self.cfg.model.frustum3d.surface_weight

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.cfg.model.mask_former.train_num_points,
            use_point_sample=True,
        )

        weight_dict = {
            "loss_geometry": geometry_weight,
        }

        for idx, lvl in enumerate([64, 128, 256]):
            weight_dict[f"loss_occupancy_{lvl}"] = occupancy_weights[idx]
            weight_dict[f"loss_panoptic_{lvl}"] = panoptic_weight

        losses = ["geometry", "occupancy", "panoptic"]

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
        """Configure optimizers and LR schedulers for 3D training.

        Returns:
            Optimizer or (optimizers, schedulers) as expected by PyTorch Lightning.
        """
        return configure_optimizers(self.cfg, self.model)

    def training_step(self, batch, batch_idx):
        """Lightning training step for the 3D stage.

        Runs the 3D model forward (without postprocessing), prepares targets, and
        computes configured 3D losses.

        Args:
            batch: Input batch dict.
            batch_idx: Batch index.

        Returns:
            Scalar total loss tensor.
        """
        kept, mapping = get_kept_mapping(self.model, self.cfg, batch, self.device)
        outputs_3d = self.model(
            batch, kept, mapping, postprocess=False, is_matterport=self.dataset == "matterport"
        )
        targets = self.prepare_targets(batch)

        # calculate losses
        losses = self.criterion(outputs_3d, targets)
        weight_dict = self.criterion.weight_dict

        loss_total = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)
        self.log(
            "train_loss", loss_total,
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )
        self.log(
            "geo", losses["loss_geometry"],
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )
        self.log(
            "occ", losses["loss_occupancy_256"],
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )
        self.log(
            "panop", losses["loss_panoptic_256"],
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )
        self.log(
            "lr", self.lr_schedulers().get_last_lr()[-1],
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )

        clear_cuda_cache()
        return loss_total

    def on_validation_epoch_start(self) -> None:
        """Lightning hook at the start of a validation epoch."""
        self.validation_outputs = []
        self.validation_metrics = 0

    def validation_step(self, batch, batch_idx):
        """Lightning validation step for PRQ evaluation.

        Runs the model with postprocessing to obtain per-sample instance info and
        accumulates PRQ statistics across the batch.
        """
        clear_cuda_cache()
        batch_size = batch["image"].shape[0]
        kept, mapping = get_kept_mapping(self.model, self.cfg, batch, self.device)

        with torch.no_grad():
            # run the model with postprocessing to get 3D results
            outputs_3d = self.model(
                batch, kept, mapping, postprocess=True, is_matterport=self.dataset == "matterport"
            )

        # calculate 3D panoptic metrics for each sample in the batch
        for batch_idx_item in range(batch_size):
            pred_instance_info = outputs_3d[batch_idx_item]["instance_info_pred"]
            gt_instance_info = batch["instance_info_gt"][batch_idx_item]

            # Calculate PQ, SQ, RQ metrics for this sample
            sample_metrics = self.pq_evaluator.add(pred_instance_info, gt_instance_info)
            per_sample_metrics_tensor = self.pq_evaluator.convert_metric_to_tensor(sample_metrics)
            self.validation_metrics += per_sample_metrics_tensor

        # prepare validation metrics
        val_metrics = {
            "val_loss": 0.0
        }
        self.validation_outputs.append(val_metrics)
        return val_metrics

    def on_validation_epoch_end(self) -> None:
        """Lightning hook to finalize PRQ metrics for the validation epoch.

        Aggregates per-sample metrics (and across distributed workers when
        applicable), logs PQ/SQ/RQ, prints per-class metrics on rank 0, and resets
        internal accumulators.
        """
        if len(self.validation_outputs) == 0:
            return
        # check if train on multiple GPUs:
        if is_dist_avail_and_initialized():
            # gather across GPUs → every process gets the same all_batch_metrics_tensor
            all_batch_metrics_tensors = self.all_gather(self.validation_metrics)
            all_batch_metrics_tensors = all_batch_metrics_tensors.sum(dim=0)
        else:
            all_batch_metrics_tensors = self.validation_metrics

        # revert to dict metric:
        all_batch_metrics = self.pq_evaluator.revert_metric_to_dict(all_batch_metrics_tensors)
        # reduce mean:
        all_batch_metrics = self.pq_evaluator.reduce_mean(all_batch_metrics)

        # Log metrics to progress bar:
        self.log("PRQ", all_batch_metrics["pq"] * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("RSQ", all_batch_metrics["sq"] * 100, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("RRQ", all_batch_metrics["rq"] * 100, on_epoch=True, prog_bar=True, sync_dist=True)

        s_logger = status_logging.get_status_logger()
        s_logger.kpi = {
            "PRQ": all_batch_metrics["pq"] * 100,
            "RSQ": all_batch_metrics["sq"] * 100,
            "RRQ": all_batch_metrics["rq"] * 100,
        }
        s_logger.write(
            message="Validation metrics generated.",
            status_level=status_logging.Status.RUNNING,
        )

        classes = self.pq_evaluator.class_id_to_name.values()

        # print metrics only once (from global rank 0)
        if self.global_rank == 0:
            summary_table = [[
                all_batch_metrics["pq"] * 100,
                all_batch_metrics["sq"] * 100,
                all_batch_metrics["rq"] * 100,
                all_batch_metrics["n"],
            ]]
            print("Result:")
            print(tabulate(
                summary_table,
                headers=["PQ", "SQ", "RQ", "N"],
                tablefmt="rst",
                floatfmt=".2f",
            ))

        if self.global_rank == 0:
            class_table = []
            for cls in classes:
                if cls not in all_batch_metrics:
                    continue
                class_table.append([
                    cls,
                    all_batch_metrics[cls]["pq"] * 100,
                    all_batch_metrics[cls]["sq"] * 100,
                    all_batch_metrics[cls]["rq"] * 100,
                    all_batch_metrics[cls]["n"],
                ])
            if class_table:
                print("Per-class metrics:")
                print(tabulate(
                    class_table,
                    headers=["Class", "PQ", "SQ", "RQ", "N"],
                    tablefmt="rst",
                    floatfmt=".2f",
                ))

        # reset:
        self.validation_outputs = []
        self.validation_metrics = 0

    def on_test_epoch_start(self):
        """Lightning hook at the start of a test epoch (reuses validation setup)."""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Lightning test step (reuses validation_step)."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Lightning hook at the end of a test epoch (reuses validation teardown)."""
        self.on_validation_epoch_end()

    def forward(self, x):
        """Forward pass."""
        outputs = self.model(x)
        return outputs

    def predict_step(self, batch, batch_idx):
        """
        Predict step for inference.

        Runs forward pass with postprocessing and returns both 2D and 3D predictions.

        Args:
            batch (dict): Input batch with 'image' and 'image_id' keys.
            batch_idx (int): Batch index.

        Returns:
            list[dict]: Per-sample dict containing both 2D and 3D outputs, including:
                - panoptic_seg_2d, depth, intrinsic, image_size
                - geometry, panoptic_seg, semantic_seg, panoptic_semantic_mapping, instance_info_pred
        """
        kept, mapping = get_kept_mapping(self.model, self.cfg, batch, self.device)

        with torch.no_grad():
            outputs = self.model(
                batch,
                kept,
                mapping,
                postprocess=True,
                is_matterport=self.dataset == "matterport",
            )
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Lightning hook to add TAO model identifier to the checkpoint."""
        checkpoint["tao_model"] = "panoptic_recon_3d"

    def prepare_targets(self, batched_inputs):
        """Prepare 3D-stage targets from a batched input dict.

        Args:
            batched_inputs: Batch dict with keys such as image, instances,
                occupancy_*, weighting3d_*, and geometry.

        Returns:
            List of per-sample target dicts consumed by SetCriterion.
        """
        images = batched_inputs["image"]
        h_pad, w_pad = images.shape[-2:]
        batch_size = images.shape[0]
        new_targets = []

        for x in range(batch_size):
            inst_per_image = batched_inputs["instances"][x]
            # pad gt
            gt_masks = inst_per_image["gt_masks"]
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_depths = inst_per_image["gt_depths"]
            padded_depths = torch.zeros(
                (gt_depths.shape[0], h_pad, w_pad),
                dtype=gt_depths.dtype,
                device=gt_depths.device
            )
            padded_depths[:, : gt_depths.shape[1], : gt_depths.shape[2]] = gt_depths

            target = {
                "labels": inst_per_image["gt_classes"].to(self.device),
                "masks": padded_masks.to(self.device),
                "depths": padded_depths.to(self.device),
                "masks_3d_256": inst_per_image["gt_masks_3d_256"].to(self.device),
                "masks_3d_128": inst_per_image["gt_masks_3d_128"].to(self.device),
                "masks_3d_64": inst_per_image["gt_masks_3d_64"].to(self.device),
                "occupancy_256": batched_inputs["occupancy_256"][x].to(self.device),
                "occupancy_128": batched_inputs["occupancy_128"][x].to(self.device),
                "occupancy_64": batched_inputs["occupancy_64"][x].to(self.device),
                "weighting3d_256": batched_inputs["weighting3d_256"][x].to(self.device),
                "weighting3d_128": batched_inputs["weighting3d_128"][x].to(self.device),
                "weighting3d_64": batched_inputs["weighting3d_64"][x].to(self.device),
                "geometry": batched_inputs["geometry"][x].to(self.device),
                # Per-sample loss weight (e.g., down-weight augmented samples).
                "loss_weight": float(inst_per_image.get("loss_weight", 1.0)),
            }
            new_targets.append(target)
        return new_targets
