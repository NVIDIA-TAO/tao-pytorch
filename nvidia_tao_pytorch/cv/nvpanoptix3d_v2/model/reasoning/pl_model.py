# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for NVPanoptix3Dv2-Reasoning projector training."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import (
    MetricScaleHead,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.qwen_builder import (
    build_qwen,
    cfg_get as _cfg,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.reasoning_model import (
    NVPanoptix3Dv2Reasoning,
    SegToSAMPromptProjector,
    set_requires_grad,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.masks import (
    gather_view_masks,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.score import (
    CanonicalPointCloudMetrics,
    score_batch_on_canonical_points,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.losses import (
    NVPanoptix3Dv2ReasoningLoss,
)

logger = logging.getLogger(__name__)


def build_sam3_image_model_from_config(model_cfg):
    """Build the frozen SAM3 image model through the shared TAO backbone helper."""
    from nvidia_tao_pytorch.cv.backbone_v2.sam3 import get_sam3_model

    reasoning_cfg = _cfg(model_cfg, "reasoning", {})
    sam_cfg = _cfg(reasoning_cfg, "sam3", {})
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")) or 0)
        torch.cuda.set_device(local_rank)
    checkpoint = _cfg(sam_cfg, "checkpoint", None)
    load_from_hf = bool(_cfg(sam_cfg, "load_from_hf", True)) and not checkpoint
    sam, _ = get_sam3_model(
        chk_base_path=str(checkpoint) if checkpoint else None,
        wrap=False,
        load_from_HF=load_from_hf,
    )
    set_requires_grad(sam, False)
    sam.eval()
    return sam


def filter_unsupported_vggt_weights(
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Drop the upstream tracking head, which is not vendored by this model."""
    filtered = {
        key: value for key, value in state.items()
        if not key.startswith("track_head.")
    }
    dropped = len(state) - len(filtered)
    if dropped:
        logger.info("Dropped %d unsupported VGGT tracking tensors", dropped)
    return filtered


def build_vggt_geometry_model_from_config(model_cfg):
    """Build the frozen VGGT backbone shared with the panoptic variant."""
    backbone_cfg = _cfg(model_cfg, "backbone", {})

    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt import VGGT

    vggt = VGGT(
        img_size=int(_cfg(model_cfg, "img_size", 518)),
        patch_size=int(_cfg(model_cfg, "patch_size", 14)),
        embed_dim=int(_cfg(model_cfg, "embed_dim", 1024)),
    )
    checkpoint = _cfg(backbone_cfg, "pretrained_backbone_path", None)
    if checkpoint and os.path.exists(str(checkpoint)):
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        state = filter_unsupported_vggt_weights(state)
        strict = bool(_cfg(backbone_cfg, "strict_load", True))
        incompatible = vggt.load_state_dict(state, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(
                "VGGT checkpoint loaded non-strictly: %d missing keys, %d unexpected keys",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
    elif bool(_cfg(backbone_cfg, "load_from_hf", False)):
        vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {checkpoint!r}; set "
            "model.backbone.pretrained_backbone_path or enable "
            "model.backbone.load_from_hf."
        )
    set_requires_grad(vggt, False)
    vggt.eval()
    return vggt


def checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load and unwrap a model state dictionary from ``path``."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def build_metric_depth_head_from_config(model_cfg):
    """Build the shared metric scale head in its scale+shift configuration."""
    backbone_cfg = _cfg(model_cfg, "backbone", {})
    head_cfg = _cfg(backbone_cfg, "metric_depth_head", {})
    if not bool(_cfg(head_cfg, "enable", True)):
        return None
    hidden_dims = _cfg(head_cfg, "hidden_dims", None)
    if hidden_dims is None:
        hidden = int(_cfg(head_cfg, "hidden_dim", 256))
        hidden_dims = [hidden, max(hidden // 4, 16)]
    metric_head = MetricScaleHead(
        scene_token_dim=int(_cfg(head_cfg, "feat_dim", 2048)),
        hidden_dims=tuple(hidden_dims),
        metric_context_views=int(_cfg(head_cfg, "metric_context_views", 5)),
        predict_shift=bool(_cfg(head_cfg, "predict_shift", True)),
    )
    return metric_head


def build_reasoning_model_from_config(cfg) -> NVPanoptix3Dv2Reasoning:
    """Assemble Qwen + frozen SAM3 + frozen VGGT + the trainable projector."""
    model_cfg = _cfg(cfg, "model")
    reasoning_cfg = _cfg(model_cfg, "reasoning", {})
    qwen = build_qwen(_cfg(reasoning_cfg, "qwen", {}))
    sam = build_sam3_image_model_from_config(model_cfg)
    vggt = build_vggt_geometry_model_from_config(model_cfg)
    metric_head = build_metric_depth_head_from_config(model_cfg)
    sam_cfg = _cfg(reasoning_cfg, "sam3", {})
    bridge_cfg = _cfg(reasoning_cfg, "sam_bridge", {})
    projector = SegToSAMPromptProjector(
        d_llm=qwen.hidden_size,
        d_sam=int(_cfg(bridge_cfg, "sam_prompt_dim", 256)),
        hidden=int(_cfg(bridge_cfg, "hidden_dim", 4096)),
        dropout=float(_cfg(bridge_cfg, "dropout", 0.0)),
    )
    conf_threshold = _cfg(reasoning_cfg, "point_conf_threshold", None)
    return NVPanoptix3Dv2Reasoning(
        qwen=qwen,
        sam3_image_model=sam,
        projector=projector,
        vggt_geometry_model=vggt,
        metric_depth_head=metric_head,
        sam_resolution=int(_cfg(sam_cfg, "resolution", 1008)),
        point_mask_threshold=float(_cfg(reasoning_cfg, "point_mask_threshold", 0.5)),
        point_conf_threshold=None if conf_threshold is None else float(conf_threshold),
        freeze_sam=True,
        freeze_vggt=True,
    )


def build_sam_criterion(cfg) -> NVPanoptix3Dv2ReasoningLoss:
    """Build the reasoning loss from ``train.reasoning`` and ``train.metric_depth``."""
    train_cfg = _cfg(cfg, "train")
    loss_cfg = _cfg(train_cfg, "reasoning", {})
    metric_cfg = _cfg(train_cfg, "metric_depth", {})
    return NVPanoptix3Dv2ReasoningLoss(
        text_weight=float(_cfg(loss_cfg, "text_weight", 1.0)),
        mask_weight=float(_cfg(loss_cfg, "mask_weight", 20.0)),
        dice_weight=float(_cfg(loss_cfg, "dice_weight", 1.0)),
        score_weight=float(_cfg(loss_cfg, "score_weight", 1.0)),
        metric_weight=float(_cfg(metric_cfg, "weight", 5.0)),
        metric_silog_weight=float(_cfg(metric_cfg, "silog_weight", 1.0)),
        metric_absrel_weight=float(_cfg(metric_cfg, "absrel_weight", 0.1)),
        metric_silog_lambda=float(_cfg(metric_cfg, "silog_lambda", 0.85)),
        metric_min_depth=float(_cfg(metric_cfg, "min_depth", 0.1)),
        metric_max_depth=float(_cfg(metric_cfg, "max_depth", 20.0)),
    )


class NVPanoptix3Dv2ReasoningPlModule(TAOLightningModule):
    """Reasoning-segmentation variant: Qwen ``[SEG]`` -> frozen SAM3 -> VGGT lift.

    Inherits from TAOLightningModule for the shared TAO checkpoint/status
    plumbing; the checkpoint policy itself is overridden in
    :meth:`configure_callbacks` because only trainable LoRA/projector rows
    are persisted.
    """

    def __init__(self, experiment_config):
        """Build the reasoning model, its criterion, and the eval metrics."""
        super().__init__(experiment_config)
        self.cfg = experiment_config
        self.train_cfg = _cfg(experiment_config, "train")
        self.model = build_reasoning_model_from_config(experiment_config)
        self.criterion = build_sam_criterion(experiment_config)
        self.checkpoint_filename = "nvpanoptix3d_v2_reasoning_model"
        # Best-checkpoint selection ranks on canonical point-cloud mIoU.
        self.monitor_metric = "val/mIoU"
        self.monitor_mode = "max"
        self.strict_loading = False
        self._log_interval = int(_cfg(self.train_cfg, "log_interval", 50))

        resume_ckpt = _cfg(self.train_cfg, "resume_training_checkpoint_path", None)
        has_resume = bool(resume_ckpt and os.path.isfile(str(resume_ckpt)))
        if not has_resume:
            results_dir = _cfg(experiment_config, "results_dir", None)
            if results_dir:
                latest = os.path.join(
                    str(results_dir), self.checkpoint_filename + "_latest.pth",
                )
                has_resume = os.path.exists(latest)

        pretrained = _cfg(self.train_cfg, "pretrained_model_path", None)
        if pretrained and os.path.isfile(str(pretrained)) and not has_resume:
            self.load_pretrained_weights(str(pretrained))
        elif pretrained and has_resume:
            logger.info(
                "Skipping pretrained_model_path (resume checkpoint found; it takes priority)"
            )
        elif pretrained:
            raise FileNotFoundError(
                f"Reasoning pretrained_model_path does not exist: {pretrained}"
            )

        # Canonical point-cloud validation metrics.
        eval_cfg = _cfg(experiment_config, "evaluate", {})
        self._eval_mask_thresh = float(_cfg(eval_cfg, "mask_threshold", 0.5))
        self._val_metrics: CanonicalPointCloudMetrics | None = None

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("NVPanoptix3Dv2-Reasoning: %.1fM total, %.1fM trainable", total / 1e6, trainable / 1e6)

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Warm-start compatible reasoning weights without optimizer state."""
        raw_state = checkpoint_state_dict(checkpoint_path)
        if not isinstance(raw_state, dict):
            raise TypeError(
                f"Expected a state dict in {checkpoint_path}, got "
                f"{type(raw_state).__name__}."
            )

        destination = self.state_dict()
        compatible = {}
        skipped_shape = []
        for source_key, value in raw_state.items():
            candidates = [source_key]
            if not source_key.startswith("model."):
                candidates.append("model." + source_key)
            target_key = next((key for key in candidates if key in destination), None)
            if target_key is None:
                continue
            if not hasattr(value, "shape") or value.shape != destination[target_key].shape:
                skipped_shape.append(target_key)
                continue
            compatible[target_key] = value

        if not compatible:
            raise ValueError(
                f"No compatible reasoning weights found in {checkpoint_path}."
            )
        self.load_state_dict(compatible, strict=False)
        logger.info(
            "Warm-started %d reasoning tensors from %s (%d shape mismatches skipped)",
            len(compatible), checkpoint_path, len(skipped_shape),
        )

    @staticmethod
    def tensors_to_pil(imgs: torch.Tensor) -> List[List[Image.Image]]:
        """``[B,S,3,H,W]`` in [0,1] -> list of PIL views for Qwen."""
        B, S = imgs.shape[:2]
        out: List[List[Image.Image]] = []
        for b in range(B):
            views = []
            for s in range(S):
                arr = (
                    imgs[b, s].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255.0
                )
                views.append(Image.fromarray(arr.astype(np.uint8)))
            out.append(views)
        return out

    def forward_batch(self, batch):
        """Run one reasoning batch and return its loss, logs, and outputs."""
        imgs = torch.stack([v["img"] for v in batch], dim=1).to(self.device)          # [B,S,3,H,W]
        pan_inst = torch.stack([v["pan_inst_id"] for v in batch], dim=1).to(self.device)  # [B,S,H,W]
        gt_depth = None
        if all("depthmap" in v for v in batch):
            gt_depth = torch.stack([v["depthmap"] for v in batch], dim=1).to(self.device)
        instruction = batch[0]["instruction"]
        answer = batch[0]["answer"]
        target_inst = batch[0]["target_inst_id"].to(self.device)
        seg_view_indices = batch[0].get("seg_view_indices") or [[0]] * len(instruction)

        qwen_inputs = self.model.qwen.build_inputs(
            self.tensors_to_pil(imgs),
            instruction,
            answer,
            device=self.device,
        )
        out = self.model(imgs, qwen_inputs, seg_view_indices=seg_view_indices)
        gt_sel = gather_view_masks(                                                    # [N,H,W]
            pan_inst, target_inst, out["seg_sample_idx"], out["seg_view_idx"]
        )
        mask_valid = target_inst[out["seg_sample_idx"]] > 0                            # [N]
        loss, logs = self.criterion(out, gt_sel, mask_valid, gt_depth=gt_depth)
        clouds = out.get("segmented_point_clouds")
        if clouds is not None:
            counts = [int(points.shape[0]) for points in clouds]
            logs["segmented_points_mean"] = float(sum(counts) / max(len(counts), 1))
        return loss, logs, out

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, logs, _ = self.forward_batch(batch)
        for k, v in logs.items():
            self.log(f"train/{k}", v, batch_size=1, prog_bar=False, sync_dist=True)
        if self.trainer.is_global_zero and batch_idx % self._log_interval == 0:
            logger.info(
                "step %d | total %.4f text %.4f mask %.4f dice %.4f score %.4f n_valid=%d",
                batch_idx,
                logs["loss_total"],
                logs["loss_text"],
                logs["loss_mask"],
                logs["loss_dice"],
                logs["loss_score"],
                int(logs["n_mask_valid"]),
            )
        return loss

    def on_validation_epoch_start(self):
        """Reset canonical point-cloud metrics for this validation epoch."""
        self._val_metrics = CanonicalPointCloudMetrics()

    def validation_step(self, batch, batch_idx):
        """Validation step with canonical point-cloud metric accumulation."""
        del batch_idx
        loss, logs, out = self.forward_batch(batch)
        for k, v in logs.items():
            self.log(f"val/{k}", v, batch_size=1, sync_dist=True)
        self.accumulate_val_metrics(batch, out)
        return loss

    def accumulate_val_metrics(self, batch, out):
        """Score this batch on canonical, metric-depth-defined points."""
        if self._val_metrics is None:
            return
        pan_inst = torch.stack(
            [view["pan_inst_id"] for view in batch], dim=1,
        ).to(self.device)
        depth = torch.stack(
            [view["depthmap"] for view in batch], dim=1,
        ).to(self.device)
        target_inst = batch[0]["target_inst_id"].to(self.device)
        canonical_valid = torch.isfinite(depth) & (depth > 0)
        score_batch_on_canonical_points(
            self._val_metrics,
            out,
            pan_inst,
            target_inst,
            canonical_valid,
            self._eval_mask_thresh,
        )

    def reduce_and_compute_metrics(self):
        """All-reduce additive state and compute metrics over the full split."""
        if self._val_metrics is None:
            return None
        stats = torch.tensor(
            self._val_metrics.raw_state(), dtype=torch.float64, device=self.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        self._val_metrics.load_raw_state(stats.tolist())
        return self._val_metrics.compute()

    def on_validation_epoch_end(self):
        """Log mIoU, mAP50, and mAP25 over the full validation split."""
        if self._val_metrics is None:
            return
        res = self.reduce_and_compute_metrics()
        if not res:
            return

        keys = ["mIoU", "mAP50", "mAP25"]
        for k in keys:
            self.log(
                f"val/{k}",
                float(res[k]),
                prog_bar=(k == "mIoU"),
                sync_dist=False,
            )
        if self.trainer.is_global_zero:
            summary = "  ".join(f"{key}={res[key]:.4f}" for key in keys)
            logger.info(
                "[NVPanoptix3Dv2-Reasoning][Ep %d][val] %s",
                self.current_epoch,
                summary,
            )

    def on_validation_start(self):
        """Skip the base dataloader size check when no val manifest is set."""
        if self.trainer.datamodule.val_dataloader() is None:
            return
        super().on_validation_start()

    def on_test_start(self):
        """Skip the base dataloader size check when no val manifest is set."""
        if self.trainer.datamodule.test_dataloader() is None:
            return
        super().on_test_start()

    def on_predict_start(self):
        """Skip the base dataloader size check when no val manifest is set."""
        if self.trainer.datamodule.predict_dataloader() is None:
            return
        super().on_predict_start()

    def on_test_epoch_start(self):
        """Reset canonical point-cloud metrics for the evaluate subtask."""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Evaluation step. Identical scoring to validation."""
        del batch_idx
        _, _, out = self.forward_batch(batch)
        self.accumulate_val_metrics(batch, out)

    def on_test_epoch_end(self):
        """Persist mIoU, mAP50, and mAP25 to results_dir."""
        res = self.reduce_and_compute_metrics()
        if not res:
            return
        keys = ["mIoU", "mAP50", "mAP25"]
        for k in keys:
            self.log(f"test/{k}", float(res[k]), sync_dist=False)
        if self.global_rank != 0:
            return
        results_dir = self.experiment_spec["results_dir"]
        if not results_dir:
            return
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "reasoning_point_cloud_metrics.json")
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump({key: float(res[key]) for key in keys}, handle, indent=2)
        logger.info("Wrote evaluation metrics to %s", results_path)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Inference step. Dumps one prediction per ``[SEG]`` binding.

        Each binding is written as a compressed ``.npz`` holding the binarized
        2D mask, its sample/view ids, and the fused segmented point cloud for
        that sample. When ``inference.save_full_point_cloud`` is enabled, one
        additional ``cloud_*.npz`` per sample stores dense points, aligned RGB,
        and the fused multi-view segmentation mask for visualization.
        """
        del dataloader_idx
        _, _, out = self.forward_batch(batch)

        inference_cfg = _cfg(self.cfg, "inference", None)
        output_dir = _cfg(inference_cfg, "output_dir", None)
        output_dir = str(output_dir) if output_dir else os.path.join(
            str(self.experiment_spec["results_dir"]), "predictions",
        )
        os.makedirs(output_dir, exist_ok=True)
        mask_threshold = float(_cfg(
            inference_cfg, "mask_threshold", self._eval_mask_thresh,
        ))

        prob = out.get("point_mask_prob")
        sample_idx = out["seg_sample_idx"].tolist()
        view_idx = out["seg_view_idx"].tolist()
        clouds = out.get("segmented_point_clouds") or []
        instruction = batch[0]["instruction"]
        for n, (b, v) in enumerate(zip(sample_idx, view_idx)):
            payload = {
                "sample_index": np.int64(b),
                "view_index": np.int64(v),
                "instruction": np.array(instruction[b], dtype=object),
            }
            if prob is not None:
                payload["mask"] = (prob[n] > mask_threshold).cpu().numpy()
                payload["mask_prob"] = prob[n].float().cpu().numpy()
            if b < len(clouds):
                payload["points"] = clouds[b].float().cpu().numpy()
            out_path = os.path.join(
                output_dir,
                f"pred_{self.global_rank:02d}_{batch_idx:06d}_{n:03d}.npz",
            )
            np.savez_compressed(out_path, **payload)

        if not bool(_cfg(inference_cfg, "save_full_point_cloud", False)):
            return

        full_points = out.get("metric_points", out.get("world_points"))
        dense_masks = out.get("segmented_point_mask")
        if full_points is None or dense_masks is None:
            logger.warning(
                "Full point-cloud output requested, but geometry or segmented "
                "masks are unavailable; no cloud_*.npz files were written."
            )
            return

        batch_size, num_views, height, width = full_points.shape[:4]
        images = torch.stack([view["img"] for view in batch], dim=1).float()
        if tuple(images.shape[-2:]) != (height, width):
            images = F.interpolate(
                images.flatten(0, 1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, num_views))
        full_colors = (
            images.permute(0, 1, 3, 4, 2).clamp(0, 1).mul(255).to(torch.uint8)
        )

        fused_masks = torch.zeros(
            (batch_size, num_views, height, width),
            dtype=torch.bool,
            device=full_points.device,
        )
        for binding_index, (sample, view) in enumerate(zip(sample_idx, view_idx)):
            fused_masks[sample, view] |= dense_masks[binding_index].to(
                device=full_points.device, dtype=torch.bool,
            )

        scenes = batch[0].get("scene") or [""] * batch_size
        for sample in range(batch_size):
            sample_bindings = [
                index for index, binding_sample in enumerate(sample_idx)
                if binding_sample == sample
            ]
            cloud_payload = {
                "sample_index": np.int64(sample),
                "scene": np.array(scenes[sample], dtype=object),
                "instruction": np.array(instruction[sample], dtype=object),
                "view_indices": np.asarray(
                    [view_idx[index] for index in sample_bindings], dtype=np.int64,
                ),
                "full_points": full_points[sample].float().cpu().numpy(),
                "full_colors": full_colors[sample].cpu().numpy(),
                "segmented_mask": fused_masks[sample].cpu().numpy(),
            }
            if sample < len(clouds):
                cloud_payload["segmented_points"] = clouds[sample].float().cpu().numpy()
            cloud_path = os.path.join(
                output_dir,
                f"cloud_{self.global_rank:02d}_{batch_idx:06d}_{sample:03d}.npz",
            )
            np.savez_compressed(cloud_path, **cloud_payload)

    def on_save_checkpoint(self, checkpoint):
        """Drop frozen SAM/Qwen base params; keep trainable LoRA/projector rows."""
        sd = checkpoint.get("state_dict")
        if not sd:
            return
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        for key in list(sd):
            if key not in trainable:
                del sd[key]

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configure standard periodic, best, and exception checkpoints.

        Only trainable LoRA/projector rows are persisted (see
        :meth:`on_save_checkpoint`). The shared TAO checkpointer ranks the best
        checkpoints after validation by canonical point-cloud mIoU by default.
        The periodic stream maintains a resumable ``*_latest.pth`` checkpoint.
        """
        results_dir = (
            self.cfg["results_dir"]
            if isinstance(self.cfg, dict)
            else self.cfg.results_dir
        )
        checkpoint_interval = int(_cfg(self.train_cfg, "checkpoint_interval", 1) or 1)
        checkpoint_interval_unit = str(_cfg(
            self.train_cfg, "checkpoint_interval_unit", "epoch",
        ))
        status_logger_callback = TAOStatusLogger(results_dir, append=True)
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"
        checkpoint_callback = ModelCheckpoint(
            dirpath=results_dir,
            monitor=None,
            save_top_k=-1,
            every_n_epochs=(
                checkpoint_interval if checkpoint_interval_unit == "epoch" else None
            ),
            every_n_train_steps=(
                checkpoint_interval if checkpoint_interval_unit == "step" else None
            ),
            save_on_train_epoch_end=True,
            save_last="link",
            filename="model_{epoch:03d}_{step:06d}",
            enable_version_counter=False,
        )
        TAOExceptionCheckpoint.FILE_EXTENSION = ModelCheckpoint.FILE_EXTENSION
        TAOExceptionCheckpoint.CHECKPOINT_NAME_LAST = ModelCheckpoint.CHECKPOINT_NAME_LAST
        exception_checkpoint_callback = TAOExceptionCheckpoint(dirpath=results_dir)
        callbacks = [status_logger_callback, checkpoint_callback, exception_checkpoint_callback]
        return self._configure_best_checkpoint(callbacks, results_dir)

    def configure_optimizers(self):
        """AdamW over all trainable reasoning parameters, warmup then cosine."""
        lr = float(_cfg(self.train_cfg, "lr", 1e-4))
        min_lr = float(_cfg(self.train_cfg, "min_lr", 1e-6))
        weight_decay = float(_cfg(self.train_cfg, "weight_decay", 0.05))

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("Reasoning model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 0
        if total_steps <= 1:
            return optimizer

        num_epochs = int(_cfg(self.train_cfg, "num_epochs", 10) or 1)
        warmup_epochs = float(_cfg(self.train_cfg, "warmup_epochs", 2.0) or 0)
        warmup_steps = int(total_steps * min(max(warmup_epochs / max(num_epochs, 1), 0.0), 0.5))
        min_ratio = max(0.0, min(min_lr / max(lr, 1e-12), 1.0))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
