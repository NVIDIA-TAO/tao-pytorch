# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Model PyTorch Lightning Module."""

import math

import torch

_MAX_LOGIT_SCALE = math.log(100)

from open_clip.loss import ClipLoss, SigLipLoss  # noqa: E402

from nvidia_tao_pytorch.core.tlt_logging import logging  # noqa: E402
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (  # noqa: E402
    TAOLightningModule,
)
from nvidia_tao_pytorch.core.loggers import (  # noqa: E402
    api_logging as status_logging,
)
from nvidia_tao_pytorch.multimodal.clip.model.clip import build_model  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (  # noqa: E402
    build_optimizer,
    compute_lr,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval import (  # noqa: E402
    RetrievalEvaluator,
    log_retrieval_metrics,
)


class CLIPPlModel(TAOLightningModule):
    """PTL module for CLIP Model with retrieval-based validation."""

    def __init__(self, experiment_spec, export=False):
        """Initialize CLIP model for training."""
        super().__init__(experiment_spec)
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = 'clip'

        clip_model = build_model(
            experiment_config=self.experiment_spec, export=export
        )
        self.model = clip_model.model
        self.tokenizer = clip_model.tokenizer
        self.preprocess_train, self.preprocess_val = (
            clip_model.preprocess_train,
            clip_model.preprocess_val,
        )

        if getattr(self.experiment_spec.train, "grad_checkpointing", False):
            self.model.set_grad_checkpointing()
            logging.info("Gradient checkpointing enabled")

        self.loss_type = self.experiment_spec.train.loss_type

        # Check if retrieval validation is configured
        val_cfg = getattr(self.experiment_spec.dataset, 'val', None)
        self.retrieval_enabled = (
            val_cfg is not None and
            getattr(val_cfg, 'datasets', None) and
            len(val_cfg.datasets) > 0
        )

    def setup(self, stage=None):
        """Set up training after Trainer is initialized."""
        if stage == 'fit':
            self.max_steps = self.trainer.estimated_stepping_batches
            self._build_criterion()

    def _build_criterion(self):
        """Build the loss function."""
        if self.loss_type == 'siglip':
            self.loss = SigLipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        elif self.loss_type == 'clip':
            self.loss = ClipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        else:
            raise NotImplementedError(
                f"loss function {self.loss_type} is not implemented"
            )
        self.criterion = self.loss

    def configure_optimizers(self):
        """Configure optimizer with per-tower parameter groups."""
        self.optimizer = build_optimizer(
            self.model, self.experiment_spec.train
        )
        self._build_tower_schedule_config()
        return self.optimizer

    def _build_tower_schedule_config(self):
        """Pre-compute per-tower schedule configs for training_step."""
        cfg = self.experiment_spec.train.optim

        self._tower_schedules = {
            'vision': {
                'lr': cfg.vision_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'text': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'logit': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
        }

    def on_train_start(self):
        """Training epoch start."""
        self.trainer.datamodule.resume_step = self.trainer.global_step

    def _forward_pass(self, batch):
        """Run forward pass."""
        image, text = batch[0], batch[1]
        return self.model(image=image, text=text)

    def _backward(self, outputs):
        """Compute loss from model outputs."""
        if len(outputs) == 3:
            image_features, text_features, logit_scale = outputs
            clip_loss = self.loss(image_features, text_features, logit_scale)
        else:
            image_features, text_features, logit_scale, logit_bias = outputs
            clip_loss = self.loss(
                image_features, text_features, logit_scale, logit_bias
            )
        return clip_loss, logit_scale

    def training_step(self, batch):
        """Training step."""
        image = batch[0]
        batch_size = (
            image['pixel_values'].shape[0]
            if isinstance(image, dict)
            else image.shape[0]
        )
        outputs = self._forward_pass(batch)
        loss, logit_scale = self._backward(outputs)

        # Update per-tower learning rates
        for param_group in self.optimizer.param_groups:
            tower = param_group.get('_tower', 'text')
            sched = self._tower_schedules.get(
                tower, self._tower_schedules['text']
            )
            param_group['lr'] = compute_lr(
                self.global_step,
                sched['lr'],
                sched['warmup'],
                self.max_steps,
                sched['scheduler'],
            )

        vision_lr = self._tower_schedules['vision']['lr']
        text_lr = self._tower_schedules['text']['lr']
        current_vision_lr = compute_lr(
            self.global_step,
            vision_lr,
            self._tower_schedules['vision']['warmup'],
            self.max_steps,
            self._tower_schedules['vision']['scheduler'],
        )
        current_text_lr = compute_lr(
            self.global_step,
            text_lr,
            self._tower_schedules['text']['warmup'],
            self.max_steps,
            self._tower_schedules['text']['scheduler'],
        )
        self.log(
            "train/vision_lr", current_vision_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/text_lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=True
        )
        loss_value = (
            loss['contrastive_loss'] if isinstance(loss, dict) else loss
        )
        self.log(
            "train_loss", loss_value,
            on_step=True, on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=batch_size,
        )
        self.log(
            "train/logit_scale", logit_scale.item(),
            on_step=True, on_epoch=False, prog_bar=False
        )

        with torch.no_grad():
            self.model.logit_scale.clamp_(0, _MAX_LOGIT_SCALE)
        return loss_value

    def on_train_epoch_end(self):
        """Log training metrics to status.json."""
        average_train_loss = (
            self.trainer.logged_metrics["train_loss_epoch"].item()
        )

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """Set up retrieval evaluator for validation."""
        if self.retrieval_enabled:
            self.retrieval_evaluator = RetrievalEvaluator(
                k_values=(1, 5, 10),
                device=self.device
            )
            self.image_embeddings = []
            self.text_embeddings = []
            logging.info("Retrieval evaluator initialized for validation.")
        else:
            self.retrieval_evaluator = None
            self.image_embeddings = []
            self.text_embeddings = []
            logging.warning(
                "No validation configured. Add datasets to val.datasets "
                "to enable retrieval evaluation."
            )

    def validation_step(self, batch, batch_idx):
        """Run validation: collect image/text embeddings for retrieval."""
        if self.retrieval_evaluator is None:
            return

        image = batch[0]
        text = batch[1]

        # Get image features
        output = self.model(image=image)
        image_features = (
            output["image_features"]
            if isinstance(output, dict)
            else output[0]
        )

        # Handle different text formats from dataloader
        if isinstance(text, list) and len(text) > 0:
            if isinstance(text[0], dict):
                text = {
                    k: torch.stack([t[k] for t in text])
                    for k in text[0].keys()
                }
            elif isinstance(text[0], torch.Tensor):
                text = torch.stack(text)
            elif isinstance(text[0], str):
                text = self.tokenizer(text)
                if isinstance(text, list):
                    text = text[0]

        # Get text features
        text_output = self.model(text=text)
        text_features = (
            text_output["text_features"]
            if isinstance(text_output, dict)
            else text_output[1]
        )

        self.image_embeddings.append(image_features.cpu())
        self.text_embeddings.append(text_features.cpu())

    def on_validation_epoch_end(self):
        """Compute and log retrieval metrics."""
        self.status_logging_dict = {}

        if self.retrieval_evaluator is not None and self.image_embeddings:
            image_emb = torch.cat(self.image_embeddings, dim=0).numpy()
            text_emb = torch.cat(self.text_embeddings, dim=0).numpy()

            retrieval_metrics = self.retrieval_evaluator.evaluate_bidirectional(
                image_emb, text_emb
            )
            log_retrieval_metrics(retrieval_metrics, prefix="val")

            # Log to PL and status
            for direction in ['image_to_text', 'text_to_image']:
                metrics = retrieval_metrics[direction]
                dir_prefix = 'i2t' if direction == 'image_to_text' else 't2i'
                self.log(f"val/{dir_prefix}_mAP", metrics.map_score, sync_dist=True)
                self.log(f"val/{dir_prefix}_R@1", metrics.recall_at_k[1], sync_dist=True)
                self.log(f"val/{dir_prefix}_R@5", metrics.recall_at_k[5], sync_dist=True)
                self.log(f"val/{dir_prefix}_MedR", metrics.median_rank, sync_dist=True)
                self.log(f"val/{dir_prefix}_MeanR", metrics.mean_rank, sync_dist=True)
                self.log(f"val/{dir_prefix}_AUC", metrics.auc, sync_dist=True)
                self.status_logging_dict[f"val/{dir_prefix}_mAP"] = str(metrics.map_score)
                self.status_logging_dict[f"val/{dir_prefix}_R@1"] = str(
                    metrics.recall_at_k[1]
                )
                self.status_logging_dict[f"val/{dir_prefix}_MedR"] = str(metrics.median_rank)
                self.status_logging_dict[f"val/{dir_prefix}_AUC"] = str(metrics.auc)

        if not self.trainer.sanity_checking and self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    # Test methods (reuse validation logic)
    def on_test_epoch_start(self) -> None:
        """Test epoch start - reuse validation setup."""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Test step - reuse validation step."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Test epoch end - compute and log retrieval metrics."""
        self.status_logging_dict = {}

        if self.retrieval_evaluator is not None and self.image_embeddings:
            image_emb = torch.cat(self.image_embeddings, dim=0).numpy()
            text_emb = torch.cat(self.text_embeddings, dim=0).numpy()

            retrieval_metrics = self.retrieval_evaluator.evaluate_bidirectional(
                image_emb, text_emb
            )
            log_retrieval_metrics(retrieval_metrics, prefix="test")

            # Log to PL and status
            for direction in ['image_to_text', 'text_to_image']:
                metrics = retrieval_metrics[direction]
                dir_prefix = 'i2t' if direction == 'image_to_text' else 't2i'
                self.log(f"test/{dir_prefix}_mAP", metrics.map_score, sync_dist=True)
                self.log(f"test/{dir_prefix}_R@1", metrics.recall_at_k[1], sync_dist=True)
                self.log(f"test/{dir_prefix}_R@5", metrics.recall_at_k[5], sync_dist=True)
                self.log(f"test/{dir_prefix}_MedR", metrics.median_rank, sync_dist=True)
                self.log(f"test/{dir_prefix}_MeanR", metrics.mean_rank, sync_dist=True)
                self.log(f"test/{dir_prefix}_AUC", metrics.auc, sync_dist=True)
                self.status_logging_dict[f"test/{dir_prefix}_mAP"] = str(metrics.map_score)
                self.status_logging_dict[f"test/{dir_prefix}_R@1"] = str(
                    metrics.recall_at_k[1]
                )
                self.status_logging_dict[f"test/{dir_prefix}_MedR"] = str(metrics.median_rank)
                self.status_logging_dict[f"test/{dir_prefix}_AUC"] = str(metrics.auc)

        if self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
