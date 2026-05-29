# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main PTL model for Sparse4D."""

import torch
import pytorch_lightning as pl

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d import build_model
from nvidia_tao_pytorch.cv.sparse4d.model.criterion import SetCriterion


class Sparse4DPlModel(TAOLightningModule):
    """PyTorch Lightning Module for Sparse4D 3D Object Detection and Tracking."""

    def __init__(self, experiment_spec, export=False):
        """Initialize the Sparse4D PyTorch Lightning Module.

        Args:
            experiment_spec: The experiment specification.
            export: Whether to export the model.
        """
        super().__init__(experiment_spec)
        self.model_config = self.experiment_spec.model
        self.train_config = self.experiment_spec.train
        self.dataset_config = self.experiment_spec.dataset

        # Build the model
        self._build_model(export)
        self.criterion = SetCriterion(self.model_config, self.model.head.instance_bank)

        # Set up monitoring and logging
        self.status_logging_dict = {}

        # Initialize metrics for tracking
        self.val_stats = {}

        # Set the checkpoint filename
        self.checkpoint_filename = 'sparse4d_model'

        # get num_iters_per_epoch from experiment_spec
        batch_size = self.dataset_config.batch_size
        num_gpus = self.train_config.num_gpus
        num_bev_groups = self.dataset_config.num_bev_groups
        num_nodes = self.train_config.num_nodes
        self.num_epochs = self.train_config.num_epochs
        self.num_iters_per_epoch = int(1000 * num_bev_groups // (num_nodes * num_gpus * batch_size))

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: The input batch.
            batch_idx: The batch index.

        Returns:
            The loss dictionary.
        """
        # Forward pass
        batch['img'] = batch['img'].float()
        outputs = self.model(batch['img'], batch)

        loss_dict = self.criterion(outputs, batch)

        # log the lr
        pg0_lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.log('lr', pg0_lr, on_step=True, prog_bar=True)   # one call, one value

        # Log losses
        for loss_name, loss_value in loss_dict.items():
            # Ensure loss_value is a scalar before logging
            if isinstance(loss_value, torch.Tensor) and loss_value.numel() > 1:
                loss_value = loss_value.mean()
            self.log(f'{loss_name}', loss_value, on_step=True, prog_bar=True)

        # Calculate total loss
        total_loss = sum([v.mean() if isinstance(v, torch.Tensor) and v.numel() > 1 else v
                          for k, v in loss_dict.items() if 'loss' in k])
        self.log('loss', total_loss, on_step=True, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        self.status_logging_dict = {}
        for k, v in self.trainer.logged_metrics.items():
            self.status_logging_dict[k] = v.item()

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self):
        """On validation epoch start."""
        self.val_dataset = self.trainer.datamodule.val_dataset

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: The input batch.
            batch_idx: The batch index.

        Returns:
            The validation output.
        """
        # Forward pass in evaluation mode
        batch['img'] = batch['img'].float()
        outputs = self.model(batch['img'], batch)
        self.val_dataset.update_results(outputs)

        return outputs

    def on_validation_epoch_end(self):
        """
        Validation epoch end.
        Compute metrics at the end of epoch based on collected outputs.
        """
        if self.trainer.is_global_zero:

            results = self.val_dataset.results
            scores = self.val_dataset.evaluate(results)

            if not self.trainer.sanity_checking:
                for k, v in scores.items():
                    self.log(k, v, on_step=False, on_epoch=True, prog_bar=False)

                self.status_logging_dict = {}
                for k, v in scores.items():
                    self.status_logging_dict[k] = v
                status_logging.get_status_logger().kpi = self.status_logging_dict
                status_logging.get_status_logger().write(
                    message="Eval metrics generated.",
                    status_level=status_logging.Status.RUNNING
                )

        self.val_dataset.clear_results()
        pl.utilities.memory.garbage_collection_cuda()

    def forward(self, batch):
        """Forward pass for inference only."""
        img = batch['img']

        # Use simple_test if available, otherwise the standard model call
        if hasattr(self.model, 'simple_test'):
            outputs = self.model.simple_test(img, batch)
        else:
            outputs = self.model(img, batch)

        return outputs

    def on_predict_epoch_start(self):
        """Predict epoch start."""
        self.test_dataset = self.trainer.datamodule.test_dataset

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference."""
        outputs = self.model(batch['img'], batch)
        self.test_dataset.update_results(outputs)
        return outputs

    def on_predict_epoch_end(self):
        """
        Predict epoch end.
        Save the result inferences at the end of epoch.
        """
        results = self.test_dataset.results
        jsonfile_prefix = self.experiment_spec.inference.jsonfile_prefix
        output_nvschema = self.experiment_spec.inference.output_nvschema
        show = self.experiment_spec.visualize.show
        out_dir = self.experiment_spec.visualize.vis_dir
        pipeline = self.trainer.datamodule.vis_transforms
        vis_score_threshold = self.experiment_spec.visualize.vis_score_threshold
        n_images_col = self.experiment_spec.visualize.n_images_col
        viz_down_sample = self.experiment_spec.visualize.viz_down_sample

        tracking = self.experiment_spec.inference.tracking
        self.test_dataset.format_results(
            results, jsonfile_prefix=jsonfile_prefix, tracking=tracking,
            output_nvschema=output_nvschema,
            show=show, out_dir=out_dir, pipeline=pipeline,
            vis_score_threshold=vis_score_threshold,
            n_images_col=n_images_col, viz_down_sample=viz_down_sample
        )

    def on_test_epoch_start(self):
        """Test epoch start."""
        self.test_dataset = self.trainer.datamodule.test_dataset

    def test_step(self, batch, batch_idx):
        """Test step."""
        # Get model predictions in test mode
        batch['img'] = batch['img'].float()
        outputs = self.model.simple_test(batch['img'], batch)
        self.test_dataset.update_results(outputs)
        return outputs

    def on_test_epoch_end(self):
        """Test epoch end."""
        results = self.test_dataset.results
        out_dir = self.experiment_spec.visualize.vis_dir
        show = self.experiment_spec.visualize.show
        pipeline = self.trainer.datamodule.vis_transforms
        vis_score_threshold = self.experiment_spec.visualize.vis_score_threshold
        n_images_col = self.experiment_spec.visualize.n_images_col
        viz_down_sample = self.experiment_spec.visualize.viz_down_sample
        output_nvschema = self.experiment_spec.inference.output_nvschema
        jsonfile_prefix = self.experiment_spec.inference.jsonfile_prefix
        metrics = self.experiment_spec.evaluate.metrics

        scores = self.test_dataset.evaluate(
            results,
            metrics=metrics,
            jsonfile_prefix=jsonfile_prefix,
            output_nvschema=output_nvschema,
            show=show,
            out_dir=out_dir,
            pipeline=pipeline,
            vis_score_threshold=vis_score_threshold,
            n_images_col=n_images_col,
            viz_down_sample=viz_down_sample
        )
        self.test_dataset.clear_results()
        self.status_logging_dict = {}
        for k, v in scores.items():
            self.status_logging_dict[k] = v
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

        return scores

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            The optimizer and scheduler configuration.
        """
        optim_config = self.train_config.optim
        bb_lr_mult = optim_config.paramwise_cfg.custom_keys.img_backbone.lr_mult
        backbone, others = [], []
        for name, p in self.model.named_parameters():
            (backbone if "img_backbone" in name else others).append(p)

        param_groups = [
            {"params": backbone, "lr": optim_config.lr * bb_lr_mult},
            {"params": others},                          # base LR
        ]

        optimizer = torch.optim.AdamW(
            param_groups, lr=optim_config.lr, weight_decay=optim_config.weight_decay
        )

        warmup_iters = optim_config.lr_scheduler.warmup_iters
        total_steps = self.num_epochs * self.num_iters_per_epoch
        min_lr = optim_config.lr * optim_config.lr_scheduler.min_lr_ratio

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=optim_config.lr_scheduler.warmup_ratio,
            total_iters=warmup_iters,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_iters,
            eta_min=min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_iters],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # step every iteration
                "frequency": 1,
            },
        }
