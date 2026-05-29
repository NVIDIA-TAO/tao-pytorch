# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for MAE. """
import os
import pandas as pd
import functools

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.classification import Accuracy

from timm.data.mixup import Mixup
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from transformers.optimization import get_cosine_schedule_with_warmup

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.ssl.mae.model.mae import mae_vit_group
from nvidia_tao_pytorch.ssl.mae.model.fcmae import fcmae_group
from nvidia_tao_pytorch.ssl.mae.model.hiera_mae import mae_hiera_group
from nvidia_tao_pytorch.ssl.mae.utils import lr_decay as lrd
from nvidia_tao_pytorch.ssl.mae.utils.pos_embed import interpolate_pos_embed


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


class MAEPlModule(TAOLightningModule):
    """MAE LightningModule."""

    def __init__(self, cfg, export=False) -> None:
        """Initialize MAE model.
        Args:
            cfg (OmegaConfig): Hydra config
            export (bool): Whether to export the model.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.export = export
        self._build_model()
        self.mixup_fn = None
        if self.cfg.train.stage != 'pretrain':
            self._build_mixup()
            self._build_criterion()
        self.status_logging_dict = {}

    @property
    def model_mapper(self):
        """Model map."""
        models = mae_vit_group + fcmae_group + mae_hiera_group
        return {i.__name__: i for i in models}

    def load_pretrained_weights(self):
        """Load pretrained weights."""
        if self.cfg.train.pretrained_model_path:
            # TODO(@yuw): load pretrained weights
            checkpoint = torch.load(self.cfg.train.pretrained_model_path, map_location='cpu')
            updated_state_dict = {}
            for key, value in list(checkpoint['state_dict'].items()):
                # for vit
                if key.startswith("model."):
                    key = key[len("model."):]
                # for convnextv2
                if key.startswith("encoder."):
                    key = key[len("encoder."):]
                updated_state_dict[key] = value

            state_dict = self.model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in updated_state_dict and updated_state_dict[k].shape != state_dict[k].shape:
                    logging.info(f"Removing key {k} from pretrained checkpoint")
                    del updated_state_dict[k]

            # interpolate position embedding
            interpolate_pos_embed(self.model, updated_state_dict)
            # load pretrained model
            msg = self.model.load_state_dict(updated_state_dict, strict=False)
            logging.info(msg)

    def _build_model(self):
        """Internal function to build the model."""
        model_arch = self.cfg.model.arch
        supported_archs = [m[len("mae_"):] for m in self.model_mapper.keys()]
        if model_arch not in supported_archs:
            raise NotImplementedError(f"Only {", ".join(supported_archs)} are supported, but {model_arch} is specified.")
        self.checkpoint_filename = model_arch
        # Enable the MAE mask for the pretrain stage and if not exporting the model.
        if self.cfg.train.stage == "pretrain" and not self.export:
            model_arch = "mae_" + model_arch
            self.model = self.model_mapper[model_arch](
                norm_pix_loss=self.cfg.train.norm_pix_loss,
                mask_ratio=self.cfg.train.mask_ratio,
            )
        else:
            # WAR to export the backbone of the model from the pretrain stage.
            if self.cfg.train.stage == "pretrain":
                # Adding the head to the model for finetune stage.
                self.model = BACKBONE_REGISTRY.get(model_arch)(
                    num_classes=self.cfg.model.num_classes,
                    freeze_at='all',
                    freeze_norm=True
                )
            elif self.cfg.train.stage == "finetune":
                logging.warning("[ATTENTION!!!] Finetune stage in MAE will be deprecated in the next release. "
                                "Please use `classification` pipeline to finetune your pretrained model.")
                logging.warning("If you wish to use the finetuned model in any TAO downstream task, "
                                "you must use `classification_pyt` endpoint to finetune your pre-trained model.")
                if model_arch.startswith("vit"):
                    model_arch = model_arch + "_mae"
                self.model = BACKBONE_REGISTRY.get(model_arch)(
                    num_classes=self.cfg.model.num_classes,
                )
            else:
                raise NotImplementedError(
                    f"Stage `{self.cfg.train.stage}` is not supported.")

            if self.cfg.train.pretrained_model_path:
                logging.info(f"Loading pretrained model from {self.cfg.train.pretrained_model_path}")
                # TODO(@yuw): load pretrained weights
                checkpoint = torch.load(self.cfg.train.pretrained_model_path, map_location='cpu')
                model_state = checkpoint.get('model', None) or checkpoint.get('model_state', None) or checkpoint.get('state_dict', None)
                updated_state_dict = {}
                for key, value in list(model_state.items()):
                    if 'vit' in model_arch or 'hiera' in model_arch:
                        # for vit
                        if key.startswith("model."):
                            key = key[len("model."):]
                    if 'convnext' in model_arch:
                        # for convnextv2
                        if key.startswith("encoder."):
                            key = key[len("encoder."):]
                        if key.startswith("model.encoder."):
                            key = key[len("model.encoder."):]
                        # for convnextv2
                        if 'decoder' in key or 'mask_token' in key or 'proj' in key or 'pred' in key:
                            logging.info(f"Skipping key {key} from pretrained checkpoint")
                            continue
                    updated_state_dict[key] = value

                if 'convnext' in model_arch:
                    for key, value in list(updated_state_dict.items()):
                        if key.endswith('bias') and len(value.shape) != 1:
                            updated_state_dict[key] = value.reshape(-1)
                        elif 'grn' in key:
                            # Reshape GRN parameters from 6D to 4D if needed
                            if value.dim() == 6:  # If parameter is 6D [1, 1, 1, 1, 1, C]
                                updated_state_dict[key] = value.squeeze(3).squeeze(3)  # Reshape to 4D [1, 1, 1, C]
                            elif value.dim() == 2:
                                updated_state_dict[key] = value.unsqueeze(0).unsqueeze(1)
                state_dict = self.model.state_dict()

                for k in ['head.weight', 'head.bias']:
                    if k in updated_state_dict and updated_state_dict[k].shape != state_dict[k].shape:
                        logging.info(f"Removing key {k} from pretrained checkpoint")
                        del updated_state_dict[k]

                # interpolate position embedding
                if 'hiera' in model_arch:
                    pass  # TODO(@yuw)
                else:
                    interpolate_pos_embed(self.model, updated_state_dict)
                # load pretrained model
                msg = self.model.load_state_dict(updated_state_dict, strict=False)
                logging.info(msg)
            # manually initialize fc layer
            if 'hiera' in model_arch:
                trunc_normal_(self.model.head.fc.weight, std=2e-5)
            else:
                trunc_normal_(self.model.head.weight, std=2e-5)
        # freeze modules
        if self.cfg.train.freeze:
            freezed_modules = []
            skipped_modules = []
            for module in self.cfg.train.freeze:
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

    def _build_mixup(self):
        mixup_active = self.cfg.dataset.augmentation.mixup > 0 or self.cfg.dataset.augmentation.cutmix > 0. or self.cfg.dataset.augmentation.cutmix_minmax is not None
        if mixup_active:
            logging.info("Mixup is activated!")
            self.mixup_fn = Mixup(
                mixup_alpha=self.cfg.dataset.augmentation.mixup,
                cutmix_alpha=self.cfg.dataset.augmentation.cutmix,
                cutmix_minmax=self.cfg.dataset.augmentation.cutmix_minmax,
                prob=self.cfg.dataset.augmentation.mixup_prob,
                switch_prob=self.cfg.dataset.augmentation.mixup_switch_prob,
                mode=self.cfg.dataset.augmentation.mixup_mode,
                label_smoothing=self.cfg.dataset.augmentation.smoothing,
                num_classes=self.cfg.model.num_classes)

    def _build_criterion(self):
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif self.cfg.dataset.augmentation.smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.cfg.dataset.augmentation.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Configure optimizers for training."""
        if self.cfg.train.stage == "pretrain":
            params = lrd.add_weight_decay(self.model, self.cfg.train.optim.weight_decay)
        else:
            if "vit" in self.cfg.model.arch:
                params = lrd.param_groups_vit(
                    self.model,
                    self.cfg.train.optim.weight_decay,
                    no_weight_decay_list=self.model.no_weight_decay(),
                    layer_decay=self.cfg.train.optim.layer_decay)
            else:
                params = lrd.param_groups_convnextv2(
                    self.model, self.cfg.train.optim.weight_decay,
                    layer_decay=self.cfg.train.optim.layer_decay,
                    layer_decay_type='group')
        optim_type = self.cfg.train.optim.type.lower()
        if optim_type == "sgd":
            optimizer = torch.optim.SGD(
                params, self.cfg.train.optim.lr, momentum=self.cfg.train.optim.momentum)
        elif optim_type == "adamw":
            optimizer = torch.optim.AdamW(
                params, self.cfg.train.optim.lr, betas=(0.9, 0.95))
        else:
            raise NotImplementedError(
                f"Optimizer type ({self.cfg.train.optim.type}) not supported.")

        lr_scheduler_type = self.cfg.train.optim.lr_scheduler.lower()
        if lr_scheduler_type == "multistep":  # Epoch based
            interval = 'epoch'
            lr_scheduler = MultiStepLR(optimizer, self.cfg.train.optim.milestones,
                                       gamma=self.cfg.train.optim.gamma)
        elif lr_scheduler_type == "cosine":
            interval = 'step'
            epoch_steps = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=epoch_steps * self.cfg.train.optim.warmup_epochs,
            )
        else:
            raise NotImplementedError(f"{lr_scheduler_type} is not supported.")
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                "interval": interval,
                "frequency": 1},
            'monitor': self.cfg.train.optim.monitor_name}

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.cfg.train.stage == 'pretrain':
            batch_size = batch['images'].shape[0]
            loss, _, _ = self.model(batch['images'])
        else:
            images, targets = batch
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)
            batch_size = images.shape[0]
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

        loss_total = loss.item()
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

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
        if self.cfg.train.stage == 'finetune':
            self.validation_outputs = []
            self.accuracy = Accuracy(task="multiclass", top_k=1, num_classes=self.cfg.model.num_classes)
            self.val_criterion = torch.nn.CrossEntropyLoss()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if self.cfg.train.stage == 'finetune':
            images, targets = batch

            batch_size = images.shape[0]
            outputs = self.model(images)

            loss = self.val_criterion(outputs, targets)
            val_loss = loss.item()

            acc = self.accuracy(outputs.cpu(), targets.cpu())
            tp = acc * batch_size

            val_metrics = {
                'val_loss': val_loss,
                'tp': tp,
                'batch_size': batch_size}
            self.validation_outputs.append(val_metrics)
            return val_metrics
        return None

    def val_epoch_end(self):
        """Common logic between validation/test epoch end"""
        if self.cfg.train.stage == 'finetune':
            average_val_loss = 0.0
            total_tp, total_n = 0, 0

            for out in self.validation_outputs:
                average_val_loss += out['val_loss']
                total_tp += out['tp']
                total_n += out['batch_size']

            average_val_loss /= len(self.validation_outputs)
            overall_acc = total_tp / total_n

            self.log("val_loss", average_val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            self.log("acc", overall_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["ACC_all"] = float(overall_acc)

            self.validation_outputs.clear()

    def on_validation_epoch_end(self):
        """
        Validation epoch end.
        Compute top K accuracy at the end of epoch.
        """
        self.val_epoch_end()

        if not self.trainer.sanity_checking and self.cfg.train.stage == 'finetune':
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics skipped.",
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
        self.val_epoch_end()
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_predict_start(self) -> None:
        """Called at the beginning of predicting."""
        self.prediction_outputs = []

    def predict_step(self, batch, batch_idx):
        """Predict step."""
        if self.cfg.train.stage == 'finetune':
            images = batch['images']
            outputs = self.model(images)
            preds = torch.argmax(outputs, axis=-1).cpu()
            return preds
        return None

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Predict batch end.
        Save the result inferences at the end of batch.
        """
        paths = batch['paths']
        predictions = outputs.cpu().numpy()
        self.prediction_outputs.extend(list(zip(paths, predictions)))

    def on_predict_end(self):
        """
        Predict end.
        Save the result inferences.
        """
        result_csv_path = os.path.join(self.cfg.results_dir, "results.csv")
        with open(result_csv_path, 'w', encoding='utf-8') as csv_f:
            # Write predictions to file
            df = pd.DataFrame(self.prediction_outputs)
            df.to_csv(csv_f, header=False, index=False)

    def forward(self, x):
        """Forward."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint with model identifier."""
        checkpoint["tao_model"] = "mae"
