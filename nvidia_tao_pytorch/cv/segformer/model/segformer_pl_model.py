# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""SegFormer_pl Model PyTorch Lightning Module"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.core.path_utils import expand_path
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.segformer.model.segformer import build_model
from nvidia_tao_pytorch.cv.segformer.dataloader.utils import build_target_class_list, build_palette
from nvidia_tao_pytorch.cv.segformer.utils.loss import cross_entropy, mmIoULoss
from nvidia_tao_pytorch.cv.segformer.utils import utils_vis
from nvidia_tao_pytorch.cv.segformer.utils.iou_metric import MeanIoUMeter
from nvidia_tao_pytorch.cv.segformer.model.segformer_utils import resize


class SegFormerPlModel(TAOLightningModule):
    """
    PTL Model for SegFormer
    """

    def __init__(self, experiment_spec, export=False):
        """pl_model initialization

        Args:
            experiment_spec (OmegaConf.DictConfig): Experiment configuration contains all the configurations. Default define in tao-core and user specify in yaml.
            export (bool, optional): No use in current SegFormer repo because the model will not change the forward/architecture. Defaults to False.
        """
        super().__init__(experiment_spec)
        # Overriding what's done in super()
        self.dataset_config = self.experiment_spec.dataset.segment
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference

        # init the model
        self._build_model(export)
        self._build_criterion()

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.n_class = self.dataset_config.num_classes
        self.running_metric = MeanIoUMeter(n_class=self.n_class)
        self.batch_size = self.dataset_config.batch_size

        # label transform indicates whether the label is normalized to [0, 1] or [0, 255]
        # if norm, then the label is normalized to [0, 1]
        # if not norm, then the label is in [0, 255]
        # used in visualize function
        self.label_transform = self.dataset_config.label_transform

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.batch = None
        self.vis_dir = self.experiment_spec.results_dir
        self.optimizer_G = None

        # define the loss functions
        self.weights = tuple(self.train_config.segment.weights)
        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches

        # This part is from mmengine, id_color_map is used for visualization when n_class > 2
        target_classes = build_target_class_list(self.dataset_config)
        PALETTE, CLASSES, label_map, id_color_map = build_palette(target_classes)
        self.palette = PALETTE
        self.classes = CLASSES
        self.label_map = label_map
        self.color_map = id_color_map

        self.checkpoint_filename = 'segformer_model'

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        assert self.train_config.segment["loss"] in ['ce', 'mmiou'], "SegFormer Segmentation pipeline currently only supports ['ce', 'mmiou'] loss functions."
        if self.train_config.segment["loss"] == 'ce':
            self._pxl_loss = cross_entropy
        elif self.train_config.segment["loss"] == 'mmiou':
            self._pxl_loss = mmIoULoss(n_classes=self.n_class).cuda()
        else:
            raise NotImplementedError(self.train_config.segment["loss"])

        self.criterion = self._pxl_loss

    def _get_parameter_groups(self, lr, weight_decay):
        """Get parameter groups for the optimizer."""
        backbone_decay = []
        backbone_no_decay = []
        decoder_decay = []
        decoder_no_decay = []
        skip_names = [
            # Common
            "bias",
            "norm",
            "gamma",
            "bn",
            "temperature",
            "token",
            # RADIOAdapter
            "level_embed",
            "spm",
            "ls",
            # SegFormer decoder
            "linear_fuse.1"  # nn.BatchNorm
        ]
        for name, param in self.model.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(s in name for s in skip_names):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        for name, param in self.model.decoder.named_parameters():
            if not param.requires_grad:
                continue
            if any(s in name for s in skip_names):
                decoder_no_decay.append(param)
            else:
                decoder_decay.append(param)
        return [
            {"params": backbone_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": decoder_decay, "lr": lr * 10.0, "weight_decay": weight_decay},
            {"params": decoder_no_decay, "lr": lr * 10.0, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        """Configure optimizers for training"""
        params = self._get_parameter_groups(self.lr, self.optimizer.weight_decay)
        # define optimizers
        if self.optimizer.optim == "sgd":
            self.optimizer_G = optim.SGD(
                params,
                lr=self.lr,
                momentum=self.optimizer.momentum,
                weight_decay=self.optimizer.weight_decay,
            )
        elif self.optimizer.optim == "adam":
            self.optimizer_G = optim.Adam(params, lr=self.lr, weight_decay=self.optimizer.weight_decay)
        elif self.optimizer.optim == "adamw":
            self.optimizer_G = optim.AdamW(
                params, lr=self.lr, betas=[0.9, 0.999], weight_decay=self.optimizer.weight_decay
            )
        else:
            raise NotImplementedError("Optimizer {} is not implemented".format(self.optimizer.optim))

        # define lr schedulers
        interval = "epoch"
        if self.lr_policy == 'linear':
            interval = "step"

            def lambda_rule(step):
                warmup_ratio = 1e-6
                warmup_steps = self.trainer.estimated_stepping_batches // 100
                max_steps = self.trainer.estimated_stepping_batches
                # k = (1 - epoch / (self.max_epochs + 1)) * (1 - warmup_ratio)
                # lr_l = (1 - k)
                # lr_l = 1.0 - epoch / float(self.max_epochs + 1)
                if step < warmup_steps:
                    lr_l = 1 - (1 - warmup_ratio) * (warmup_steps - step) / warmup_steps
                else:
                    lr_l = 1 - (step - warmup_steps) / float(max_steps - warmup_steps + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif self.lr_policy == 'step':
            step_size = self.max_epochs // 3
            # args.lr_decay_iters
            scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=step_size, gamma=0.1)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.lr_policy)

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = {
            "scheduler": self.lr_scheduler,
            "interval": interval,
            "frequency": 1
        }
        optim_dict['monitor'] = self.monitor_name
        return optim_dict

    def _update_metric(self):
        """
        update metric
        Calculates running metrics for train, validate and test.
        """
        target = self.batch['mask'].to(self.device).detach()
        pred = self.final_pred.detach()
        pr = resize(
            pred,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        pr = torch.argmax(pr, dim=1)

        self.running_metric.update_cm(pr=pr.cpu(), gt=target.cpu())  # current_score

    def _visualize_pred_multi(self, i):
        """Helper function to visualize predictions for multi class segmentation"""
        pred = torch.argmax(self.final_pred[i].unsqueeze(0), dim=1, keepdim=True)
        return pred

    def _visualize_predictions(self, batch_idx=-1, vis_afer_n_batches=100):
        """
        Visualizes two input images along with GT and prediction of
        segmentation change map as a linear grid of images.
        Used to only visualize predictions during evaluation
        input:
            flag:
                    after every 100 images, not used during validation
        """
        if np.mod(batch_idx, vis_afer_n_batches) == 0:
            for i in range(len(self.batch['img'])):
                vis_input = resize(
                    self.batch['img'][i].unsqueeze(0),
                    size=self.batch['mask'].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                vis_input = utils_vis.make_numpy_grid(
                    utils_vis.de_norm(
                        vis_input, mean=self.dataset_config["augmentation"]["mean"], std=self.dataset_config["augmentation"]["std"]
                    ),
                    already_img=True
                )

                vis_pred, _ = utils_vis.make_numpy_grid(
                    self._visualize_pred_multi(i),
                    gt=self.batch['mask'][i].unsqueeze(0),
                    color_map=self.color_map,
                    label_transform=self.label_transform
                )
                vis_gt = utils_vis.make_numpy_grid(
                    self.batch['mask'][i].unsqueeze(0),
                    color_map=self.color_map,
                    label_transform=self.label_transform
                )

                # Combining horizontally
                line_width = 10  # width of the black line in pixels
                line = np.zeros((vis_input.shape[0], line_width, 3), dtype=np.uint8)  # create a black line

                # Combine predictions in the order of Image A, Image B, Prediction, GT.
                vis = np.concatenate([vis_input, line, vis_pred, line, vis_gt], axis=1)

                vis = np.clip(vis, a_min=0.0, a_max=1.0)

                # Save combined visualisation in a different folder.
                vis_combined_dir = os.path.join(self.vis_dir, "combined_visualisation")
                if not os.path.exists(vis_combined_dir):
                    os.makedirs(vis_combined_dir)
                file_name = expand_path(os.path.join(
                    vis_combined_dir, self.batch['name'][i]))
                plt.imsave(file_name, vis)

                # Dump Predictions only
                vis_pred_only = np.clip(vis_pred, a_min=0.0, a_max=1.0)
                file_name = expand_path(os.path.join(
                    self.vis_dir, self.batch['name'][i]))
                plt.imsave(file_name, vis_pred_only)

    def _visualize_infer_output(self, batch_idx, vis_afer_n_batches=1):
        """
        Visualizes two input images along with segmentation change map prediction
        as a linear grid of images. Does not include GT segmentation change map
        during inference.
        """
        if np.mod(batch_idx, vis_afer_n_batches) == 0:
            for i in range(len(self.batch['img'])):
                vis_input = utils_vis.make_numpy_grid(
                    utils_vis.de_norm(
                        self.batch['img'][i].unsqueeze(0), mean=self.dataset_config["augmentation"]["mean"], std=self.dataset_config["augmentation"]["std"]
                    ),
                    already_img=True
                )

                vis_pred = utils_vis.make_numpy_grid(
                    self._visualize_pred_multi(i),
                    color_map=self.color_map,
                    label_transform=self.label_transform
                )

                # Combining horizontally
                line_width = 10  # width of the black line in pixels
                line = np.zeros((vis_input.shape[0], line_width, 3), dtype=np.uint8)  # create a black line
                vis = np.concatenate([vis_input, line, vis_pred], axis=1)

                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                # Save combined visualisation in a different folder.
                vis_combined_dir = os.path.join(self.vis_dir, "combined_visualisation")
                if not os.path.exists(vis_combined_dir):
                    os.makedirs(vis_combined_dir)
                file_name = expand_path(os.path.join(
                    vis_combined_dir, self.batch['name'][i]))
                plt.imsave(file_name, vis)

                # Dump Predictions only
                vis_pred_only = np.clip(vis_pred, a_min=0.0, a_max=1.0)
                file_name = expand_path(os.path.join(
                    self.vis_dir, self.batch['name'][i]))
                plt.imsave(file_name, vis_pred_only)

    def _clear_cache(self):
        self.running_metric.clear()

    def _collect_epoch_states(self):
        """Collect evaluation metrics for each epoch (train, val, test)"""
        scores, mean_score_dict = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        # message = 'Scores per class'
        self.log_dict(scores, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # message = 'Mean scores for all classes: '
        self.log_dict(mean_score_dict, on_step=False, on_epoch=True, sync_dist=True)
        return scores, mean_score_dict

    def _prepare_batch(self, batch):
        """Prepare batch for forward pass."""
        if isinstance(batch, dict):
            for key in batch.keys():
                if isinstance(batch[key], list) and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = torch.stack(batch[key])
                    batch[key] = torch.squeeze(batch[key], dim=0)
        return batch

    def _forward_pass(self, batch, split='train'):
        """Forward pass for training, validation and testing."""
        self.batch = self._prepare_batch(batch)
        self.final_pred = self.model(self.batch['img'])
        if split == 'predict':
            self.loss = None
        else:
            self.loss = self.criterion(self.final_pred, self.batch['mask'].long())

        return self.final_pred, self.loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        _, loss = self._forward_pass(batch, "train")
        self._update_metric()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log(
            "lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True
        )
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
        """Validation epoch start."""
        if not self.trainer.sanity_checking:
            # FLUSHING TRAINING EPOCH METRICS
            _, _ = self._collect_epoch_states()  # logs all evaluation metrics
        self._clear_cache()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        _, loss = self._forward_pass(batch, "val")
        self._update_metric()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end.
        compute mAP at the end of epoch
        """
        # FLUSHING VALIDATION EPOCH METRICS
        if not self.trainer.sanity_checking:
            scores, mean_scores = self._collect_epoch_states()  # logs all evaluation metrics
            self.log("val_acc", scores['acc'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_miou", scores['miou'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._clear_cache()

        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["val_acc"] = scores['acc']
            self.status_logging_dict["val_miou"] = scores['miou']
            self.status_logging_dict["val_mf1"] = scores['mf1']
            self.status_logging_dict["val_mprecision"] = mean_scores['mprecision']
            self.status_logging_dict["val_mrecall"] = mean_scores['mrecall']
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self) -> None:
        """ Test epoch start."""
        self._clear_cache()

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate """
        _, loss = self._forward_pass(batch, "test")

        # Calculate running metrics
        self._update_metric()
        self.final_pred = resize(
            self.final_pred,
            size=batch["mask"].shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        self._visualize_predictions(batch_idx, vis_afer_n_batches=self.vis_after_n_batches)

        self.log("test_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    def on_test_epoch_end(self):
        """Test epoch end."""
        scores, mean_scores = self._collect_epoch_states()  # needed for update metrics

        self.status_logging_dict = {}
        self.status_logging_dict["test_acc"] = scores['acc']
        self.status_logging_dict["test_miou"] = scores['miou']
        self.status_logging_dict["test_mf1"] = scores['mf1']
        self.status_logging_dict["test_mprecision"] = mean_scores['mprecision']
        self.status_logging_dict["test_mrecall"] = mean_scores['mrecall']
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference """
        outputs, _ = self._forward_pass(batch, "predict")
        self.final_pred = resize(
            self.final_pred,
            size=batch["img"].shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        self._visualize_infer_output(batch_idx, vis_afer_n_batches=self.vis_after_n_batches_infer)

        return outputs
