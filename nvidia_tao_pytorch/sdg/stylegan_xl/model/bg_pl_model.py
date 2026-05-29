# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigDatasetGAN Model PyTorch Lightning Module"""

import os
import torch
import torch.nn.functional as F
import torchvision
import PIL
import numpy as np
import torchmetrics
from pytorch_lightning.callbacks import Callback

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bigdatasetgan import build_model
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import colorize, gen_utils


# The callback means non-essential logic and can be disabled without affect the training.
# See the details from configure_callbacks of BigdatasetganPlModel
class VisualizationCallback(Callback):
    """ Visualization Callback for BigDatasetGAN Model."""

    def __init__(self, n_class, every_n_steps=10):
        """Init Callback."""
        # tool for coloring segmentation masks
        self.voc_viz = colorize.VOCColorize(n=n_class)
        # interval for saving generated masks for visualization
        self.every_n_steps = every_n_steps
        # The folder for saving generated masks. The path will be given at setup by loading the pl module configuration
        self.run_dir = None

    def setup(self, trainer, pl_module, stage):
        """Pytorch Lightning built-in function for setup. Will be called BEFORE the setup of pl module."""
        self.run_dir = os.path.join(pl_module.experiment_spec['results_dir'], "outputs")  # TODO double 'train'or not?
        if stage in ('fit', None):
            # Create two folders for 1. generated masks from seeds of training data and 2. generated masks from random sampling seeds
            if pl_module.trainer.global_rank == 0:
                os.makedirs(self.run_dir, exist_ok=True)
                os.makedirs(os.path.join(self.run_dir, 'viz/train'), exist_ok=True)
                os.makedirs(os.path.join(self.run_dir, 'viz/sample'), exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Pytorch Lightning built-in function for the end of an train batch. Will be called BEFORE the on_train_batch_end of pl module."""
        if batch_idx % self.every_n_steps == 0:
            # Visualize generated masks from seeds of training data
            seg_logits = outputs["seg_logits"]
            img_list = outputs["img_list"]
            labels = outputs["labels"]

            labels_pred = seg_logits.argmax(dim=1)
            labels_pred_rgb = self.get_segmentation_viz(labels_pred)
            labels_gt_rgb = self.get_segmentation_viz(labels)
            # scale up sample predicted image if the shape does not match the predicted mask
            viz_tensors = torch.cat([img_list[0].cpu().unsqueeze(0) if img_list[0].shape[1:] == labels_pred_rgb[0].shape[1:] else F.interpolate(img_list, size=labels_pred_rgb[0].shape[1:], mode='bilinear', align_corners=False)[0].cpu().unsqueeze(0),
                                     labels_gt_rgb[0].cpu().unsqueeze(0),
                                     labels_pred_rgb[0].cpu().unsqueeze(0)], dim=0)

            if trainer.global_rank == 0:
                save_path = os.path.join(self.run_dir,
                                         'viz/train/train_{0:08d}.jpg'.format(trainer.current_epoch))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torchvision.utils.save_image(viz_tensors, save_path,
                                             nrow=4, normalize=True, scale_each=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """Pytorch Lightning built-in function for the end of an train epoch. Will be called BEFORE the on_train_epoch_end of pl module."""
        if trainer.global_rank == 0:
            # Visualize generated masks from random sampled seeds
            random_seeds = torch.randint(200, 10000, size=(4,)).to(pl_module.device)  # Since the images from seed0~200 are manually labeled and used in training and validation

            img_list, seg_logits = pl_module.model(random_seeds, class_idx=pl_module.class_idx, device=pl_module.device, truncation_psi=0.9)
            labels = seg_logits.argmax(dim=1)

            labels_rgb = self.get_segmentation_viz(labels)
            # scale up sample predicted image if the shape does not match the predicted mask
            viz_tensors = torch.cat([img_list.cpu() if img_list[0].shape[1:] == labels_rgb[0].shape[1:] else F.interpolate(img_list, size=labels_rgb[0].shape[1:], mode='bilinear', align_corners=False).cpu(),
                                     labels_rgb], dim=0)
            save_path = os.path.join(self.run_dir,
                                     'viz/sample/sample_{0:08d}.jpg'.format(trainer.current_epoch))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(viz_tensors, save_path,
                                         nrow=4, normalize=True, scale_each=True)

    def get_segmentation_viz(self, masks):
        """Custom function for coloring segmentation masks.
        Args:
            masks: output masks generated from the BigDatasetGAN

        Returns:
            label_pred_rgbs: numpy rgb masks

        """
        label_pred_rgbs = []
        for i in range(masks.shape[0]):
            label_pred_rgb = self.voc_viz(masks[i].cpu().numpy())
            label_pred_rgbs.append(label_pred_rgb)
        label_pred_rgbs = np.stack(label_pred_rgbs)
        label_pred_rgbs = torch.from_numpy(label_pred_rgbs).float() / 255.0
        return label_pred_rgbs


class BigdatasetganPlModel(TAOLightningModule):
    """ PL module for BigDatasetGAN Model."""

    def __init__(self, experiment_spec, dm, export=False):
        """Init training for BigDatasetGAN Model."""
        super().__init__(experiment_spec)

        self.checkpoint_filename = 'bigdatasetgan_model'
        self.dm = dm
        self.cudnn_benchmark = True

        # Init the model
        self._build_model(export)
        self.class_idx = self.model.class_idx  # the generated images belong to the specific class of the StyleGAN-XL generator
        self.n_class = self.model.n_class  # the total segmentation classes of the BigDatasetGAN

        # Init the loss function
        self._build_criterion()

        # Init metrics for logging
        self.iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=self.n_class)
        self.loss_metric = torchmetrics.MeanMetric()

    def configure_callbacks(self):
        """Pytorch Lightning built-in function for setting up callbacks."""
        callbacks = super().configure_callbacks()
        # For monitering the quality of the generated masks
        # Note that the callbacks means non-essential operations for the overall training
        # Thus can be disabled by commenting out the following line.
        visualization_callback = VisualizationCallback(n_class=self.n_class)
        callbacks.append(visualization_callback)

        return callbacks

    def setup(self, stage=None):
        """Pytorch Lightning built-in function for model setup before launching of training, evaluation, and inference."""
        # Set cuda configurations
        if stage in ('fit', 'predict', 'test', None):
            torch.backends.cudnn.benchmark = self.cudnn_benchmark   # Improves training speed.
            torch.backends.cuda.matmul.allow_tf32 = False           # Improves numerical accuracy.
            torch.backends.cudnn.allow_tf32 = False                 # Improves numerical accuracy.

            # To get deterministic results at every launching
            if self.experiment_spec['train']['deterministic_all']:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                # RuntimeError: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation
                # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Set the CUBLAS_WORKSPACE_CONFIG for torch.use_deterministic_algorithms(True)
                # torch.use_deterministic_algorithms(True)

        # Create two folders for inference. 1. for generated images and 2. for corresponding generated masks.
        if stage == 'predict':
            outdir = self.experiment_spec['inference']['results_dir']
            self.pred_images_dir = os.path.join(outdir, "images")
            self.pred_masks_dir = os.path.join(outdir, "masks")
            if self.trainer.global_rank == 0:
                os.makedirs(self.pred_images_dir, exist_ok=True)
                os.makedirs(self.pred_masks_dir, exist_ok=True)

    def configure_optimizers(self):
        """Pytorch Lightning built-in function for optimizers initialization and configuration."""
        if self.experiment_spec['train']['bigdatasetgan']['optim_labeller']['optim'] == "AdamW":
            return torch.optim.AdamW(self.model.feature_labeller.parameters(),
                                     lr=self.experiment_spec['train']['bigdatasetgan']['optim_labeller']['lr'],
                                     betas=self.experiment_spec['train']['bigdatasetgan']['optim_labeller']['betas'])

        else:
            raise NotImplementedError("Optimizer {} is not implemented".format(self.experiment_spec['train']['bigdatasetgan']['optim_labeller']['optim']))

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for training step."""
        seeds, labels = batch

        img_list, seg_logits = self.model(seeds, class_idx=self.class_idx, device=self.device)
        loss = self.criterion(seg_logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "seg_logits": seg_logits, "img_list": img_list, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for validation step."""
        seeds, labels = batch

        _, seg_logits = self.model(seeds, class_idx=self.class_idx, device=self.device)
        loss = self.criterion(seg_logits, labels)
        labels_pred = seg_logits.argmax(dim=1)

        # Update metrics
        self.iou_metric.update(labels_pred, labels)
        self.loss_metric.update(loss)

    def on_validation_epoch_end(self):
        """Pytorch Lightning built-in function for the end of each validation epoch."""
        avg_iou = self.iou_metric.compute()
        avg_loss = self.loss_metric.compute()

        # Reset the metrics for the next epoch
        self.iou_metric.reset()
        self.loss_metric.reset()

        # Log the metrics to status.json
        average_metrics_list = {'avg_loss': avg_loss, 'avg_iou': avg_iou}
        self.status_logging_by_stage(metrics=average_metrics_list, stage='fit')

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for test step."""
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Pytorch Lightning built-in function for the end of an test epoch."""
        avg_iou = self.iou_metric.compute()
        avg_loss = self.loss_metric.compute()

        # Reset the metrics for the next epoch
        self.iou_metric.reset()
        self.loss_metric.reset()

        # Log the metrics to status.json
        average_metrics_list = {'avg_loss': avg_loss, 'avg_iou': avg_iou}
        self.status_logging_by_stage(metrics=average_metrics_list, stage='test')

    def predict_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for predict step."""
        cpu_tensor_seeds = batch.cpu()  # cuda int tensors -> cpu int tensors

        # Generate images.
        truncation_psi = self.experiment_spec['inference']['truncation_psi']
        translate = self.experiment_spec['inference']['translate']
        rotate = self.experiment_spec['inference']['rotate']
        centroids_path = self.experiment_spec['inference']['centroids_path']
        class_idx = self.experiment_spec['inference']['class_idx']

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        img_list, seg_logits = self.model.forward_with_numpy_images(seeds=cpu_tensor_seeds,
                                                                    centroids_path=centroids_path,
                                                                    translate=translate, rotate=rotate,
                                                                    class_idx=class_idx,
                                                                    device=self.device,
                                                                    truncation_psi=truncation_psi)
        labels = seg_logits.argmax(dim=1)

        # Iterates all images and masks generated from the given seeds
        for idx, cpu_tensor_seed in enumerate(cpu_tensor_seeds):
            seed = int(cpu_tensor_seed)  # cpu int tensor -> int
            print("Saving", f'{self.pred_images_dir}/seed{seed:04d}.png')
            PIL.Image.fromarray(gen_utils.create_image_grid(img_list[idx]), 'RGB').save(f'{self.pred_images_dir}/seed{seed:04d}.png')

            labels_np = labels[idx].squeeze().cpu().numpy()
            labels_np = (labels_np * 255).astype(np.uint8)
            print("Saving", f'{self.pred_masks_dir}/seed{seed:04d}.png')
            PIL.Image.fromarray(labels_np, mode='L').save(f'{self.pred_masks_dir}/seed{seed:04d}.png')

    def status_logging_by_stage(self, metrics, stage):
        """Custom function for metrics logging to status.json based on stage.
        Args:
            metrics: evaluation metrics involoved in the stage
            stage: current stage (fit, test, predict).

        Returns:
            None

        """
        # set the prefix string which is used in logging
        if stage == 'fit':
            short_prefix = 'val'
            full_prefix = 'Validation'
        elif stage == 'test':
            short_prefix = 'test'
            full_prefix = 'Evaluation'
        else:
            raise NotImplementedError("Stage {} is not implemented".format(stage))

        self.status_logging_dict = {}
        for metric_name in metrics:
            # Log to screen's progress bar
            self.log(short_prefix + "_" + str(metric_name), metrics[metric_name], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Prepare data for the below logging to status.json
            self.status_logging_dict[short_prefix + "_" + metric_name] = float(metrics[metric_name])
        self.status_logging_dict["epoch"] = int(self.trainer.current_epoch)

        # Log to status.json
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message=full_prefix + " metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
