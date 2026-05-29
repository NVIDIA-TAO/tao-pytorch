# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main PTL model file for OCDnet."""

import pathlib
import time
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import os
import math
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from copy import deepcopy
from typing import Any
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.cv.ocdnet.model.build_nn_model import build_ocd_model
from nvidia_tao_pytorch.cv.ocdnet.lr_schedulers.schedulers import WarmupPolyLR
from nvidia_tao_pytorch.cv.ocdnet.model.model import build_loss
from nvidia_tao_pytorch.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_pytorch.cv.ocdnet.utils.ocr_metric.icdar2015.quad_metric import get_metric
from nvidia_tao_pytorch.cv.ocdnet.utils.util import create_logger, draw_bbox, save_result, show_img
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


# TODO @seanf: cc says this is never used
# pylint:disable=too-many-ancestors
class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        """Init."""
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = lambda x: decay * (1 - math.exp(-float(x) / 2000))
        self.updates = 0
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        """Implementation of updating the module."""
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """Update the EMA module."""
        self.updates += 1
        d = self.decay(self.updates)
        self._update(model, update_fn=lambda e, m: d * e + (1. - d) * m)

    def set(self, model):
        """Set the new EMA module."""
        self._update(model, update_fn=lambda e, m: m)


class OCDnetModel(TAOLightningModule):
    """PTL module for single stream OCDnet."""

    def __init__(self, experiment_spec, dm, task, export=False):
        """Init training for OCDnet model.

        Args:
            experiment_spec (dict): The experiment specification.
            dm (DataModule)
            task (str)
            export (bool, optional): Whether to build the model that can be exported to ONNX format. Defaults to False
        """
        super().__init__(experiment_spec)
        self.train_config = self.experiment_spec["train"]
        self.epochs = self.train_config["num_epochs"]
        self.post_process = get_post_processing(self.train_config['post_processing'])
        self.box_thresh = self.train_config['post_processing']["args"]["box_thresh"]
        self.checkpoint_dir = self.train_config["results_dir"]
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.train_loss = 0.0
        self.criterion = build_loss(self.train_config['loss'])
        # init the model
        self._build_model(export)
        self.model_ema = None
        if self.train_config['model_ema']:
            self.model_ema = ModelEmaV2(self.model, decay=self.train_config['model_ema_decay'])
            self.metrics.update({'ema_recall': 0, 'precision': 0, 'ema_hmean': 0, 'ema_best_model_epoch': 0})
        self.name = self.model.name

        self.dm = dm
        if task == 'fit':
            self.train_loader_len = self.dm.train_loader_len

        self.console_logger = create_logger()
        self.status_logging_dict = {}

        self.checkpoint_filename = 'ocd_model'

    def _build_model(self, export):
        """Internal function to build the model.

        This method constructs a model using the specified experiment specification and export flag. It returns the model.

        Args:
            experiment_spec (dict): The experiment specification.
            export (bool): Whether to build the model that can be exported to ONNX format.
        """
        self.model = build_ocd_model(experiment_config=self.experiment_spec,
                                     export=export)

    def forward(self, x):
        """Forward of the ocdnet model."""
        output = self.model(x)
        return output

    def configure_optimizers(self):
        """Configure optimizers for training"""
        optim_dict = {}

        self.warmup_epochs = self.train_config['lr_scheduler']['args']['warmup_epoch']
        self.warmup_iters = self.warmup_epochs * self.train_loader_len
        self.optimizer = self._initialize('optimizer', torch.optim, self.model.parameters())
        self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                      warmup_iters=self.warmup_iters, warmup_epochs=self.warmup_epochs, epochs=self.epochs,
                                      **self.train_config['lr_scheduler']['args'])

        optim_dict["optimizer"] = self.optimizer
        optim_dict["lr_scheduler"] = self.scheduler

        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (Tensor): Batch of data.
            batch_idx (int): Index of batch.

        Returns:
            loss (float): Loss value for each step in training.

        """
        self.train_loss = 0.
        preds = self.model(batch['img'])
        batch_size = batch['img'].shape[0]
        loss_dict = self.criterion(preds, batch)
        loss = loss_dict['loss']
        self.train_loss += loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

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

    def on_validation_epoch_start(self):
        """Perform on validation."""
        self.raw_metrics = []

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        preds = self.model(batch['img'])
        self.metric_cls = get_metric(self.train_config['metric'])
        boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
        raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)

        if self.model_ema is not None:
            ema_preds = self.model_ema.module(batch['img'])
            boxes, scores = self.post_process(batch, ema_preds, is_output_polygon=self.metric_cls.is_output_polygon)
            ema_raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)
            self.raw_metrics.append((raw_metric, ema_raw_metric))
            return (raw_metric, ema_raw_metric)

        self.raw_metrics.append((raw_metric,))
        return (raw_metric,)

    def on_validation_epoch_end(self):
        """Validation step end."""
        if self.model_ema is not None:
            ema_metrics = self.metric_cls.gather_measure([ema_metrics[1] for ema_metrics in self.raw_metrics])
            ema_recall = ema_metrics['recall'].avg
            ema_precision = ema_metrics['precision'].avg
            ema_hmean = ema_metrics['hmean'].avg
            self.log("ema_recall", ema_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("ema_precision", ema_precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("ema_hmean", ema_hmean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = self.metric_cls.gather_measure([metrics[0] for metrics in self.raw_metrics])
        recall = metrics['recall'].avg
        precision = metrics['precision'].avg
        hmean = metrics['hmean'].avg

        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("hmean", hmean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        save_best = False
        save_best_ema = False
        if self.metric_cls is not None:
            if hmean >= self.metrics['hmean']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['hmean'] = hmean
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['best_model_epoch'] = self.current_epoch

            if self.model_ema is not None:
                if ema_hmean >= self.metrics['ema_hmean']:
                    save_best_ema = True
                    self.metrics['ema_hmean'] = ema_hmean
                    self.metrics['ema_precision'] = ema_precision
                    self.metrics['ema_recall'] = ema_recall
                    self.metrics['ema_best_model_epoch'] = self.current_epoch
        else:
            if (self.train_loss / self.train_loader_len) <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.train_loss / self.train_loader_len
                self.metrics['best_model_epoch'] = self.current_epoch
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.print(best_str)

        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)
        if self.model_ema is not None:
            net_save_path_best_ema = '{}/model_best_ema.pth'.format(self.checkpoint_dir)

        if save_best:
            self._save_checkpoint(self.current_epoch, net_save_path_best)
            self.print("Saving current best: {}".format(net_save_path_best))

        # TODO @seanf: the code coverage report shows that we never test the ema mode at all
        if save_best_ema:
            self._save_checkpoint(self.current_epoch, net_save_path_best_ema, save_ema=True)

        if self.trainer.is_global_zero:
            self.console_logger.info('**********************Start logging Evaluation Results **********************')
            self.console_logger.info('current_epoch : {}'.format(self.current_epoch))
            self.console_logger.info('lr : {:.9f}'.format(*self.scheduler.get_lr()))
            self.console_logger.info('recall : {:2.5f}'.format(recall))
            self.console_logger.info('precision : {:2.5f}'.format(precision))
            self.console_logger.info('hmean : {:2.5f}'.format(hmean))
            if self.model_ema:
                self.console_logger.info('ema_recall : {:2.5f}'.format(ema_recall))
                self.console_logger.info('ema_precision : {:2.5f}'.format(ema_precision))
                self.console_logger.info('ema_hmean : {:2.5f}'.format(ema_hmean))

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["recall"] = str(recall)
            self.status_logging_dict["precision"] = str(precision)
            self.status_logging_dict["hmean"] = str(hmean)
            if self.model_ema:
                self.status_logging_dict["ema_recall"] = str(ema_recall)
                self.status_logging_dict["ema_precision"] = str(ema_precision)
                self.status_logging_dict["ema_hmean"] = str(ema_hmean)
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
        self.raw_metrics.clear()
        pl.utilities.memory.garbage_collection_cuda()
        return metrics

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.test_post_process = get_post_processing(self.experiment_spec['evaluate']['post_processing'])
        self.test_metric_cls = get_metric(self.experiment_spec['evaluate']['metric'])
        self.test_box_thresh = self.experiment_spec['evaluate']['post_processing']["args"]["box_thresh"]
        self.test_thresh_range = [i * 0.1 for i in range(1, 10)]
        self.test_raw_metrics = {thresh: [] for thresh in self.test_thresh_range}
        self.test_total_frame = 0.0
        self.test_total_time = 0.0

    def test_step(self, batch, batch_idx):
        """Test step"""
        start = time.time()
        preds = self.model(batch['img'])
        for thresh in self.test_thresh_range:
            self.test_post_process.thresh = thresh
            boxes, scores = self.test_post_process(batch, preds, is_output_polygon=self.test_metric_cls.is_output_polygon)
            self.test_total_frame += batch['img'].size()[0]
            self.test_total_time += time.time() - start
            raw_metric = self.test_metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.test_box_thresh)
            self.test_raw_metrics[thresh].append(raw_metric)

    def on_test_epoch_end(self):
        """Test epoch end"""
        metrics = {thresh: {} for thresh in self.test_thresh_range}
        metrics['best'] = None
        best_hmean = 0
        for thresh in self.test_thresh_range:
            metric = self.test_metric_cls.gather_measure(self.test_raw_metrics[thresh])
            metrics[thresh] = {'recall': metric['recall'].avg, 'precision': metric['precision'].avg, 'hmean': metric['hmean'].avg}
            msg = f"thresh: {round(thresh, 1)}, recall: {metric['recall'].avg}, precision: {metric['precision'].avg}, hmean: {metric['hmean'].avg}"
            status_logging.get_status_logger().write(message=msg)
            if metric['hmean'].avg > best_hmean:
                best_hmean = metric['hmean'].avg
                metrics['best'] = {'Thresh': round(thresh, 1), 'Recall': metric['recall'].avg, 'Precision': metric['precision'].avg, 'Hmean': metric['hmean'].avg}

        status_logging.get_status_logger().kpi = metrics['best']
        status_logging.get_status_logger().write(
            message="Test metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        self.predict_post_process = get_post_processing(self.experiment_spec['inference']['post_processing'])
        self.predict_post_process.box_thresh = self.experiment_spec['inference']['post_processing']['args']['box_thresh']
        self.predict_polygon = self.experiment_spec['inference']['polygon']
        self.predict_show = self.experiment_spec['inference']['show']

    def predict_step(self, batch, batch_idx):
        """Predict step"""
        # For now, we assume (and set) the batch size is 1
        tensor, img_path = batch["img"], batch["img_path"][0]
        preds = self.model(tensor)
        box_list, score_list = self.predict_post_process(batch, preds, is_output_polygon=self.predict_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            if self.predict_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # filer bbox has all 0
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []

        preds = preds[0, 0, :, :].detach().cpu().numpy()

        im = cv2.imread(img_path)
        img = draw_bbox(im[:, :, ::-1], box_list)
        if self.predict_show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # save result
        img_path = pathlib.Path(img_path)
        inference_results_dir = self.experiment_spec["results_dir"]
        output_path = os.path.join(inference_results_dir, img_path.stem + '_result.jpg')
        pred_path = os.path.join(inference_results_dir, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), box_list, score_list, self.predict_polygon)

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.train_config[name]['type']
        module_args = self.train_config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        if module_name == "SGD":
            module_args.pop("amsgrad")
        elif module_name == "Adam":
            module_args.pop("momentum")
        return getattr(module, module_name)(*args, **module_args)

    # TODO @seanf: is this function necessary? It seems like on_validation_epoch_end is manually saving, but we have a callback for this?
    def _save_checkpoint(self, epoch, file_name, save_ema=False):
        """Saving checkpoints

        Args:
            epoch: Current epoch number
            log: The logging information of the epoch
            save_best: If True, rename the saved checkpoint with 'model_best' prefix
        """
        state_dict = self.model.state_dict()
        if save_ema:
            state_dict = self.model_ema.module.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.experiment_spec,
            'metrics': self.metrics
        }
        torch.save(state, file_name)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """train batch end."""
        if self.model_ema is not None:
            self.model_ema.update(self.model)
