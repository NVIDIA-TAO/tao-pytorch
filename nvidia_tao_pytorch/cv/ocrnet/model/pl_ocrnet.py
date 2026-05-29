# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Main PTL model file for OCRNet """

from copy import deepcopy
import math
import random
from typing import Any, Dict
import pickle
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import modelopt.torch.opt as mto

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.model.build_nn_model import build_ocrnet_model
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import (CTCLabelConverter,
                                                      AttnLabelConverter)
TABLE_HEADER = ['Ground Truth', 'Prediction', 'Confidence && T/F']


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
        # self.decay = decay
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


# pylint:disable=too-many-ancestors
class OCRNetModel(TAOLightningModule):
    """ PTL module for OCRNet."""

    def __init__(self, experiment_spec: ExperimentConfig, dm):
        """Init training for OCRNet."""
        super().__init__(experiment_spec)
        self.dm = dm
        self.quantized = experiment_spec.model.quantize
        if self.quantized:
            self.save_func = mto.save
        else:
            self.save_func = torch.save

        with open(self.dataset_config.character_list_file, "r") as f:
            self.characters = "".join([ch.strip() for ch in f.readlines()])

        # init the label converter and criterion
        self.max_label_length = self.dataset_config.max_label_length
        if 'CTC' in self.model_config.prediction:
            self.ctc = True
            self.converter = CTCLabelConverter(self.characters)
            self.criterion = torch.nn.CTCLoss(zero_infinity=True)
        else:
            self.ctc = False
            self.converter = AttnLabelConverter(self.characters)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0

        self.num_class = len(self.converter.character)
        # init the model
        self.model = None
        self._build_model()
        self.model_ema = None
        if self.experiment_spec.train.model_ema:
            self.model_ema = ModelEmaV2(self.model)

        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_class)
        self.best_acc = -1
        if self.model_ema is not None:
            self.val_ema_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_class)
            self.best_ema_acc = -1
        self.check_val_batch_idx = 0

        self.status_logging_dict = {}

        if self.model_ema is not None:
            self.status_logging_dict["val_ema_acc"] = 0.0

        self.checkpoint_filename = 'ocr_model'

    def _build_model(self):
        """Internal function to build the model."""
        self.model = build_ocrnet_model(experiment_spec=self.experiment_spec,
                                        num_class=self.num_class)

    def configure_optimizers(self):
        """Configure optimizers for training"""
        self.train_config = self.experiment_spec["train"]
        optim_dict = {}
        # filter that only require gradient decent
        filtered_parameters = []
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            filtered_parameters.append(p)

        if self.train_config['optim']['name'] == 'adam':
            optim = torch.optim.Adam(filtered_parameters,
                                     lr=self.train_config['optim']['lr'],
                                     betas=(0.9, 0.999))
        elif self.train_config['optim']['name'] == 'adadelta':
            optim = torch.optim.Adadelta(filtered_parameters,
                                         lr=self.train_config['optim']['lr'],
                                         rho=0.95,
                                         eps=1e-8)
        optim_dict["optimizer"] = optim
        # # Uncomment the following codes to enable learning rate scheduler
        # scheduler_type = self.train_config['optim']['lr_scheduler']
        # if scheduler_type == "AutoReduce":
        #     lr_scheduler = ReduceLROnPlateau(optim, 'min',
        #                                      patience=self.train_config['optim']['patience'],
        #                                      min_lr=self.train_config['optim']['min_lr'],
        #                                      factor=self.train_config['optim']["lr_decay"],
        #                                      verbose=True)
        # elif scheduler_type == "MultiStep":
        #     lr_scheduler = MultiStepLR(optimizer=optim,
        #                                milestones=self.train_config['optim']["lr_steps"],
        #                                gamma=self.train_config['optim']["lr_decay"],
        #                                verbose=True)
        # else:
        #     raise ValueError("Only [AutoReduce, MultiStep] scheduler is supported")

        # optim_dict["lr_scheduler"] = lr_scheduler
        # optim_dict['monitor'] = self.train_config['optim']['lr_monitor']

        return optim_dict

    def forward(self, x):
        """Forward of the OCRNet model. No decode in the forward."""
        image = x
        batch_size = image.size(0)
        # For max length prediction
        text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0)
        if 'CTC' in self.model_config.prediction:
            preds = self.model(image, text_for_pred)
        else:
            preds = self.model(image, text_for_pred, is_train=False)

        return preds

    def training_step(self, batch, batch_idx):
        """Training step."""
        image, labels = batch
        text, length = self.converter.encode(labels, batch_max_length=self.max_label_length)
        batch_size = image.size(0)

        if self.ctc:
            preds = self.model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = self.criterion(preds, text, preds_size, length)

        else:
            preds = self.model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        self.log("train_loss", cost, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return cost

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update the EMA model at the train batch end."""
        if self.model_ema is not None:
            self.model_ema.update(self.model)

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

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        image, labels = batch
        batch_size = image.size(0)
        # For max length prediction
        length_for_pred = torch.IntTensor([self.max_label_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.max_label_length)

        # For ordinary model
        fake_target, preds_str, confidence_score_list, cost = self._evaluate_batch(self.model, image, labels, batch_size,
                                                                                   length_for_pred, text_for_pred,
                                                                                   text_for_loss, length_for_loss)
        show = min(5, batch_size)
        if batch_idx == self.check_val_batch_idx:
            table_data = []
            for gt, pred, confidence in zip(labels, preds_str[:show], confidence_score_list[:show]):
                if not self.ctc:
                    pred = pred[:pred.find('[s]')]

                table_data.append((gt, pred, f"{confidence:0.4f} {str(pred == gt)}"))
            table = tabulate(table_data, headers=TABLE_HEADER, tablefmt='psql')
            self.infer_table = table
            self.check_val_batch_idx = random.randint(0, max(self.dm.val_batch_num - 1, 0))

        fake_output = torch.IntTensor([1] * batch_size)
        self.val_accuracy.update(fake_output, fake_target)
        self.log("val_loss", cost, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc_1", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # For EMA model
        if self.model_ema is not None:
            fake_target, _, _, _ = self._evaluate_batch(self.model_ema.module, image, labels, batch_size,
                                                        length_for_pred, text_for_pred,
                                                        text_for_loss, length_for_loss)
            self.val_ema_accuracy.update(fake_output, fake_target)
            self.log("ema_val_acc_1", self.val_ema_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return cost

    def on_validation_epoch_end(self) -> None:
        """Save the best accuracy model after validation"""
        @rank_zero_only
        def val_epoch_end():
            current_acc = self.val_accuracy.compute()
            if current_acc > self.best_acc:
                self.save_func(self.model, f'{self.experiment_spec.train.results_dir}/best_accuracy.pth')
                self.best_acc = current_acc
            current_model_log = f'{"Current_accuracy":17s}: {current_acc:0.3f}'
            best_model_log = f'{"Best_accuracy":17s}: {self.best_acc:0.3f}'
            self.dm.console_logger.info(f'{current_model_log}')
            self.dm.console_logger.info(f'{best_model_log}')
            if self.model_ema is not None:
                current_ema_acc = self.val_ema_accuracy.compute()
                if current_ema_acc > self.best_ema_acc:
                    self.save_func(self.model_ema.module, f'{self.experiment_spec.train.results_dir}/best_accuracy_ema.pth')
                    self.best_ema_acc = current_ema_acc
                current_model_log = f'{"Current_ema_accuracy":17s}: {current_ema_acc:0.3f}'
                best_model_log = f'{"Best_ema_accuracy":17s}: {self.best_ema_acc:0.3f}'
                self.dm.console_logger.info(f'{current_model_log}')
                self.dm.console_logger.info(f'{best_model_log}')
            infer_table_list = self.infer_table.split("\n")
            for table in infer_table_list:
                self.dm.console_logger.info(table)

        val_epoch_end()

        # status logging
        average_val_loss = self.trainer.logged_metrics["val_loss"].item()

        if not self.trainer.sanity_checking:
            self.status_logging_dict = {}
            self.status_logging_dict["val_loss"] = average_val_loss
            self.status_logging_dict["val_acc"] = self.val_accuracy.compute().item()
            if self.model_ema is not None:
                self.status_logging_dict["val_acc"] = self.val_ema_accuracy.compute().item()
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

        pl.utilities.memory.garbage_collection_cuda()

    def on_test_epoch_start(self):
        """Test epoch start"""
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                   num_classes=self.num_class,
                                                   compute_on_cpu=False,
                                                   sync_on_compute=False,
                                                   dist_sync_on_step=True)

    def test_step(self, batch, batch_idx):
        """Test step"""
        image, labels = batch
        batch_size = image.size(0)
        # For max length prediction
        length_for_pred = torch.IntTensor([self.max_label_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.max_label_length)

        # For ordinary model
        fake_target, _, _, _ = self._evaluate_batch(self.model, image, labels, batch_size,
                                                    length_for_pred, text_for_pred,
                                                    text_for_loss, length_for_loss)

        fake_output = torch.IntTensor([1] * batch_size)
        self.test_accuracy.update(fake_output, fake_target)

    def on_test_epoch_end(self):
        """Test epoch end"""
        test_acc = self.test_accuracy.compute().item() * 100
        if self.trainer.local_rank == 0:
            self.dm.console_logger.info(f"Accuracy: {test_acc:0.3f}")
            self.status_logging_dict = {}
            self.status_logging_dict["test_acc"] = test_acc
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
            self.test_accuracy = None

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        self.table_header = ["image_path", "predicted_labels", "confidence score"]
        self.table_data = []

    def predict_step(self, batch, batch_idx):
        """Predict step"""
        image, img_path_list = batch
        batch_size = image.size(0)
        # For max length prediction
        length_for_pred = torch.IntTensor([self.max_label_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, self.max_label_length + 1).fill_(0)

        self._infer_batch(self.model, image, img_path_list, batch_size, length_for_pred, text_for_pred)

    def on_predict_epoch_end(self):
        """Predict epoch end"""
        print(tabulate(self.table_data, headers=self.table_header, tablefmt="psql"))
        self.table_data = []

    def _evaluate_batch(self, model, image, labels, batch_size,
                        length_for_pred, text_for_pred,
                        text_for_loss, length_for_loss):
        """Evaluate batch of data."""
        if self.ctc:
            preds = model(image, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            # if opt.baiduCTC:
            #     cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            cost = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            # if opt.baiduCTC:
            #     _, preds_index = preds.max(2)
            #     preds_index = preds_index.view(-1)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred, is_train=False)

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        fake_target = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if not self.ctc:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                fake_target.append(1)
            else:
                fake_target.append(0)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except Exception:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

        fake_target = torch.IntTensor(fake_target)

        return fake_target, preds_str, confidence_score_list, cost

    def _infer_batch(self, model, image, img_path_list, batch_size,
                     length_for_pred, text_for_pred):
        if self.ctc:
            preds = model(image, text_for_pred)
            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for img_name, pred, pred_max_prob in zip(img_path_list, preds_str, preds_max_prob):
            if not self.ctc:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except Exception:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

            self.table_data.append((img_name, pred, f"{confidence_score:0.4f}"))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save the model architecture in the checkpoint"""
        # This is for resuming training from a pruned model. The default ckpt is saved with weights only.
        if not self.quantized:
            # ModelOpt quantized model cannot be pickle dump.
            checkpoint["whole_model"] = pickle.dumps(self.model)
