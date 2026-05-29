# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom TAO Status Logger for Segformer"""

# import os
from typing import Dict, Optional, Sequence, Union
# import json
from time import strftime, gmtime

from mmengine.registry import HOOKS
from mmengine.hooks import LoggerHook

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]
STATUS_JSON_FILENAME = "status.json"


@HOOKS.register_module()
class TAOBEVFusionLoggerHook(LoggerHook):
    """ Logger hook in text. """

    def __init__(self, *args, **kwargs):
        """ init """
        super().__init__(*args, **kwargs)
        self.s_logger = status_logging.get_status_logger()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader,
                                batch_idx) and (not self.ignore_last or len(runner.train_dataloader) <= self.interval)):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

        # log_str looks like:
        # Epoch(train) [1][5/5]  base_lr: 1.6017e-07 lr: 1.6017e-07  eta: 0:12:26  time: 0.7499  data_time: 0.0277  memory: 3490  loss: 0.5401  decode.loss_ce: 0.5401  decode.acc_seg: 69.6967
        # Epoch(train)  [1][  3/477]  lr: 6.7201e-05  eta: 5:32:50  time: 4.1892  data_time: 0.3183  memory: 9917  grad_norm: 50.4355  loss: 6.3231  loss_heatmap: 2.7920  layer_-1_loss_cls: 0.4822  layer_-1_loss_bbox: 3.0489  matched_ious: 0.1486

        self.monitor_data = {}
        self.monitor_data["mode"] = "train"

        self.monitor_data["epoch"] = runner.iter + 1
        time = float(log_str.split(" time: ")[1].split(" ")[0])
        if time < 1:
            self.monitor_data["time_per_iter"] = f"00:00:0{time}"
        else:
            self.monitor_data["time_per_iter"] = strftime("%H:%M:%S", gmtime(time))

        self.monitor_data["eta"] = log_str.split(" eta: ")[1].split(" ")[0]

        self.monitor_data["lr"] = float(log_str.split(" lr: ")[1].split(" ")[0])

        running_avg_loss = float(log_str.split(" loss: ")[1].split(" ")[0])
        self.monitor_data["loss"] = running_avg_loss

        matched_ious = float(log_str.split(" matched_ious: ")[1].split(" ")[0])
        self.monitor_data["matched_ious"] = matched_ious

        self.s_logger.graphical = {
            "loss": running_avg_loss,
            "matched_ious": matched_ious}

        self.write_to_status(runner)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Record logs after testing iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'test')
            runner.logger.info(log_str)

            # log_str looks like:
            # "aAcc: 87.08 mIoU: 63.78 mAcc: 71.76 data_time: 0.028624820709228515 time: 0.10705928802490235"

            self.monitor_data = {}
            self.monitor_data["mode"] = "test"
            miou = float(log_str.split(" mIoU: ")[1].split(" ")[0])
            macc = float(log_str.split(" mAcc: ")[1].split(" ")[0])
            self.monitor_data["Mean IOU"] = miou
            self.monitor_data["mAcc"] = macc
            self.s_logger.kpi = {"Mean IOU": miou, "mAcc": macc}
            self.write_to_status(runner)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if (isinstance(runner._train_loop, dict) or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict) or runner._train_loop is None):
                iter = 0  # noqa pylint: disable=W0622
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(
                tag, step=iter, file_path=self.json_log_path)

        # log_str looks like:
        # "aAcc: 85.69 mIoU: 61.1 mAcc: 69.86 data_time: 0.07468271255493164 time: 0.7449096838633219 step: 500"
        print(log_str)
        self.monitor_data = {}
        self.monitor_data["mode"] = "val"
        self.monitor_data["epoch"] = epoch
        mAP40 = float(log_str.split(" TAO3D metric/pred_instances_3d/TAO3D/person_3D_AP40_moderate_strict: ")[1].split(" ")[0])
        mAP11 = float(log_str.split(" TAO3D metric/pred_instances_3d/TAO3D/person_3D_AP11_moderate_strict: ")[1].split(" ")[0])
        self.monitor_data["AP40"] = mAP40
        self.monitor_data["AP11"] = mAP11
        self.s_logger.kpi = {"AP40": mAP40, "AP11": mAP11}
        self.write_to_status(runner)

    def write_to_status(self, runner):
        """Write the monitor data to the status logger."""
        try:
            self.s_logger.write(
                data=self.monitor_data,
                status_level=status_logging.Status.RUNNING)
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass
