# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status Logger callback."""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from datetime import timedelta

import os
import time

from pytorch_lightning import Callback

import numpy as np
import six

from nvidia_tao_pytorch.core.loggers.api_logging import (
    get_status_logger,
    set_status_logger,
    Status,
    StatusLogger,
    Verbosity
)

# Get default status logger() if it's been previously defined.
logger = get_status_logger()

KEY_MAP = {
    "val_loss": "validation_loss",
    "val_acc": "validation_accuracy",
    "loss": "loss",
    "acc": "training_accuracy",
    "lr": "learning_rate",
    "mAP": "mean average precision"
}


class TAOStatusLogger(Callback):
    """Callback that streams the data training data to a status.json file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    logger = TAOStatusLogger('/path/to/results_dir')
    model.fit(X_train, Y_train, callbacks=[logger])
    ```

    # Arguments
        results_dir (str): The directory where the logs will be saved.
        verbosity (status_logger.verbosity.Verbosity()): Verbosity level.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, results_dir,
                 verbosity=Verbosity.INFO,
                 append=False):
        """Instantiate the TAOStatusLogger."""
        # Make sure that the status logger obtained is always
        # an instance of iva.common.logging.logging.StatusLogger.
        # Otherwise, this data get's rendered in stdout.
        if isinstance(get_status_logger(), StatusLogger):
            self.logger = get_status_logger()
        else:
            set_status_logger(StatusLogger(
                filename=os.path.join(results_dir, "status.json"),
                verbosity=verbosity,
                append=append)
            )
            self.logger = get_status_logger()
        self.keys = None
        # Used for avg timing
        self._epoch_start_time = -1
        self._step_start_time = -1
        self.num_steps_in_experiment = 0
        self.avg_time_per_batch = 0
        super(TAOStatusLogger, self).__init__()

    def on_train_start(self, trainer, pl_module):
        """Write data beginning of the training."""
        self.logger.write(
            status_level=Status.STARTED,
            message="Starting Training Loop."
        )

    @staticmethod
    def _handle_value(k):
        is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
        if isinstance(k, six.string_types):
            return k
        if isinstance(k, Iterable) and not is_zero_dim_ndarray:
            return '"[%s]"' % (', '.join(map(str, k)))
        return k

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Routines to be run at the beginning of a batch."""
        if self._epoch_start_time == -1:
            # Resuming in the middle of an epoch, so on_train_epoch_start() won't be run
            self._epoch_start_time = time.time()
        self._step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Computing timing for training on a batch/step"""
        step_end_time = time.time()
        time_per_batch = step_end_time - self._step_start_time
        # Computing the cumulative average
        self.avg_time_per_batch = (time_per_batch + self.num_steps_in_experiment * self.avg_time_per_batch) / (self.num_steps_in_experiment + 1)
        self.num_steps_in_experiment += 1

    def on_train_epoch_start(self, trainer, pl_module):
        """Routines to be run at the beginning of the epoch."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect data at the end of an epoch."""
        data = {}
        data["epoch"] = trainer.current_epoch
        data["step"] = trainer.global_step
        data["max_epoch"] = trainer.max_epochs - 1
        if trainer.max_steps < 1:
            data["max_step"] = trainer.max_epochs * trainer.num_training_batches
        else:
            data["max_step"] = trainer.max_steps
        epoch_end_time = time.time()
        time_per_epoch = epoch_end_time - self._epoch_start_time
        eta = (trainer.max_epochs - 1 - trainer.current_epoch) * time_per_epoch
        data["time_per_epoch"] = str(timedelta(seconds=time_per_epoch))
        data["time_per_step"] = str(timedelta(seconds=self.avg_time_per_batch))
        # TODO @seanf: extra logic for eta when using steps instead of epochs
        data["eta"] = str(timedelta(seconds=eta))
        self.logger.write(data=data, message="Training loop in progress")

    def on_train_end(self, trainer, pl_module):
        """Callback function run at the end of training."""
        self.logger.write(
            status_level=Status.RUNNING,
            message="Training loop complete."
        )
