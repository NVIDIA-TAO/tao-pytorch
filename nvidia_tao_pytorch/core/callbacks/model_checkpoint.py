# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File containing Overloaded version of PTL OnExceptionCheckpoint."""

import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, OnExceptionCheckpoint


class TAOExceptionCheckpoint(OnExceptionCheckpoint):
    """A custom checkpointing callback for when training is interrupted.

    Extending Lightning's OnExceptionCheckpoint since it (in v.2.3.0) only supports saving
    to a provided path. We want to extend its capabilities to also symlink the *_latest.pth
    file to this dumped checkpoint.
    """

    CHECKPOINT_NAME_LAST = ""

    def __init__(self, dirpath):  # pylint: disable=useless-parent-delegation
        """Callback init. Uses parent's default filename since we override below"""
        super().__init__(dirpath)

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        """Overriden function that saves and links the checkpoint"""
        self.filename = f"model_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}"
        super().on_exception(trainer)
        ModelCheckpoint._link_checkpoint(trainer, self.ckpt_path, os.path.join(self.dirpath, self.CHECKPOINT_NAME_LAST + self.FILE_EXTENSION))
