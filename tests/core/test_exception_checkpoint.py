# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exception-time checkpoint behavior."""

from unittest.mock import MagicMock, patch

from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.model_checkpoint import (
    TAOExceptionCheckpoint,
)


def test_distributed_exception_skips_collective_checkpoint(tmp_path):
    """A rank-local distributed failure must not enter checkpoint barriers."""
    callback = TAOExceptionCheckpoint(str(tmp_path))
    trainer = MagicMock(world_size=8, current_epoch=2, global_step=7)

    with patch.object(ModelCheckpoint, "_link_checkpoint") as link_checkpoint:
        callback.on_exception(trainer)

    trainer.save_checkpoint.assert_not_called()
    link_checkpoint.assert_not_called()


def test_single_process_exception_still_saves_and_links(tmp_path):
    """Single-process training should retain the existing recovery checkpoint."""
    callback = TAOExceptionCheckpoint(str(tmp_path))
    trainer = MagicMock(world_size=1, current_epoch=2, global_step=7)

    with (
        patch.object(TAOExceptionCheckpoint, "CHECKPOINT_NAME_LAST", "clip_latest"),
        patch.object(TAOExceptionCheckpoint, "FILE_EXTENSION", ".pth"),
        patch.object(ModelCheckpoint, "_link_checkpoint") as link_checkpoint,
    ):
        callback.on_exception(trainer)

    expected_checkpoint = str(tmp_path / "model_epoch_002_step_00007.pth")
    trainer.save_checkpoint.assert_called_once_with(expected_checkpoint)
    link_checkpoint.assert_called_once_with(
        trainer,
        expected_checkpoint,
        str(tmp_path / "clip_latest.pth"),
    )
