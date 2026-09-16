# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-only checkpoint publication and retention."""

import os
from pathlib import Path

import torch

from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import CustomModelCheckpoint
from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint_retention import prune_periodic_ssl_checkpoints
from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint


def _atomic_torch_save(value, path: str) -> None:
    """Publish a checkpoint export without exposing a partial final file."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class DinoV3ModelCheckpoint(CustomModelCheckpoint):
    """Custom callback for saving DINOv3 checkpoint"""

    FILE_EXTENSION = ".pth"
    CHECKPOINT_EQUALS_CHAR = "_"
    CHECKPOINT_NAME_LAST = "dinov3_model_latest"

    def __init__(self, *args, save_final_epoch=False, keep_last_n=0, **kwargs):
        """Optionally publish the final manifest-training extent off cadence."""
        super().__init__(*args, **kwargs)
        self.save_final_epoch = save_final_epoch
        self.keep_last_n = int(keep_last_n)

    def on_train_epoch_end(self, trainer, pl_module):
        """Complete manifest training without increasing periodic save frequency."""
        super().on_train_epoch_end(trainer, pl_module)
        if (
            self.save_final_epoch and
            trainer.max_epochs is not None and
            trainer.current_epoch + 1 == trainer.max_epochs and
            self._last_global_step_saved != trainer.global_step and
            not self._should_skip_saving_checkpoint(trainer)
        ):
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Saves the model checkpoint, including custom handling for student and teacher states.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance, providing access to model and training information.
            filepath (str): The file path where the checkpoint will be saved.
        """
        # Call the original save_checkpoint method to save the checkpoint as usual
        trainer.save_checkpoint(filepath, self.save_weights_only)
        # FSDP state-dict extraction is collective, so every rank must enter it.
        # Only global rank zero may publish the converted checkpoint files.
        state_dict = trainer.strategy.lightning_module_state_dict()

        if not trainer.is_global_zero:
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath
            return

        names = ("student", "student_ema") if (
            trainer.lightning_module.model_config.distill.enable
        ) else ("student", "teacher")
        for name in names:
            exported = _backbone_state_dict(state_dict, name)
            output = os.path.join(
                trainer.default_root_dir,
                f"{name}_epoch_{trainer.current_epoch:03d}_step_"
                f"{trainer.global_step:05d}{self.FILE_EXTENSION}",
            )
            _atomic_torch_save(exported, output)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        prune_periodic_ssl_checkpoints(trainer.default_root_dir, self.keep_last_n)

        # Notify loggers
        for logger in trainer.loggers:
            logger.after_save_checkpoint(self)


class DinoV3ExceptionCheckpoint(TAOExceptionCheckpoint):
    """Use DINOv3 checkpoint names without mutating the shared callback class."""

    FILE_EXTENSION = ".pth"
    CHECKPOINT_NAME_LAST = "dinov3_model_latest"


def _backbone_state_dict(state_dict, model_name):
    """Extract one stripped DINOv3 backbone from a Lightning state dictionary."""
    prefix = f"{model_name}.backbone."
    exported = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if stripped == "mask_token" or stripped.startswith(("dino_head.", "ibot_head.")):
            continue
        exported[stripped] = value
    if not exported:
        raise RuntimeError(f"No {model_name} backbone tensors were available for export")
    return exported
