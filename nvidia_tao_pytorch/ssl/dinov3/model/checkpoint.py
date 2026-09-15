# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-only checkpoint publication and retention."""

import os
from pathlib import Path
import re

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

    def __init__(self, *args, save_final_epoch=False, **kwargs):
        """Optionally publish the final manifest-training extent off cadence."""
        super().__init__(*args, **kwargs)
        self.save_final_epoch = save_final_epoch

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
        state_dict = trainer.lightning_module.state_dict()

        if not trainer.is_global_zero:
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = filepath
            return

        if trainer.lightning_module.model_config.distill.enable:
            student_state_dict = {}
            student_ema_state_dict = {}
            for k, v in list(state_dict.items()):
                k_save = k
                if "student.backbone." in k:
                    k_save = k.replace("student.backbone.", "")
                elif "student_ema.backbone." in k:
                    k_save = k.replace("student_ema.backbone.", "")
                else:
                    continue

                if re.match(r"dino_head\.", k_save):
                    continue
                if re.match(r"mask_token", k_save):
                    continue

                if "student.backbone." in k:
                    student_state_dict[k_save] = v
                elif "student_ema.backbone." in k:
                    student_ema_state_dict[k_save] = v

            _atomic_torch_save(student_state_dict, os.path.join(trainer.default_root_dir, f'student_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))
            _atomic_torch_save(student_ema_state_dict, os.path.join(trainer.default_root_dir, f'student_ema_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))

        else:
            student_state_dict = {}
            teacher_state_dict = {}
            for k, v in list(state_dict.items()):
                k_save = k
                if "student.backbone." in k:
                    k_save = k.replace("student.backbone.", "")
                elif "teacher.backbone." in k:
                    k_save = k.replace("teacher.backbone.", "")
                else:
                    continue

                if re.match(r"dino_head\.", k_save):
                    continue
                if re.match(r"mask_token", k_save):
                    continue

                if "student.backbone." in k:
                    student_state_dict[k_save] = v
                elif "teacher.backbone." in k:
                    teacher_state_dict[k_save] = v

            _atomic_torch_save(student_state_dict, os.path.join(trainer.default_root_dir, f'student_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))
            _atomic_torch_save(teacher_state_dict, os.path.join(trainer.default_root_dir, f'teacher_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        prune_periodic_ssl_checkpoints(trainer.default_root_dir)

        # Notify loggers
        for logger in trainer.loggers:
            logger.after_save_checkpoint(self)


class DinoV3ExceptionCheckpoint(TAOExceptionCheckpoint):
    """Use DINOv3 checkpoint names without mutating the shared callback class."""

    FILE_EXTENSION = ".pth"
    CHECKPOINT_NAME_LAST = "dinov3_model_latest"
