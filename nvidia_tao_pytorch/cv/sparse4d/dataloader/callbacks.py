# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightning callbacks for Sparse4D training."""

from pytorch_lightning import Callback

from nvidia_tao_pytorch.core.tlt_logging import logging


class PklResampleCallback(Callback):
    """Re-sample pkl files at every epoch boundary.

    Calls ``dataset.resample_pkls(epoch)`` every *num_iters_per_epoch*
    global steps, then refreshes the batch sampler so it picks up the
    new flags/groups/scenes.  DataLoader workers are restarted
    automatically by Lightning at epoch boundaries.

    Args:
        num_iters_per_epoch: Trigger interval in global steps.
    """

    def __init__(self, num_iters_per_epoch):
        super().__init__()
        self.num_iters_per_epoch = num_iters_per_epoch

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Check if we should resample at this step."""
        global_step = trainer.global_step
        if global_step == 0 or global_step % self.num_iters_per_epoch != 0:
            return

        current_epoch = global_step // self.num_iters_per_epoch

        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        dataset = train_dataloader.dataset
        if not hasattr(dataset, "resample_pkls"):
            return

        logging.info(
            f"[PklResampleCallback] Resampling pkl files for epoch {current_epoch} "
            f"at step {global_step}"
        )
        dataset.resample_pkls(current_epoch)

        batch_sampler = train_dataloader.batch_sampler
        if hasattr(batch_sampler, "update_from_dataset"):
            batch_sampler.update_from_dataset()
