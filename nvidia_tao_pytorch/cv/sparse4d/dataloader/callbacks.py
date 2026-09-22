# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightning callbacks for Sparse4D training."""

from pytorch_lightning import Callback

from nvidia_tao_pytorch.core.tlt_logging import logging


class PklResampleCallback(Callback):
    """Re-sample pkl files at every epoch boundary.

    Resampling happens after an epoch has consumed its final batch, so the next
    epoch cannot prefetch from stale worker-side dataset copies. The training
    entrypoint reloads the DataLoader each epoch while this callback is active.

    Args:
        num_iters_per_epoch: Historical step-count argument retained for
            compatibility and diagnostics.
    """

    def __init__(self, num_iters_per_epoch):
        super().__init__()
        self.num_iters_per_epoch = num_iters_per_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """Resample before Lightning constructs the next epoch's DataLoader."""
        current_epoch = int(trainer.current_epoch) + 1
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        dataset = train_dataloader.dataset
        if not hasattr(dataset, "resample_pkls"):
            return

        logging.info(
            f"[PklResampleCallback] Resampling pkl files for epoch {current_epoch} "
            f"after step {trainer.global_step}"
        )
        dataset.resample_pkls(current_epoch)

        batch_sampler = train_dataloader.batch_sampler
        if hasattr(batch_sampler, "update_from_dataset"):
            batch_sampler.update_from_dataset()

        # Persistent-worker DataLoaders cache their iterator on the loader.
        # Explicitly retire it once; Lightning will build a fresh loader/iterator
        # for the next epoch.
        # Private PyTorch API: verified against TAO's 2.11.0a0 nv26.03 runtime.
        # Recheck this retirement path when upgrading the container/PyTorch.
        iterator = getattr(train_dataloader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        if hasattr(train_dataloader, "_iterator"):
            train_dataloader._iterator = None
