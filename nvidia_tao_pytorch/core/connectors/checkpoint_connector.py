# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File containing Overloaded version of PTL CheckpointCollector."""

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.trainer.connectors.checkpoint_connector import \
    _CheckpointConnector
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.cloud_io import _load as pl_load

from nvidia_tao_pytorch.core.checkpoint_encryption import decrypt_checkpoint, encrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


class TLTCheckpointConnector(_CheckpointConnector):
    """
    Overloaded version of PTL CheckpointCollector, with additional encryption of intermediate checkpoints.
    """

    def restore(self, checkpoint_path: str, on_gpu: bool) -> bool:
        """
        Load model/training states from a 'PyTorch-Lightning checkpoint' file through file-read and state-restore.
        All restored states are listed in return value description of `dump_checkpoint`.
        """
        # Try to read the checkpoint file at `checkpoint_path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(checkpoint_path)
        if not fs.exists(checkpoint_path):
            rank_zero_warn("No checkpoint file exists at `resume_from_checkpoint`. Start from scratch")
            return False

        # read a checkpoint dictionary object from the 'PyTorch-Lightning checkpoint' file at `checkpoint_path`
        checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        # acquire the model
        model = self.trainer.get_model()

        # restore model and datamodule state
        self.restore_model_state(model, checkpoint)

        if on_gpu:
            model.cuda(self.trainer.root_gpu)

        # restore training state
        self.restore_training_state(checkpoint)

        rank_zero_info(f"Restored states from the checkpoint file at {checkpoint_path}")
        return True

    def restore_model(self) -> None:
        """
        Restores a model's weights from a PyTorch Lightning checkpoint. Hooks are called first go give
        the LightningModule a chance to modify the contents, then finally the model gets updated with
        the loaded weights.
        """
        if not self._loaded_checkpoint:
            return

        model = self.trainer.lightning_module

        # hook: give user access to checkpoint if needed.
        model.on_load_checkpoint(self._loaded_checkpoint)

        # call hpc specific hook
        if self._hpc_resume_path:
            model.on_hpc_load(self._loaded_checkpoint)

        # restore model state_dict
        self.restore_model_state(model, self._loaded_checkpoint)

    def restore_model_state(self, model: LightningModule, checkpoint) -> None:
        """
        Restore model states from a 'PyTorch-Lightning checkpoint' dictionary object
        """
        # restore datamodule states
        # As of the Common Trainer update, this will throw the following error:
        # AttributeError: 'ARDataModule' (or respective DataModule) object has no attribute 'on_load_checkpoint'
        # if self.trainer.datamodule is not None:
        #     self.trainer.datamodule.on_load_checkpoint(checkpoint)

        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = decrypt_checkpoint(checkpoint, key)

        # hook: give user access to checkpoint if needed.
        model.on_load_checkpoint(checkpoint)

        # restore model state_dict
        model.load_state_dict(checkpoint['state_dict'])

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Creating a model checkpoint dictionary object from various component states.

            If encryption key is provided (in NeMoCookbook), encrypts the model checkpoint.

        Args:
            weights_only: saving model weights only
        Return:
            structured dictionary: {
                'epoch':                     training epoch
                'global_step':               training global step
                'pytorch-lightning_version': PyTorch Lightning's version
                'callbacks':                 "callback specific state"[] # if not weights_only
                'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
                'native_amp_scaling_state':  PT amp's state_dict         # if not weights_only and use native amp
                'amp_scaling_state':         Apex's state_dict           # if not weights_only and use apex amp
                'state_dict':                Model's state_dict (e.g. network weights)
                CHECKPOINT_HYPER_PARAMS_NAME:
                CHECKPOINT_HYPER_PARAMS_KEY:
                CHECKPOINT_HYPER_PARAMS_TYPE:
                something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
                LightningDataModule.__class__.__name__: pl DataModule's state
            }
        """
        # Retrieve checkpoint using the connector.
        checkpoint = super().dump_checkpoint(weights_only)

        # Retrieve encryption key from TLTPyTorchCookbook.
        key = TLTPyTorchCookbook.get_passphrase()
        if key is not None:
            checkpoint = encrypt_checkpoint(checkpoint, key)

        return checkpoint
