# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File containing generic cookbook for PyTorch models."""

import operator
from dataclasses import dataclass
from eff.core import Archive
from eff.validator.conditions import Expression
from eff.validator.validator import validate_metadata
from nvidia_tao_pytorch.core.cookbooks.cookbook import Cookbook

_TLT_PYTORCH_MODEL = "tlt_pytorch_model.pth"


@dataclass
class TLTPyTorchCheckpointCondition:
    """Condition used to activate the recipes able to save/restore full model (metadata, model_checkpoints, config)"""

    # EFF format version
    format_version: Expression = Expression(operator.eq, 1)
    # Version of the TLT archive
    tlt_archive_version: Expression = Expression(operator.eq, 1)
    # Runtime that can execute the model stored in the archive
    runtime: Expression = Expression(operator.eq, 'PyTorch')
    # indicate that if the model_checkpoints can be used to resume training
    resume_training: bool = True


@dataclass
class TLTPyTorchPretrainCondition:
    """Condition used to activate the recipes able to save/restore full model (metadata, model_checkpoints, config)"""

    # EFF format version
    format_version: Expression = Expression(operator.eq, 1)
    # Version of the TLT archive
    tlt_archive_version: Expression = Expression(operator.eq, 1)
    # Runtime that can execute the model stored in the archive
    runtime: Expression = Expression(operator.eq, 'PyTorch')
    # indicate that if the model_checkpoints can be used to resume training
    resume_training: bool = False


class TLTPyTorchCookbook(Cookbook):
    """ Class providing recipes for storing/restoring TLT-PyTorch models. """

    def save_checkpoint_to(self, ckpt, save_path: str, force: bool = True):
        """
        Saves model instance (weights and configuration) into an EFF archive.

        Method creates an EFF-based file that is an archive (tar.gz) with the following:
            manifest.yaml - yaml file describing the content of the archive.
            model_ckpt.pth - model checkpoint

        ..note:
            For NVIDIA TLT the EFF archives will use .tlt postfix.

        Args:
            save_path: Path to archive file where model instance should be saved.
            force: Setting to True enables to overwrite the existing properties/files with the ones coming from
                class (DEFAULT:True)
        """
        # Create EFF archive. Set some standard fields, most importantly:
        #  * obj_cls - fully classified class name (with modules).
        with Archive.create(
            save_path=save_path,
            encryption_key=TLTPyTorchCookbook.get_passphrase(),
            origin='TLT',
            runtime='PyTorch',
            tlt_archive_version=1,
            resume_training=True,
        ) as effa:

            # Add additional metadata stored by the TLT PyTorch class.
            effa.add_metadata(force=force, **self.class_metadata)  # pylint: disable=not-a-mapping

            # Add model weights to archive - encrypt when the encryption key is provided.
            model_ckpt_file = effa.create_file_handle(
                name=_TLT_PYTORCH_MODEL,
                description="File containing model weights and states to resume training",
                encrypted=(TLTPyTorchCookbook.get_passphrase() is not None),
            )
            import torch
            # Save models state using torch save.
            torch.save(ckpt, model_ckpt_file)
            if not validate_metadata(TLTPyTorchCheckpointCondition, metadata=effa.metadata):
                raise TypeError("Archive doesn't have the required  format, version or object class type")

    def save_pretrain_to(self, model_state_dict, save_path: str, force: bool = True):
        """
        Saves model instance (weights and configuration) into an EFF archive.

        Method creates an EFF-based file that is an archive (tar.gz) with the following:
            manifest.yaml - yaml file describing the content of the archive.
            model_weights.pth - model weights only

        ..note:
            For NVIDIA TLT the EFF archives will use .tlt postfix.

        Args:
            save_path: Path to archive file where model instance should be saved.
            force: Setting to True enables to overwrite the existing properties/files with the ones coming from
                class (DEFAULT:True)
        """
        # Create EFF archive. Set some standard fields, most importantly:
        #  * obj_cls - fully classified class name (with modules).
        with Archive.create(
            save_path=save_path,
            encryption_key=TLTPyTorchCookbook.get_passphrase(),
            origin='TLT',
            runtime='PyTorch',
            tlt_archive_version=1,
            resume_training=False,
        ) as effa:

            # Add additional metadata stored by the TLT PyTorch class.
            effa.add_metadata(force=force, **self.class_metadata)  # pylint: disable=not-a-mapping

            # Add model weights to archive - encrypt when the encryption key is provided.
            model_file = effa.create_file_handle(
                name=_TLT_PYTORCH_MODEL,
                description="File containing only model weights",
                encrypted=(TLTPyTorchCookbook.get_passphrase() is not None),
            )
            import torch

            # Save models state using torch save.
            torch.save(model_state_dict, model_file)
            if not validate_metadata(TLTPyTorchPretrainCondition, metadata=effa.metadata):
                raise TypeError("Archive doesn't have the required  format, version or object class type")

    def restore_from_ckpt(
        self,
        restore_path: str,
    ):
        """
        Restores model checkpoint from EFF Archive.

        ..note:
            For NVIDIA TLT the EFF archives will use .tlt postfix.

        Args:
            restore_path: path to file from which model should be instantiated

        Returns:
            model checkpoint
        """
        # Restore the archive.
        with Archive.restore_from(
            restore_path=restore_path, passphrase=TLTPyTorchCookbook.get_passphrase()
        ) as restored_effa:

            # Validate the indicated archive using the conditions associated with this recipe.
            if not validate_metadata(TLTPyTorchCheckpointCondition, metadata=restored_effa.metadata):
                raise TypeError("Archive doesn't have the required runtime, format, version or resume training status")

            # Restore the model checkpoint.
            import torch

            model_ckpt_file, _ = restored_effa.retrieve_file_handle(name=_TLT_PYTORCH_MODEL)
            model_ckpt = torch.load(model_ckpt_file)

        return model_ckpt

    def restore_from_pretrain(
        self,
        restore_path: str,
    ):
        """
        Restores model pretrain from EFF Archive.

        ..note:
            For NVIDIA TLT the EFF archives will use .tlt postfix.

        Args:
            restore_path: path to file from which model should be instantiated

        Returns:
            model checkpoint
        """
        # Restore the archive.
        with Archive.restore_from(
            restore_path=restore_path, passphrase=TLTPyTorchCookbook.get_passphrase()
        ) as restored_effa:
            # Validate the indicated archive using the conditions associated with this recipe.
            if not validate_metadata(TLTPyTorchPretrainCondition, metadata=restored_effa.metadata):
                raise TypeError("Archive doesn't have the required runtime, format, version or resume training status")
            # Restore the model checkpoint.
            import torch
            model_state_dict_file, _ = restored_effa.retrieve_file_handle(name=_TLT_PYTORCH_MODEL)
            model_state_dict = torch.load(model_state_dict_file)

        return model_state_dict
