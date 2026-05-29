# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distiller core for TAO Toolkit models."""
from abc import abstractmethod
from typing import Any, Dict

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule


class Distiller(TAOLightningModule):
    """PTL Module for Distillation."""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        super().__init__(experiment_spec)

        self.distillation_config = experiment_spec.distill
        self.eval_class_ids = self.dataset_config.get("eval_class_ids", None)
        self.dataset_type = self.dataset_config.get("dataset_type", None)

        self.status_logging_dict = {}

        self._build_model(export)
        self._build_criterion()

    @abstractmethod
    def _build_model(self, export):
        """Internal function to build the model."""
        # Should instantiate student and teacher model from experiment_spec
        raise NotImplementedError('Subclasses must implement _build_model')

    @abstractmethod
    def _build_criterion(self):
        """Internal function to build the loss function."""
        raise NotImplementedError('Subclasses must implement _build_criterion')

    @abstractmethod
    def configure_optimizers(self):
        """Internal function to build the optimizer."""
        self.train_config = self.experiment_spec.train

    @abstractmethod
    def on_train_epoch_start(self):
        """Train epoch start. Declaring output list."""
        # raise NotImplementedError('Subclasses must implement on_train_epoch_start')
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Internal function to define the training step."""
        raise NotImplementedError('Subclasses must implement training_step')

    @abstractmethod
    def on_train_epoch_end(self):
        """Internal function to define the training epoch end."""
        raise NotImplementedError('Subclasses must implement training_epoch_end')

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Internal function to define the validation step."""
        raise NotImplementedError('Subclasses must implement validation_step')

    @abstractmethod
    def on_validation_epoch_end(self):
        """Internal function to define the validation epoch end."""
        raise NotImplementedError('Subclasses must implement validation_epoch_end')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Encrpyt the checkpoint. The encryption is done in TLTCheckpointConnector."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Decrpyt the checkpoint."""
        pass
