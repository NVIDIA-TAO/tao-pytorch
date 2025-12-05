# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common Lightning Module"""

from typing import Any, Dict, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint


class TAOLightningModule(pl.LightningModule):
    """
    Common PyTorch Lightning module for TAO (Train, Adapt, Optimize) workflows.

    This class provides a standardized interface for training, validation, testing, and inference
    tasks in the TAO framework. It includes built-in checkpoint management, status logging,
    and data validation to ensure robust training workflows.

    Attributes:
        experiment_spec (dict): Complete experiment configuration including dataset, model,
            training parameters, and results directory.
        dataset_config (dict): Dataset-specific configuration extracted from experiment_spec.
        model_config (dict): Model-specific configuration extracted from experiment_spec.
        checkpoint_filename (str): Base filename for checkpoint files (must be set in subclasses).

    Example:
        >>> class MyModel(TAOLightningModule):
        ...     def __init__(self, experiment_spec):
        ...         super().__init__(experiment_spec)
        ...         self.checkpoint_filename = "my_model"
        ...         # Initialize your model components
    """

    def __init__(self, experiment_spec, **kwargs):
        """
        Initialize the TAO Lightning module.

        Args:
            experiment_spec (dict): Complete experiment configuration dictionary containing:
                - dataset: Dataset configuration
                - model: Model configuration
                - train: Training parameters including checkpoint_interval
                - results_dir: Directory to save results and checkpoints
            **kwargs: Additional keyword arguments passed to the parent LightningModule
        """
        super().__init__(**kwargs)
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec["dataset"]
        self.model_config = experiment_spec["model"]

        self.checkpoint_filename = None

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """
        Configure logging and checkpoint-saving callbacks for the training workflow.

        This method is automatically called by PyTorch Lightning when trainer.fit() is invoked.
        It sets up:
        - Status logging for training progress
        - Regular checkpoint saving based on epoch intervals
        - Exception-based checkpoint saving for error recovery

        Returns:
            Sequence[Callback]: List of configured callbacks including:
                - TAOStatusLogger: Logs training status and progress
                - ModelCheckpoint: Saves model checkpoints at regular intervals
                - TAOExceptionCheckpoint: Saves checkpoints on exceptions

        Raises:
            NotImplementedError: If checkpoint_filename is not set in the subclass __init__ method
        """
        # This is called when trainer.fit() is called

        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval_unit = self.experiment_spec["train"].get("checkpoint_interval_unit", "epoch")
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True
        )

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError("checkpoint_filename not set in __init__() of model")
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = ModelCheckpoint(every_n_epochs=checkpoint_interval if checkpoint_interval_unit == "epoch" else None,
                                              every_n_train_steps=checkpoint_interval if checkpoint_interval_unit == "step" else None,
                                              dirpath=results_dir,
                                              save_on_train_epoch_end=True,
                                              monitor=None,
                                              save_top_k=-1,
                                              save_last='link',
                                              filename='model_{epoch:03d}_{step:05d}',
                                              enable_version_counter=False)

        # For now, we use our custom one since Lightning's callback for this is minimal
        TAOExceptionCheckpoint.FILE_EXTENSION = ModelCheckpoint.FILE_EXTENSION
        TAOExceptionCheckpoint.CHECKPOINT_NAME_LAST = ModelCheckpoint.CHECKPOINT_NAME_LAST
        exception_checkpoint_callback = TAOExceptionCheckpoint(dirpath=results_dir)

        callbacks = [status_logger_callback, checkpoint_callback, exception_checkpoint_callback]

        # top-K ModelCheckpoing(only append when it is set in config)
        checkpointer_cfg = self.experiment_spec["train"].get("checkpointer", None)
        if checkpointer_cfg and checkpointer_cfg.get("enable_topk", False):
            callbacks.append(ModelCheckpoint(
                dirpath=checkpointer_cfg.get("dirpath", "./checkpoints"),
                filename=checkpointer_cfg.get("filename", "topk_epoch{epoch:03d}-m{val_miou:.4f}"),
                monitor=checkpointer_cfg.get("monitor", "val_miou"),
                mode=checkpointer_cfg.get("mode", "max"),
                save_top_k=checkpointer_cfg.get("save_top_k", 3),
                save_last=checkpointer_cfg.get("save_last", True),
                auto_insert_metric_name=checkpointer_cfg.get("auto_insert_metric_name", False),
            ))

        # LearningRateMonitor(only append when it is set in config)
        if self.experiment_spec["train"].get("enable_lr_monitor", False):
            callbacks.append(LearningRateMonitor(logging_interval='step'))

        return callbacks

    def _dataloader_batch_check(self, dataloader, task):
        """
        Validate that the dataset size is sufficient for the specified batch size.

        This method checks if the dataset contains enough samples to fill at least one batch
        during training/validation/testing. This is particularly important when using
        drop_last=True in dataloaders, as Lightning may report completion without actually
        processing data if the batch cannot be filled.

        Args:
            dataloader: PyTorch DataLoader instance to validate
            task (str): Description of the task being performed (e.g., "train", "validation")

        Raises:
            ValueError: If the dataset size is smaller than the total batch size across all devices
            AssertionError: If the dataloader doesn't have a batch_sampler when batch_size is None
        """
        batch_size = dataloader.batch_size
        # Using a BatchSampler
        if not batch_size:
            assert hasattr(dataloader, "batch_sampler"), "Loader should have batch sampler initiated if batch size isn't defined."
            batch_size = dataloader.batch_sampler.batch_size
        dataset_len = len(dataloader.dataset)
        total_batch_size = batch_size * self.trainer.num_devices

        if dataset_len < total_batch_size:
            raise ValueError(f"Dataset size ({dataset_len}) is smaller than the total batch size "
                             f"({total_batch_size}). Not enough data for {task}.")

    def on_fit_start(self):
        """
        Hook called before training begins.

        Performs validation that the training dataloader has sufficient data to begin training.
        This ensures that training doesn't start with insufficient data, which could lead to
        silent failures or misleading completion reports.
        """
        self._dataloader_batch_check(self.trainer.datamodule.train_dataloader(), "train")

    def on_validation_start(self):
        """
        Hook called before validation begins.

        Performs validation that the validation dataloader has sufficient data to begin validation.
        This ensures that validation doesn't start with insufficient data.
        """
        self._dataloader_batch_check(self.trainer.datamodule.val_dataloader(), "validation")

    def on_test_start(self):
        """
        Hook called before testing begins.

        Performs validation that the test dataloader has sufficient data to begin evaluation.
        This ensures that testing doesn't start with insufficient data.
        """
        self._dataloader_batch_check(self.trainer.datamodule.test_dataloader(), "evaluation")

    def on_predict_start(self):
        """
        Hook called before inference/prediction begins.

        Performs validation that the prediction dataloader has sufficient data to begin inference.
        This ensures that inference doesn't start with insufficient data.
        """
        self._dataloader_batch_check(self.trainer.datamodule.predict_dataloader(), "inference")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook called when saving a checkpoint.

        In the TAO framework, checkpoint encryption is handled by TLTCheckpointConnector,
        so this method is intentionally left empty. Subclasses can override this method
        to add custom checkpoint processing logic if needed.

        Args:
            checkpoint (dict): The checkpoint dictionary to be saved
        """
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook called when loading a checkpoint.

        This method handles checkpoint decryption if the checkpoint is encrypted.
        It retrieves the encryption key from TLTPyTorchCookbook and decrypts the
        checkpoint state dict before loading.

        Args:
            checkpoint (dict): The checkpoint dictionary to be loaded

        Raises:
            PermissionError: If the checkpoint is encrypted and the encryption key is not available
        """
        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = patch_decrypt_checkpoint(checkpoint, key)
