# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Re-Identification model."""
import math
import os

from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.config.re_identification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.pl_reid_data_module import REIDDataModule
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel

from pytorch_lightning import Trainer


def run_experiment(experiment_config, key):
    """
    Start the training process.

    This function initializes the re-identification model with the provided experiment configuration.
    It sets up the necessary components such as the status logger and checkpoint callbacks.
    The training is performed using the PyTorch Lightning Trainer.

    Args:
        experiment_config (ExperimentConfig): The experiment configuration containing the model, training, and other parameters.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        AssertionError: If checkpoint_interval is greater than num_epochs.
    """
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    dm = REIDDataModule(experiment_config)
    dm.setup('fit')
    reid_model = ReIdentificationModel(experiment_config, prepare_for_training=True)

    grad_clip = experiment_config['train']['grad_clip']

    # @seanf: there's some buggy behavior with Lightning/Pytorch when using custom samplers
    # Patch fix is to either (a) set the length of the sampler to infinity, but then the progress bar will
    # show ? for the number of epochs (though it will function correctly), or (b) have it do the validation
    # check right before the last batch. We choose the latter
    # See: https://github.com/Lightning-AI/pytorch-lightning/issues/10290
    num_batches = len(dm.train_dataloader())
    # Round down to 2 decimal places
    val_check_interval = math.floor(((num_batches - 1) / num_batches) * 100) / 100

    acc_flag = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        acc_flag = 'ddp_find_unused_parameters_true'

    trainer = Trainer(**trainer_kwargs,
                      num_sanity_val_steps=0,
                      val_check_interval=val_check_interval,
                      precision='16-mixed',
                      strategy=acc_flag,
                      use_distributed_sampler=False,
                      sync_batchnorm=True,
                      gradient_clip_val=grad_clip)

    # Overload connector to enable intermediate ckpt encryption and decryption.
    trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(reid_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ReIdentification", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """
    Run the training process.

    This function serves as the entry point for the training script.
    It loads the experiment specification, obfuscates logs, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.

    Raises:
        KeyboardInterrupt: If the training is interrupted manually.
        SystemExit: If the system or program finishes abruptly.
        Exception: For any other types of exceptions thrown during training.
    """
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
