# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train StyleGAN-XL model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bg_pl_model import BigdatasetganPlModel


def get_latest_tlt_model(results_dir):
    """Utility function to return the latest tlt model in a dir."""
    trainable_ckpts = [int(item.split('.')[0].split('_')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".tlt")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = expand_path(os.path.join(results_dir, f"iter_{latest_step}.tlt"))
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}")
    return latest_checkpoint


def run_experiment(experiment_config, key):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes
    strategy = 'auto'

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    # StyleGAN-XL only supports 'stylegan' and 'bigdatasetgan' tasks
    if experiment_config.task == 'stylegan':
        # Default to 1 GPU if none provided
        if len(trainer_kwargs['devices']) > 1:
            # This is necessary or else Lightning will raise an error since not all params are used in training_step
            strategy = 'ddp_find_unused_parameters_true'

        # build dataloader
        dm = SXDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint if provided
        if pretrained_path:
            model = StyleganPlModel.load_from_checkpoint(pretrained_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config,
                                                         dm=dm
                                                         )
        else:
            model = StyleganPlModel(experiment_config, dm)

    elif experiment_config.task == 'bigdatasetgan':

        # build dataloader
        dm = BGDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint if provided
        if pretrained_path:
            model = BigdatasetganPlModel.load_from_checkpoint(pretrained_path,
                                                              map_location="cpu",
                                                              experiment_spec=experiment_config,
                                                              dm=dm
                                                              )
        else:
            model = BigdatasetganPlModel(experiment_config, dm)

    else:
        raise NotImplementedError("Task {} is not implemented".format(experiment_config.task))

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy=strategy,
                      precision='32-true',
                      # use_distributed_sampler=False,
                      sync_batchnorm=False,
                      num_sanity_val_steps=0,
                      )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="StyleGAN-XL", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
