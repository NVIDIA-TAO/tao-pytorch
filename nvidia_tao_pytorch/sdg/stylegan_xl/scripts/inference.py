# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference with StyleGAN-XL model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bg_pl_model import BigdatasetganPlModel


def run_experiment(experiment_config, key):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)
    num_nodes = experiment_config.train.num_nodes

    # StyleGAN-XL only supports 'stylegan' and 'bigdatasetgan' tasks
    if experiment_config.task == 'stylegan':
        # build dataloader
        dm = SXDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        model = StyleganPlModel.load_from_checkpoint(model_path,
                                                     map_location="cpu",
                                                     experiment_spec=experiment_config,
                                                     dm=dm
                                                     )

    elif experiment_config.task == 'bigdatasetgan':
        # build dataloader
        dm = BGDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        model = BigdatasetganPlModel.load_from_checkpoint(model_path,
                                                          map_location="cpu",
                                                          experiment_spec=experiment_config,
                                                          dm=dm
                                                          )

    else:
        raise NotImplementedError("Task {} is not implemented".format(experiment_config.task))

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      # use_distributed_sampler=False,
                      )

    trainer.predict(model, dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="StyleGAN XL", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
