# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Classification model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel


def run_experiment(experiment_config, key, lightning_module=ClassifierPlModel):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(
            f"{experiment_config.train.precision} is not supported. Only bf16, fp16, and fp32 are supported")

    dm = CLDataModule(experiment_config.dataset)
    dm.setup(stage="fit")

    model = lightning_module(experiment_config)

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        strategy = 'ddp_find_unused_parameters_true'

    trainer = Trainer(
        **trainer_kwargs,
        gradient_clip_val=clip_grad_norm,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=True,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Classification", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(
        experiment_config=cfg,
        key=cfg.encryption_key,
        lightning_module=ClassifierPlModel
    )


if __name__ == "__main__":
    main()
