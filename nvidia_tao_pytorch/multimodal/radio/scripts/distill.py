# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-teacher distillation for RADIO."""

import os

from pytorch_lightning import LightningModule, Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.multimodal.radio.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.multimodal.radio.dataloader.radio_data_module import RadioDataModule
from nvidia_tao_pytorch.multimodal.radio.distillation.distiller import MultiTeacherDistiller


def run_experiment(experiment_config, key):
    """Start the distillation training."""
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

    dm = RadioDataModule(experiment_config.dataset, experiment_config=experiment_config)
    dm.setup(stage="fit")

    # Resuming without needing to save teacher weights
    LightningModule.strict_loading = False
    model = MultiTeacherDistiller(experiment_config)

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        strategy = 'ddp_find_unused_parameters_true'

    parity_batches = os.environ.get("PARITY_DUMP_BATCHES")
    parity_overrides = {}
    if parity_batches:
        parity_overrides["limit_train_batches"] = int(parity_batches)
        parity_overrides["limit_val_batches"] = 0
        parity_overrides["num_sanity_val_steps"] = 0
        trainer_kwargs["max_epochs"] = 1

    trainer = Trainer(
        **trainer_kwargs,
        gradient_clip_val=clip_grad_norm,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=True,
        **parity_overrides,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="distill",
    schema=ExperimentConfig,
)
@monitor_status(name="Class_pt", mode="distill")
def main(cfg: ExperimentConfig) -> None:
    """Run the distillation process."""
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
