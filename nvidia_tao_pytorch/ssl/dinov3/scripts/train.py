# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Train Script.

Thin wrapper that builds :class:`DinoV3PlModel` and reuses the inherited (nvdinov2)
Lightning training flow and data module.
"""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig, validate_img_size
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel


def _resolve_strategy(experiment_config):
    """Pick the Lightning distributed strategy.

    FSDP (needed for high-resolution and the larger ViT-L/H backbones, where DDP's full
    per-GPU replication does not fit) is selected via ``train.distributed_strategy``
    (``auto`` | ``ddp`` | ``fsdp``); the ``DINOV3_STRATEGY`` env var overrides the config when
    set, kept for the de-risking smokes. The inherited ``DinoV2PlModel.configure_model``
    already wraps the student/teacher ModuleDicts as FSDP units and its EMA update is
    FSDP-aware; this just constructs the strategy. Defaults to Lightning ``'auto'``
    (single-GPU / DDP), unchanged.
    """
    choice = os.environ.get(
        "DINOV3_STRATEGY", experiment_config.train.distributed_strategy
    ).lower()
    if choice == "fsdp":
        from torch.distributed.fsdp import ShardingStrategy
        # FULL_SHARD params/grads/optimizer; gather full state on save so the
        # CustomModelCheckpoint (which pulls student/teacher state dicts) works.
        return FSDPStrategy(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            state_dict_type="full",
        )
    return choice


def run_experiment(experiment_config, key):
    """Start the training."""
    # Fail before logger/trainer/data-module initialization. Hydra validates the field type,
    # but the INT_FIELD ``valid_options`` enum is schema metadata rather than a runtime guard.
    validate_img_size(experiment_config.model.backbone)

    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = experiment_config.train.precision

    dm = DinoV2DataModule(experiment_config)

    model = DinoV3PlModel(experiment_config)

    if pretrained_path:
        model.pretrained_weights = pretrained_path
        model.restore_pretrained_weights()

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy=_resolve_strategy(experiment_config),
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
@monitor_status(name="DINOv3", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
