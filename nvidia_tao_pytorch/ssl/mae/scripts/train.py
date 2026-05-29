# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAE training script."""
import logging
import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment

from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.mae.dataloader.pl_data_module import MAEDataModule
from nvidia_tao_pytorch.ssl.mae.model.pl_model import MAEPlModule

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
@monitor_status(name="MAE", mode="train")
def run_experiment(cfg: ExperimentConfig) -> None:
    """Run training experiment."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(cfg)

    if cfg.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif cfg.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif cfg.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{cfg.train.precision} is not supported. \
                                  Only bf16, fp16, and fp32 are supported")

    distributed_strategy = cfg.train.distributed_strategy
    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        # By default find_unused_parameters is set to False in Lightning
        # If true, it introduces extra overhead and can't work with activation checkpointing
        if distributed_strategy.lower() == "ddp":
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            # Override to FP16 for fsdp as there's an error with FP32 during Positional Embedding forward pass
            logging.info("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and fsdp are supported")

    logger.info("Setting up dataloader...")
    data_module = MAEDataModule(cfg=cfg)

    logger.info("Building MAE models...")
    pl_model = MAEPlModule(cfg=cfg)

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=cfg.train.num_nodes,
        strategy=strategy,
        precision=precision,
        num_sanity_val_steps=0 if cfg.train.stage == "pretrain" else 1,
        accumulate_grad_batches=cfg.train.accum_grad_batches)

    trainer.fit(pl_model, data_module, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    run_experiment()
