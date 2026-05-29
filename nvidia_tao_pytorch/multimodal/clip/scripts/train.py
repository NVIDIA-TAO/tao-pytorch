# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train CLIP model."""

import os
from datetime import timedelta

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import (
    initialize_train_experiment,
)
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

from nvidia_tao_pytorch.config.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
    CLIPPlModel,
)
from nvidia_tao_pytorch.multimodal.clip.dataloader.pl_clip_data_module import (
    CLIPDataModule,
)

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import (
    load_pretrained_weights,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    register_checkpoint_safe_globals,
)

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


def run_experiment(experiment_config, key):
    """Start the training."""
    register_checkpoint_safe_globals()
    resume_ckpt, trainer_kwargs = initialize_train_experiment(
        experiment_config,
    )
    train_cfg = experiment_config.train
    num_nodes = train_cfg.num_nodes
    _ = train_cfg.num_epochs
    validation_interval = train_cfg.validation_interval
    val_check_interval = train_cfg.val_check_interval

    pretrained_path = train_cfg.pretrained_model_path
    if pretrained_path:
        # experiment_config.model.pretrained_backbone_path = None
        pt_model = CLIPPlModel(experiment_config)
        state_dict = load_pretrained_weights(pretrained_path)
        pt_model.model.load_state_dict(state_dict, strict=True)
        logging.info(f"Model loaded from {pretrained_path}")
    else:
        pt_model = CLIPPlModel(experiment_config)
    logging.info("Model loaded")

    sync_batchnorm = False
    # trainer_kwargs = {}

    # TODO: Move to core/check if all these kwargs moved to core by @seanf
    if val_check_interval:
        trainer_kwargs['val_check_interval'] = val_check_interval
        logging.warning(
            "Both `validation_interval` and `val_check_interval` are defined. "
            "`val_check_interval` takes precedence."
        )
    else:
        trainer_kwargs['check_val_every_n_epoch'] = validation_interval

    _PRECISION_MAP = {
        'fp16': '16-mixed',
        'bf16': 'bf16-mixed',
        'fp32': '32-true',
    }
    prec_key = train_cfg.precision.lower()
    precision = _PRECISION_MAP.get(prec_key)
    if precision is None:
        raise NotImplementedError(
            f"Precision '{prec_key}' is not supported. "
            f"Supported: {list(_PRECISION_MAP)}"
        )

    distributed_strategy = train_cfg.distributed_strategy
    strategy = 'auto'
    grad_ckpt = getattr(train_cfg, 'grad_checkpointing', False)

    nccl_timeout = timedelta(hours=2)

    if len(trainer_kwargs['devices']) > 1:
        ds = distributed_strategy.lower()
        if ds == "ddp" and grad_ckpt:
            strategy = DDPStrategy(
                timeout=nccl_timeout,
                find_unused_parameters=False,
            )
        elif ds == "ddp" and not grad_ckpt:
            strategy = DDPStrategy(
                timeout=nccl_timeout,
                find_unused_parameters=True,
            )
        elif ds == "fsdp":
            strategy = 'fsdp'
            # FP32 causes errors in positional embedding
            logging.info("Overriding precision to FP16 for FSDP")
            precision = '16-mixed'
        else:
            raise NotImplementedError(
                f"{distributed_strategy} is not implemented. "
                "Only ddp and fsdp are supported"
            )

    logging.info(f"Using distributed strategy with {nccl_timeout} timeout")

    clip_norm = getattr(
        experiment_config.train, "grad_clip_norm", None,
    )
    if clip_norm is not None:
        trainer_kwargs['gradient_clip_val'] = clip_norm
        trainer_kwargs['gradient_clip_algorithm'] = "norm"

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=0,
    )
    dm = CLIPDataModule(
        experiment_config.dataset,
        pt_model.tokenizer,
        resume_step=0,
        preprocess=(pt_model.preprocess_train, pt_model.preprocess_val),
        world_size=trainer.world_size,
    )
    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)
    logging.info("Training finished")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name="CLIP", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
