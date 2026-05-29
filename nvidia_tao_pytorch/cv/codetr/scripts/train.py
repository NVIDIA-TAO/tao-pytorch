# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train CoDETR model."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.experimental import experimental
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.codetr.model.utils import codetr_parser, ptm_adapter


@experimental("CoDETR training is not extensively tested.")
def run_experiment(experiment_config, key, lightning_module=CoDETRPlModel):
    """Execute CoDETR training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    dm = ODDataModule(experiment_config.dataset)
    dm.setup(stage="fit")

    # Disable activation checkpointing for smaller models with multi-GPU DDP
    if (experiment_config.train.activation_checkpoint and
            len(experiment_config.model.return_interm_indices) < 4 and
            experiment_config.train.num_gpus > 1):
        experiment_config.train.activation_checkpoint = False
        logging.info("Disabled activation checkpointing (smaller model + multi-GPU)")

    activation_checkpoint = experiment_config.train.activation_checkpoint
    pretrained_path = experiment_config.train.pretrained_model_path

    if pretrained_path:
        experiment_config.model.pretrained_backbone_path = None
        pt_model = lightning_module(experiment_config)
        current_sd = pt_model.model.state_dict()
        checkpoint = load_pretrained_weights(pretrained_path,
                                             parser=codetr_parser,
                                             ptm_adapter=ptm_adapter)
        new_sd = {}
        for k in sorted(current_sd.keys()):
            v = checkpoint.get(k, None)
            if v is not None and v.size() == current_sd[k].size():
                new_sd[k] = v
            else:
                if v is not None:
                    logging.warning("Skip layer %s: size mismatch %s vs %s",
                                    k, list(v.size()), list(current_sd[k].size()))
                else:
                    logging.warning("Skip layer %s: not in checkpoint", k)
                new_sd[k] = current_sd[k]
        m = pt_model.model.load_state_dict(new_sd, strict=False)
        logging.info("Loaded pretrained weights from %s\n%s", pretrained_path, m)
    else:
        pt_model = lightning_module(experiment_config)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    precision = '16-mixed' if experiment_config.train.precision.lower() == 'fp16' else '32-true'

    sync_batchnorm = False
    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp":
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"Strategy {distributed_strategy} not supported")

        if "fan" in experiment_config.model.backbone:
            sync_batchnorm = True

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_norm,
                      use_distributed_sampler=False,
                      sync_batchnorm=sync_batchnorm,
                      fast_dev_run=is_dry_run)

    if resume_ckpt and resume_ckpt.endswith('.tlt'):
        trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
@monitor_status(name="CoDETR", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run CoDETR training."""
    run_experiment(experiment_config=cfg, key=cfg.encryption_key,
                   lightning_module=CoDETRPlModel)


if __name__ == "__main__":
    main()
