# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 training script.

Variant-agnostic: ``model.model_type`` selects which Lightning module and data
module are built. The panoptic module reports validation mAP, mAP50, and mAP25
and uses ``val/mAP`` for optional metric-ranked checkpoints. The reasoning
module reports canonical point-cloud mIoU, mAP50, and mAP25 and ranks its
checkpoints on ``val/mIoU``.
"""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import PANOPTIC, build_pl_model

PRECISION_MAP = {
    "fp32": "32-true",
    "bf16": "bf16-mixed",
    "fp16": "16-mixed",
}


def run_experiment(experiment_config):
    """Start the training."""
    model_type = str(experiment_config.model.model_type)
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config)

    precision = PRECISION_MAP.get(str(experiment_config.train.precision).lower())
    if precision is None:
        raise NotImplementedError(
            f"{experiment_config.train.precision} is not supported. "
            f"Supported precisions: {', '.join(sorted(PRECISION_MAP))}"
        )

    # Build the data module first, then load the native ScanNet++ taxonomy before
    # the panoptic model's first forward pass.
    pl_data = build_pl_data_module(experiment_config)
    pl_model = build_pl_model(experiment_config)

    if model_type == PANOPTIC:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
            resolve_vocabulary,
        )
        # Category metadata selects the ``isthing`` subset used by instance AP.
        pl_model.set_classes(**resolve_vocabulary(experiment_config.dataset.panoptic))

    strategy = "auto"
    if len(trainer_kwargs["devices"]) > 1 or experiment_config.train.num_nodes > 1:
        # Both variants leave frozen branches without gradients: VGGT for the
        # panoptic variant, plus the SAM3 and Qwen base weights for reasoning.
        strategy = DDPStrategy(find_unused_parameters=True)

    # The panoptic data module builds its own DistributedSampler and forwards the
    # epoch to it from ``on_train_epoch_start``; the reasoning data module lets
    # Lightning inject one.
    trainer_kwargs["use_distributed_sampler"] = model_type != PANOPTIC

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=experiment_config.train.num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=experiment_config.train.accum_iter,
        gradient_clip_val=experiment_config.train.clip_grad_norm,
        val_check_interval=experiment_config.train.val_check_interval,
        log_every_n_steps=experiment_config.train.log_interval,
        num_sanity_val_steps=0,
        fast_dev_run=experiment_config.train.is_dry_run,
        limit_val_batches=0 if pl_data.val_dataloader() is None else 1.0,
    )
    trainer.callbacks.append(LearningRateMonitor(logging_interval="step"))

    if resume_ckpt:
        logging.info(f"Resuming training from {resume_ckpt}")
    trainer.fit(pl_model, pl_data, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_panoptic", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3Dv2", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
