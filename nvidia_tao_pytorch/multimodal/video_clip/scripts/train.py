# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train CLIP model."""

import math
import os
from datetime import timedelta

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import (
    initialize_train_experiment,
)
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.pl_video_clip_model import (
    VideoCLIPPlModel,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.pl_video_clip_data_module import (
    VideoCLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.video_text_loader import (
    _entries_from_cfg,
)

from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    load_partial_pretrained_weights,
    register_checkpoint_safe_globals,
    to_lightning_precision,
)

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


def _device_count(devices):
    """Return a scalar device count from a Trainer devices argument."""
    if not isinstance(devices, (str, bytes)) and hasattr(devices, "__len__"):
        return len(devices)
    return int(devices)


def _set_video_text_train_batches(experiment_config, trainer_kwargs, num_nodes):
    """Set finite train batches so Lightning's progress bar has a total."""
    train_data_cfg = experiment_config.dataset.train
    if getattr(train_data_cfg, "type", None) != "video_text":
        return
    num_samples = len(_entries_from_cfg(train_data_cfg.video_text))
    world_size = max(
        1,
        _device_count(trainer_kwargs["devices"]) * max(1, int(num_nodes)),
    )
    per_rank_samples = (
        math.ceil(num_samples / world_size)
        if world_size > 1
        else num_samples
    )
    batches_per_epoch = per_rank_samples // train_data_cfg.batch_size
    if batches_per_epoch <= 0:
        raise ValueError(
            "Video-text train split is smaller than one full per-rank batch: "
            f"{num_samples} samples, world_size={world_size}, "
            f"batch_size={train_data_cfg.batch_size}."
        )
    trainer_kwargs["limit_train_batches"] = batches_per_epoch
    logging.info(
        "Training video-text schedule: %d samples, %d steps/epoch "
        "(world_size=%d, batch_size=%d)",
        num_samples,
        batches_per_epoch,
        world_size,
        train_data_cfg.batch_size,
    )


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
        pt_model = VideoCLIPPlModel(experiment_config)
        load_partial_pretrained_weights(
            pt_model.model,
            pretrained_path,
            prefixes=("module.", "model."),
            source=pretrained_path,
        )
    else:
        pt_model = VideoCLIPPlModel(experiment_config)
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

    precision = to_lightning_precision(train_cfg.precision, strict=True)

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

    _set_video_text_train_batches(
        experiment_config,
        trainer_kwargs,
        num_nodes,
    )

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=0,
    )
    dm = VideoCLIPDataModule(
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
@monitor_status(name="VideoCLIP", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
