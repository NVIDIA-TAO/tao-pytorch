# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common Training Flow"""

import os
# from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.backends.cudnn as cudnn

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.mlops import check_wandb_logged_in, initialize_wandb
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint


def initialize_train_experiment(cfg, key=None):
    """Common training steps for all models"""
    TLTPyTorchCookbook.set_passphrase(key)

    results_dir = cfg["results_dir"]
    loggers = [TensorBoardLogger(save_dir=results_dir, version=1, name="lightning_logs")]

    total_epochs = cfg["train"]["num_epochs"]
    validation_interval = cfg["train"]["validation_interval"]
    checkpoint_interval = cfg["train"]["checkpoint_interval"]
    checkpoint_interval_unit = cfg["train"].get("checkpoint_interval_unit", "epoch")

    if checkpoint_interval_unit == "epoch":
        assert checkpoint_interval <= total_epochs, (
            f"Checkpoint interval {checkpoint_interval} > Number of epochs {total_epochs}."
            f"Please set experiment_config.train.checkpoint_interval <= {total_epochs}"
        )
    assert validation_interval <= total_epochs, (
        f"Validation interval {validation_interval} > Number of epochs {total_epochs}."
        f"Please set experiment_config.train.validation_interval <= {total_epochs}"
    )

    # If seed is set -1, disable setting fixed seed.
    if cfg["train"]["seed"] >= 0:
        seed_everything(cfg["train"]["seed"], workers=True)

    deterministic = cfg["train"]["cudnn"]["deterministic"]
    cudnn.benchmark = cfg["train"]["cudnn"]["benchmark"]
    cudnn.deterministic = deterministic
    if deterministic:
        # cuBLAS needs a workspace config selected before the first CUDA context
        # for its GEMM kernels to be deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True: ops without a deterministic CUDA kernel (e.g. bilinear
        # F.interpolate backward, used by several TAO models) will warn instead
        # of raising, so existing training runs keep working.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # scaled_dot_product_attention's flash / mem-efficient backends have
        # non-deterministic backward kernels (only the math backend is
        # deterministic). Force the math backend so attention-heavy backbones
        # (e.g. C-RADIO ViT) are reproducible. Slower / more memory, but only
        # incurred when determinism is requested.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    resume_ckpt = cfg["train"]["resume_training_checkpoint_path"] or get_latest_checkpoint(results_dir)
    if resume_ckpt:
        if resume_ckpt.endswith('.tlt') or resume_ckpt.endswith('.pth'):
            logging.info(f"Setting resume checkpoint to {resume_ckpt}")
            status_logging.get_status_logger().write(
                message=f"Resuming training from checkpoint: {resume_ckpt}",
                status_level=status_logging.Status.RUNNING
            )
        else:
            raise ValueError("Resume checkpoint has an invalid file format; it must be either a .tlt or a .pth")

    # This env var is set in the common entrypoint to be consistent with num_gpus and gpu_ids
    gpus = [int(gpu) for gpu in os.environ['TAO_VISIBLE_DEVICES'].split(',')]
    cfg["train"]["num_gpus"] = len(gpus)
    cfg["train"]["gpu_ids"] = gpus

    # Get wandb logger for train.
    if hasattr(cfg, "wandb"):
        wandb_config = cfg.wandb
        wandb_logged_in = check_wandb_logged_in()
        if wandb_logged_in and wandb_config.enable:
            wandb_logger = initialize_wandb(
                project=wandb_config.project,
                entity=wandb_config.entity,
                name=wandb_config.name,
                run_id=wandb_config.run_id,
                results_dir=results_dir,
                wandb_logged_in=wandb_logged_in,
                tags=wandb_config.tags,
                config=cfg,
                group=wandb_config.group
            )
            loggers.append(wandb_logger)

    trainer_kwargs = {'logger': loggers,
                      'devices': gpus,
                      'max_epochs': total_epochs,
                      'check_val_every_n_epoch': validation_interval,
                      'default_root_dir': results_dir,
                      'accelerator': 'gpu',
                      # This is false since we define our own ModelCheckpoint callbacks
                      'enable_checkpointing': False,
                      'deterministic': "warn" if deterministic else False,
                      }

    return resume_ckpt, trainer_kwargs


def initialize_evaluation_experiment(cfg, key=None):
    """Common evaluation steps for all models"""
    TLTPyTorchCookbook.set_passphrase(key)

    results_dir = cfg["results_dir"]

    # TODO @seanf: model checkpoint checking
    # Perhaps we should do something here where, if this isn't provided, we look at latest_checkpoint just like train?

    model_path = cfg["evaluate"]["checkpoint"]

    status_logging.get_status_logger().write(
        message=f"Loading checkpoint: {model_path}",
        status_level=status_logging.Status.RUNNING)

    # This env var is set in the common entrypoint to be consistent with num_gpus and gpu_ids
    gpus = [int(gpu) for gpu in os.environ['TAO_VISIBLE_DEVICES'].split(',')]
    cfg["evaluate"]["num_gpus"] = len(gpus)
    cfg["evaluate"]["gpu_ids"] = gpus

    trainer_kwargs = {'devices': gpus,
                      'default_root_dir': results_dir,
                      'accelerator': 'gpu',
                      'strategy': 'auto'
                      }

    return model_path, trainer_kwargs


def initialize_inference_experiment(cfg, key=None):
    """Common inference steps for all models"""
    TLTPyTorchCookbook.set_passphrase(key)

    results_dir = cfg["results_dir"]

    # TODO @seanf: model checkpoint checking
    # Perhaps we should do something here where, if this isn't provided, we look at latest_checkpoint just like train?

    model_path = cfg["inference"]["checkpoint"]

    status_logging.get_status_logger().write(
        message=f"Loading checkpoint: {model_path}",
        status_level=status_logging.Status.RUNNING
    )

    # This env var is set in the common entrypoint to be consistent with num_gpus and gpu_ids
    gpus = [int(gpu) for gpu in os.environ['TAO_VISIBLE_DEVICES'].split(',')]
    cfg["inference"]["num_gpus"] = len(gpus)
    cfg["inference"]["gpu_ids"] = gpus

    trainer_kwargs = {'devices': gpus,
                      'default_root_dir': results_dir,
                      'accelerator': 'gpu',
                      'strategy': 'auto'
                      }

    return model_path, trainer_kwargs
