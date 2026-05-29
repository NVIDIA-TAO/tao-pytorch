# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training script for PointPillars."""
import datetime
import os
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn as nn
from tensorboardX import SummaryWriter

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import (
    build_model_and_optimizer,
    model_fn_decorator
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.optimization import build_optimizer, build_scheduler
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.train_utils import train_model


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cfgs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="pointpillar_general", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Main function."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    if cfg.train.num_gpus == 1:
        dist_train = False
        total_gpus = 1
        cfg.local_rank = 0
    elif cfg.train.num_gpus > 1:
        total_gpus, cfg.local_rank = common_utils.init_dist_pytorch(
            int(os.environ["LOCAL_RANK"]), backend='nccl'
        )
        dist_train = True
    else:
        raise ValueError(f"Number of GPUs should be >=1, got: {cfg.train.num_gpus}")
    if cfg.train.random_seed is not None:
        common_utils.set_random_seed(cfg.train.random_seed)
    if cfg.results_dir is None:
        raise OSError("Either provide output_dir in config file or provide output_dir as a CLI argument")
    output_dir = Path(expand_path(cfg.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.local_rank)
    # Set up status logger
    status_file = os.path.join(str(output_dir), "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting PointPillars training"
    )
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * cfg.train.batch_size))
    if cfg.local_rank == 0:
        os.makedirs(expand_path(output_dir), exist_ok=True)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.local_rank == 0 else None
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        batch_size=cfg.train.batch_size,
        dist=dist_train,
        workers=cfg.dataset.num_workers,
        logger=logger,
        info_path=cfg.dataset.data_info_path,
        training=True,
        merge_all_iters_to_one_epoch=cfg.train.merge_all_iters_to_one_epoch,
        total_epochs=cfg.train.num_epochs
    )
    model, opt_state, start_epoch, it = build_model_and_optimizer(
        cfg.model,
        len(cfg.dataset.class_names),
        train_set,
        cfg.train.pruned_model_path,
        cfg.train.resume_training_checkpoint_path,
        cfg.model.pretrained_model_path,
        dist_train,
        logger,
        cfg.key
    )
    optimizer = build_optimizer(model, cfg.train)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    if cfg.model.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    # Reload optimizer states after model moved to cuda() so the optimizer
    # states are also moved to the same device
    optimizer.load_state_dict(optimizer.state_dict())
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank % torch.cuda.device_count()])
    # Build LR scheduler
    last_epoch = -1
    if start_epoch > 0:
        last_epoch = start_epoch + 1
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfg.train.num_epochs,
        last_epoch=last_epoch, optim_cfg=cfg.train
    )
    logger.info('**********************Start training**********************')
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.train,
        start_epoch=start_epoch,
        total_epochs=cfg.train.num_epochs,
        start_iter=it,
        rank=cfg.local_rank,
        tb_log=tb_log,
        status_logging=status_logging,
        ckpt_save_dir=output_dir,
        key=cfg.key,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=cfg.train.checkpoint_interval,
        max_ckpt_save_num=cfg.train.max_checkpoint_save_num,
        merge_all_iters_to_one_epoch=cfg.train.merge_all_iters_to_one_epoch
    )

    logger.info('**********************End training**********************')


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Training finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
