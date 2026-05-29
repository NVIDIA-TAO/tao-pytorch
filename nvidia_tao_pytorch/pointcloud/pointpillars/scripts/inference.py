# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference script for PointPillars."""
import datetime
import os
from pathlib import Path

import torch
import torch.multiprocessing

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.eval_utils import eval_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.config import log_config_to_file
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils


def parse_epoch_num(model_file):
    """Parse epoch number from model file."""
    model_base = os.path.basename(model_file)
    epoch_string = model_base[:-4].split("_")[-1]
    return int(epoch_string)


def infer_single_ckpt(
    model, test_loader, cfg,
    infer_output_dir, logger
):
    """Do inference with PyTorch model."""
    model.cuda()
    eval_utils.infer_one_epoch(
        cfg, model, test_loader, logger,
        result_dir=infer_output_dir,
        save_to_file=cfg.inference.save_to_file
    )


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cfgs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="pointpillar_general", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Main function."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    if cfg.results_dir is None:
        raise OSError("Either provide output_dir in config file or provide output_dir as a CLI argument")
    output_dir = Path(expand_path(cfg.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_output_dir = output_dir / 'infer'
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = infer_output_dir / ('log_infer_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    # log to file
    logger.info('**********************Start logging**********************')
    # Set status logging
    status_file = os.path.join(str(infer_output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars inference")
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    log_config_to_file(cfg, logger=logger)
    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        batch_size=cfg.inference.batch_size,
        dist=False, workers=cfg.dataset.num_workers,
        logger=logger, training=False,
        info_path=cfg.dataset.data_info_path
    )[1]
    model = load_checkpoint(
        cfg.inference.checkpoint,
        cfg.key
    )[0]
    # Load model from checkpoint
    with torch.no_grad():
        infer_single_ckpt(
            model, test_loader, cfg, infer_output_dir,
            logger
        )


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
