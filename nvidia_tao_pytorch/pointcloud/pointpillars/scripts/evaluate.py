# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation script for PointPillars."""
import datetime
import os
from pathlib import Path

import torch
import torch.multiprocessing
import pycuda
import pycuda.autoinit

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.eval_utils import eval_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils


pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


def parse_epoch_num(model_file):
    """Parse epoch number from model file."""
    model_base = os.path.basename(model_file)
    epoch_string = model_base[:-4].split("_")[-1]
    return int(epoch_string)


def eval_single_ckpt(
    model, test_loader, cfg,
    eval_output_dir, logger, epoch_id,
    dist_test=False
):
    """Evaluation with PyTorch model."""
    model.cuda()
    # start evaluation
    return eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=cfg.evaluate.save_to_file
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
    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # Set status logging
    status_file = os.path.join(str(eval_output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting PointPillars evaluation"
    )
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        batch_size=cfg.evaluate.batch_size,
        dist=False,
        workers=cfg.dataset.num_workers,
        logger=logger,
        training=False,
        info_path=cfg.dataset.data_info_path
    )[1]
    checkpoint_loaded = load_checkpoint(
        cfg.evaluate.checkpoint,
        cfg.key
    )
    model = checkpoint_loaded[0]
    epoch_num = checkpoint_loaded[2]
    # Load model from checkpoint
    with torch.no_grad():
        ret_dict = eval_single_ckpt(
            model, test_loader, cfg, eval_output_dir,
            logger, epoch_num, dist_test=False
        )
    status_logging.get_status_logger().kpi = ret_dict
    status_logging.get_status_logger().write(
        message="Evaluation metrics generated.",
        status_level=status_logging.Status.RUNNING
    )


if __name__ == '__main__':
    pyc_ctx.push()
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
    pyc_ctx.pop()
