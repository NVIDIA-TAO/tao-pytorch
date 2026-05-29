# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pruning script for PointPillars."""
import datetime
import os
from pathlib import Path
import tempfile
import torch
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
import nvidia_tao_pytorch.pruning.torch_pruning_v0 as tp
from nvidia_tao_pytorch.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import (
    load_checkpoint,
    load_data_to_gpu
)
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.train_utils import (
    encrypt_pytorch
)


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cfgs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="pointpillar_general", schema=ExperimentConfig
)
def prune_model(cfg: ExperimentConfig) -> None:
    """Prune the PointPillars model."""
    threshold = cfg.prune.pruning_thresh
    if cfg.results_dir is None:
        raise OSError("Either provide output_dir in config file or provide output_dir as a CLI argument")
    cfg.results_dir = expand_path(cfg.results_dir)
    output_dir = Path(cfg.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.local_rank)
    # Set status logging
    status_file = os.path.join(str(output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars Pruning")
    # -----------------------create dataloader & network & optimizer---------------------------
    train_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        batch_size=1,
        dist=False,
        workers=cfg.dataset.num_workers,
        logger=logger,
        training=True,
        info_path=cfg.dataset.data_info_path,
        merge_all_iters_to_one_epoch=False,
        total_epochs=cfg.train.num_epochs
    )[1]
    input_dict = next(iter(train_loader))
    load_data_to_gpu(input_dict)
    if cfg.prune.model is None:
        raise OSError("Please provide prune.model in config file")
    if not os.path.exists(expand_path(cfg.prune.model)):
        raise OSError(f"Model not found: {cfg.prune.model}")
    model = load_checkpoint(cfg.prune.model, cfg.key)[0]
    model = model.cuda()
    model = model.eval()
    unpruned_total_params = sum(p.numel() for p in model.parameters())
    strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=input_dict)
    # conv layers
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    # Exclude heads
    black_list = layers[-3:]
    count = 0
    for layer in layers:
        if layer in black_list:
            continue
        # can run some algo here to generate threshold for every node
        threshold_run = threshold
        pruning_idxs = strategy(layer.weight, amount=threshold_run)
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruning_idxs)
        if pruning_plan is not None:
            pruning_plan.exec()
        else:
            continue
        count += 1
    pruned_total_params = sum(p.numel() for p in model.parameters())
    print("Pruning ratio: {}".format(
        pruned_total_params / unpruned_total_params)
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message="Pruning ratio: {}".format(pruned_total_params / unpruned_total_params)
    )
    save_path = expand_path(f"{cfg.results_dir}/pruned_{threshold}.tlt")
    handle, temp_file = tempfile.mkstemp()
    os.close(handle)
    torch.save(model, temp_file)
    encrypt_pytorch(temp_file, save_path, cfg.key)
    print(f"Pruned model saved to {save_path}")
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message=f"Pruned model saved to {save_path}"
    )
    return model


if __name__ == "__main__":
    try:
        prune_model()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Pruning finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Pruning was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
