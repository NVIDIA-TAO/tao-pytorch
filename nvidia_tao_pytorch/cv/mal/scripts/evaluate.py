# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MAL evaluation script."""

import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.mal.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule
from nvidia_tao_pytorch.cv.mal.models.mal import MAL
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config
warnings.filterwarnings("ignore")
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
@monitor_status(name="MAL", mode="evaluate")
def run_evaluation(cfg: ExperimentConfig) -> None:
    """Run evaluation."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(cfg)
    cfg = update_config(cfg, 'evaluate')

    cfg.train.lr = 0
    cfg.train.min_lr = 0
    cfg.train.batch_size = cfg.evaluate.batch_size
    num_workers = len(cfg.evaluate.gpu_ids) * cfg.dataset.num_workers_per_gpu

    dm = WSISDataModule(
        num_workers=num_workers,
        cfg=cfg)
    dm.setup(stage='test')

    model = MAL.load_from_checkpoint(model_path,
                                     map_location='cpu',
                                     cfg=cfg,
                                     num_iter_per_epoch=1,
                                     categories=dm.val_dataset.coco.dataset['categories'])

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=cfg.evaluate.num_nodes,
        max_epochs=-1,
        precision='16-mixed',
        check_val_every_n_epoch=1,
        accumulate_grad_batches=1)

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    run_evaluation()
