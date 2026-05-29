# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MAL training script."""
import logging
import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.mal.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.mal.models.mal import MAL
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config
warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="train", schema=ExperimentConfig
)
@monitor_status(name="MAL", mode="train")
def run_experiment(cfg: ExperimentConfig) -> None:
    """Run training experiment."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(cfg)
    cfg = update_config(cfg, 'train')

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        # This is necessary or else Lightning will raise an error since not all params are used in training_step
        strategy = 'ddp_find_unused_parameters_true'

    cfg.train.lr = cfg.train.lr * len(trainer_kwargs['devices']) * cfg.train.batch_size
    cfg.train.min_lr = cfg.train.lr * cfg.train.min_lr_rate
    num_workers = len(trainer_kwargs['devices']) * cfg.dataset.num_workers_per_gpu

    logger.info("Setting up dataloader...")
    dm = WSISDataModule(
        num_workers=num_workers,
        cfg=cfg)
    dm.setup(stage='fit')

    num_iter_per_epoch = len(dm.train_dataloader())

    if resume_ckpt:
        cfg.train.pretrained_model_path = None

    logger.info("Building MAL models...")
    model = MAL(
        cfg=cfg, num_iter_per_epoch=num_iter_per_epoch,
        categories=dm.train_dataset.coco.dataset['categories'])

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=cfg.train.num_nodes,
        strategy=strategy,
        precision='16-mixed' if cfg.train.use_amp else '32-true',
        accumulate_grad_batches=cfg.train.accum_grad_batches)

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


if __name__ == '__main__':
    run_experiment()
