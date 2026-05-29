# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAL inference script."""

import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.mal.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule
from nvidia_tao_pytorch.cv.mal.models.mal import MALPseudoLabels
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config
warnings.filterwarnings("ignore")
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name="MAL", mode="inference")
def run_inference(cfg: ExperimentConfig) -> None:
    """Run pseudo-label generation."""
    model_path, trainer_kwargs = initialize_inference_experiment(cfg)
    cfg = update_config(cfg, 'inference')

    cfg.train.lr = 0
    cfg.train.min_lr = 0

    num_workers = len(cfg.inference.gpu_ids) * cfg.dataset.num_workers_per_gpu
    # override data path and batch_size
    cfg.dataset.val_ann_path = cfg.inference.ann_path
    cfg.dataset.val_img_dir = cfg.inference.img_dir
    cfg.dataset.load_mask = cfg.inference.load_mask
    cfg.train.batch_size = cfg.inference.batch_size
    cfg.evaluate.use_mixed_model_test = False
    cfg.evaluate.use_teacher_test = False
    cfg.evaluate.comp_clustering = False
    cfg.evaluate.use_flip_test = False

    dm = WSISDataModule(
        num_workers=num_workers,
        cfg=cfg)
    dm.setup(stage='predict')

    # Phase 2: Generating pseudo-labels
    model = MALPseudoLabels.load_from_checkpoint(model_path,
                                                 map_location='cpu',
                                                 cfg=cfg,
                                                 categories=dm.val_dataset.coco.dataset['categories'])

    trainer = Trainer(
        **trainer_kwargs,
        precision='16-mixed',
        check_val_every_n_epoch=1
    )
    trainer.predict(model, datamodule=dm)


if __name__ == '__main__':
    run_inference()
