# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 evaluation script.

Variant-agnostic: the panoptic variant reports mAP/mAP50/mAP25, while the
reasoning variant reports mIoU/mAP50/mAP25 on canonical point clouds.
"""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment

from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import PANOPTIC, load_pl_model


def run_experiment(experiment_config):
    """Start the evaluation."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)

    pl_data = build_pl_data_module(experiment_config)
    pl_model = load_pl_model(experiment_config, model_path, strict=False)

    if str(experiment_config.model.model_type) == PANOPTIC:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
            resolve_vocabulary,
        )
        # Category metadata selects the ``isthing`` subset used by instance AP.
        pl_model.set_classes(**resolve_vocabulary(experiment_config.dataset.panoptic))

    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.test(pl_model, datamodule=pl_data)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_panoptic", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3Dv2", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
