# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 inference script.

Variant-agnostic: the panoptic variant writes per-sample panoptic maps, the
reasoning variant writes per-``[SEG]`` masks and fused point clouds.
"""

import json
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import PANOPTIC, load_pl_model


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)

    pl_data = build_pl_data_module(experiment_config)
    pl_model = load_pl_model(experiment_config, model_path, strict=False)

    if str(experiment_config.model.model_type) == PANOPTIC:
        from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.pl_data_module import (
            resolve_vocabulary,
        )
        vocabulary = resolve_vocabulary(experiment_config.dataset.panoptic)

        # An explicit categories JSON replaces the dataset taxonomy for both the
        # text prompts and the category IDs the predicted segments index into.
        categories_json = experiment_config.inference.categories_json
        if categories_json:
            with open(categories_json, "r", encoding="utf-8") as handle:
                categories = json.load(handle)
            classes = [category["name"] for category in categories]
            vocabulary.update(classes=classes, categories=categories)
            logging.info(f"Predicting against {len(classes)} classes from {categories_json}")
        pl_model.set_classes(**vocabulary)

    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(pl_model, pl_data, return_predictions=False)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_panoptic", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3Dv2", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
