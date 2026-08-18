# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate CLIP model using retrieval metrics."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import (
    initialize_evaluation_experiment,
)
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.pl_video_clip_model import VideoCLIPPlModel
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.pl_video_clip_data_module import (
    VideoCLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    load_model_from_checkpoint,
    to_lightning_precision,
)


def run_experiment(experiment_config, key):
    """Run retrieval evaluation experiment.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration object containing dataset, model,
        and evaluation settings.
    key : str
        Encryption key (unused, kept for TAO API compatibility).

    Raises
    ------
    ValueError
        If no evaluation data is configured (missing captions_dir).
    """
    del key  # Unused but required by TAO API

    # Validate that retrieval evaluation is configured (video_text only).
    val_cfg = getattr(experiment_config.dataset, 'val', None)
    video_text_cfg = getattr(val_cfg, 'video_text', None) if val_cfg else None
    if not (video_text_cfg is not None and bool(getattr(video_text_cfg, 'metadata', None))):
        raise ValueError(
            "No evaluation data configured. For the evaluate task set:\n"
            "  dataset.val.video_text.metadata: /path/to/metadata.json\n"
            "Optionally add dataset.val.video_text.relevance_file (e.g. a frozen "
            "domain_test.json) for per-slice explicit-relevance retrieval."
        )

    relevance_file = getattr(video_text_cfg, 'relevance_file', None)
    if relevance_file:
        logging.info(
            "Per-slice explicit-relevance evaluation: gallery=%s, queries=%s",
            video_text_cfg.metadata, relevance_file,
        )
    else:
        logging.info(f"Video-text retrieval evaluation: {video_text_cfg.metadata}")

    model_path, trainer_kwargs = initialize_evaluation_experiment(
        experiment_config, experiment_config.encryption_key
    )

    logging.info(f"Loading model from {model_path}")
    model = load_model_from_checkpoint(
        model_path, experiment_config, VideoCLIPPlModel)

    dm = VideoCLIPDataModule(
        experiment_config.dataset,
        model.tokenizer,
        resume_step=0,
        preprocess=(model.preprocess_train, model.preprocess_val),
        world_size=1
    )
    dm.setup(stage="test")

    logging.info("Starting retrieval evaluation")
    trainer_kwargs["precision"] = to_lightning_precision(
        experiment_config.train.precision
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.test(model, datamodule=dm)

    logging.info("Evaluation finished")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
@monitor_status(name="VideoCLIP", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Hydra configuration object populated from experiment spec.
    """
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
