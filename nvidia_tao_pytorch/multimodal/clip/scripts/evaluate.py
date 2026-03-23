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

from nvidia_tao_core.config.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.dataloader.pl_clip_data_module import (
    CLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    load_model_from_checkpoint,
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

    # Validate that retrieval evaluation is configured
    val_cfg = getattr(experiment_config.dataset, 'val', None)
    if val_cfg is None or not getattr(val_cfg, 'datasets', None):
        raise ValueError(
            "No evaluation data configured. For evaluate task, you must provide:\n"
            "  dataset.val.datasets:\n"
            "  - image_dir: /path/to/images\n"
            "    caption_dir: /path/to/captions"
        )

    logging.info(f"Retrieval evaluation: {len(val_cfg.datasets)} dataset(s)")
    for i, ds in enumerate(val_cfg.datasets):
        logging.info(f"  Dataset {i + 1}: images={ds.image_dir}, captions={ds.caption_dir}")

    model_path, trainer_kwargs = initialize_evaluation_experiment(
        experiment_config, experiment_config.encryption_key
    )

    if model_path:
        logging.info(f"Loading model from {model_path}")
        model = load_model_from_checkpoint(
            model_path, experiment_config, CLIPPlModel)
    else:
        logging.info(
            f"No checkpoint provided. Building model from pretrained "
            f"weights: {experiment_config.model.type}"
        )
        model = CLIPPlModel(experiment_config)

    dm = CLIPDataModule(
        experiment_config.dataset,
        model.tokenizer,
        resume_step=0,
        preprocess=(model.preprocess_train, model.preprocess_val),
        world_size=1
    )
    dm.setup(stage="test")

    logging.info("Starting retrieval evaluation")
    trainer = Trainer(**trainer_kwargs)
    trainer.test(model, datamodule=dm)

    logging.info("Evaluation finished")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
@monitor_status(name="CLIP", mode="evaluate")
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
