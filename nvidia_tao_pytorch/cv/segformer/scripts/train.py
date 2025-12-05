# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Train Segformer model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_core.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def run_experiment(experiment_config, key):
    """Start the training."""
    # results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)
    if experiment_config.get("train", {}).get("resume_training_checkpoint_path", None) == "":
        experiment_config["train"]["resume_training_checkpoint_path"] = None

    if experiment_config.get("train", {}).get("pretrained_model_path", None) == "":
        experiment_config["train"]["pretrained_model_path"] = None

    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes
    enable_tensorboard = experiment_config.train.tensorboard.enabled

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = '32-true'

    assert enable_tensorboard is False, "Currently tensorboard visualization is not supported for Segmentation"

    dm = SFDataModule(experiment_config.dataset.segment)

    if pretrained_path:
        model = SegFormerPlModel.load_from_checkpoint(
            pretrained_path,
            map_location="cpu",
            experiment_spec=experiment_config
        )
    else:
        model = SegFormerPlModel(experiment_config)

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        strategy = 'ddp_find_unused_parameters_true'

    trainer_kwargs.pop("enable_checkpointing", None)
    trainer_kwargs["enable_checkpointing"] = True

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=True,  # SegFormer head has BatchNorm.
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="SegFormer", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(
        experiment_config=cfg,
        key=cfg.encryption_key
    )


if __name__ == "__main__":
    main()
