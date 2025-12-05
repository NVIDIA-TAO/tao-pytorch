# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Training script for OneFormer unified segmentation model."""
from nvidia_tao_core.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import SemSegmDataModule
from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import OneformerPlModule
import os
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch

os.environ["PYTORCH_WARN_ONLY"] = "1"
torch.set_float32_matmul_precision("medium")


def run_experiment(cfg):
    """Run the training process."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(cfg)
    data_module = SemSegmDataModule(cfg)
    data_module.prepare_data()
    data_module.setup("fit")
    model = OneformerPlModule(cfg)

    if not resume_ckpt or not os.path.isfile(resume_ckpt):
        if cfg.train.pretrained_backbone:
            print(f"Loading backbone weights from {cfg.train.pretrained_backbone}")
            model.load_backbone_weights(cfg.train.pretrained_backbone)
        if cfg.train.pretrained_model:
            print(f"Loading pretrained weights from {cfg.train.pretrained_model}")
            model.load_pretrained_weights(cfg.train.pretrained_model)

    ddp_kwargs = {
        "find_unused_parameters": True,
        "gradient_as_bucket_view": True,
        "process_group_backend": "nccl",
    }

    strategy = DDPStrategy(**ddp_kwargs)

    trainer = pl.Trainer(
        num_nodes=cfg.train.num_nodes,
        strategy=strategy,
        precision=cfg.train.precision,
        sync_batchnorm=True,
        deterministic=cfg.train.seed,
        **trainer_kwargs
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_coco", schema=ExperimentConfig
)
@monitor_status(name="OneFormer", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
