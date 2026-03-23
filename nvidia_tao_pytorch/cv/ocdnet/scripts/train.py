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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Train OCDnet model."""
import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_core.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocdnet.data_loader.pl_ocd_data_module import OCDDataModule
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel


def run_experiment(experiment_config):
    """Start the training."""
    experiment_config = OmegaConf.to_container(experiment_config)
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config)
    dm = OCDDataModule(experiment_config)
    dm.setup(stage='fit')
    ocd_model = OCDnetModel(experiment_config, dm, 'fit')

    assert trainer_kwargs['max_epochs'] != experiment_config['train']['lr_scheduler']['args']['warmup_epoch'], "num_epochs should not be same as warmup_epoch."
    clip_grad = experiment_config['train']['trainer']['clip_grad_norm']
    activation_checkpoint = experiment_config['model']['activation_checkpoint']
    distributed_strategy = experiment_config['train']['distributed_strategy']
    is_dry_run = experiment_config['train']['is_dry_run']

    if experiment_config['train']['precision'].lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config['train']['precision'].lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config['train']['precision'].lower()} is not supported. Only fp32 and fp16 are supported")

    sync_batchnorm = False
    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        # By default find_unused_parameters is set to False in Lightning
        # This introduces extra overhead and can't work with activation checkpointing
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            # Override to FP16 for fsdp as there's an error with FP32 during Positional Embedding forward pass
            print("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        elif distributed_strategy.lower() == "deepspeed_stage_3_offload":
            strategy = 'deepspeed_stage_3_offload'
            print("Overriding Precision to FP16 for deepspeed_stage_3_offload")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp , fsdp and deepspeed are supported")

        if "fan" in experiment_config['model']['backbone']:
            print("Setting sync batch norm")
            sync_batchnorm = True

    use_distributed_sampler = experiment_config['train']['use_distributed_sampler']

    trainer = Trainer(**trainer_kwargs,
                      strategy=strategy,
                      gradient_clip_val=clip_grad,
                      num_sanity_val_steps=0,
                      precision=precision,
                      use_distributed_sampler=use_distributed_sampler,
                      sync_batchnorm=sync_batchnorm,
                      fast_dev_run=is_dry_run)

    trainer.fit(ocd_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="OCDNet", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
