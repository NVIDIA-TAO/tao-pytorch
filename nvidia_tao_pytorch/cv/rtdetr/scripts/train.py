# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train RT-DETR model."""
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.rtdetr.default_config import ExperimentConfig

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
from nvidia_tao_pytorch.cv.rtdetr.model.utils import rtdetr_parser, ptm_adapter


def run_experiment(experiment_config, lightning_module=RTDETRPlModel):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config)

    dm = ODDataModule(experiment_config.dataset)
    dm.setup(stage="fit")

    # find_unuser_parameters=False and activation_checkpoint combination
    # requires every output in forward function to participate in
    # loss calculation. When return_interm_indices < 4, we must disable
    # activation checkpointing
    if experiment_config.train.activation_checkpoint and \
        len(experiment_config.model.return_interm_indices) < 4 and \
            experiment_config.train.num_gpus > 1:
        experiment_config.train.activation_checkpoint = False
        logging.info("Disabling  activation checkpointing since model is smaller")

    activation_checkpoint = experiment_config.train.activation_checkpoint

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path:
        # Ignore backbone weights if we get pretrained path for the entire detector
        experiment_config.model.pretrained_backbone_path = None
        pt_model = lightning_module(experiment_config)
        current_model_dict = pt_model.model.state_dict()
        checkpoint = load_pretrained_weights(
            pretrained_path,
            parser=rtdetr_parser,
            ptm_adapter=ptm_adapter
        )
        new_checkpoint = {}
        for k in sorted(checkpoint.keys()):
            v = checkpoint[k]
            # Handle PTL format
            if current_model_dict.get(k, None) is None:
                continue
            if v.size() == current_model_dict[k].size():
                new_checkpoint[k] = v
            else:
                # Skip layers that mismatch
                logging.info(
                    "skip layer: %s, checkpoint layer size: %s, current model layer size: %s",
                    k, list(v.size()), list(current_model_dict[k].size())
                )
                new_checkpoint[k] = current_model_dict[k]
        # Load pretrained weights
        msg = pt_model.model.load_state_dict(new_checkpoint, strict=False)
        if get_global_rank() == 0:
            logging.info(f"Loaded pretrained weights from {pretrained_path}")
            logging.info(f"{msg}")
    else:
        pt_model = lightning_module(experiment_config)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. \
                                  Only bf16, fp16, and fp32 are supported")

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        # By default find_unused_parameters is set to False in Lightning
        # If true, it introduces extra overhead and can't work with activation checkpointing
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        elif distributed_strategy.lower() == "ddp" and not activation_checkpoint:
            strategy = 'ddp_find_unused_parameters_true'
        elif distributed_strategy.lower() == "fsdp":
            strategy = 'fsdp'
            # Override to FP16 for fsdp as there's an error with FP32 during Positional Embedding forward pass
            logging.info("Overriding Precision to FP16 for fsdp")
            precision = '16-mixed'
        else:
            raise NotImplementedError(f"{distributed_strategy} is not implemented. Only ddp and fsdp are supported")

    trainer = Trainer(**trainer_kwargs,
                      num_nodes=num_nodes,
                      strategy=strategy,
                      precision=precision,
                      gradient_clip_val=clip_grad_norm,
                      use_distributed_sampler=False,
                      sync_batchnorm=len(trainer_kwargs['devices']) > 1,
                      fast_dev_run=is_dry_run)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="RT-DETR", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   lightning_module=RTDETRPlModel)


if __name__ == "__main__":
    main()
