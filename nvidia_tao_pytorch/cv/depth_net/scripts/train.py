# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train depth network model."""
import os
import torch
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import TLTCheckpointConnector
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.cv.depth_net.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.depth_net.utils.misc import parse_mono_depth_checkpoint
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model, get_pl_module


def run_experiment(experiment_config, key):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    dm = build_pl_data_module(experiment_config.dataset)
    dm.setup(stage="fit")

    activation_checkpoint = experiment_config.train.activation_checkpoint

    # build model class.
    pt_model = build_pl_model(experiment_config)
    pretrained_path = experiment_config.train.pretrained_model_path

    # Load pretrained model as starting point if pretrained path is provided,
    if pretrained_path:
        # FFS commercial ckpt is a research-pickled nn.Module (not a PL ckpt
        # nor a plain state_dict). Route it through load_ffs_pretrained
        # which handles the pickle stub + prefix/substring remap and
        # reports missing/unexpected explicitly. Mirrors scripts/inference.py
        # and scripts/evaluate.py.
        if experiment_config.model.model_type == 'FastFoundationStereo':
            from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.ckpt_utils import (
                load_ffs_pretrained,
            )
            result = load_ffs_pretrained(pt_model.model, pretrained_path)
            assert not result['missing'], (
                f"FFS ckpt missing keys: {result['missing']}")
            assert not result['unexpected'], (
                f"FFS ckpt unexpected keys: {result['unexpected']}")
        else:
            model_dict = torch.load(pretrained_path, map_location="cpu")
            if "pytorch-lightning_version" not in model_dict and experiment_config.model.model_type in ['MetricDepthAnything', 'RelativeDepthAnything']:
                # parse public checkpoint
                modified_dict = parse_mono_depth_checkpoint(model_dict, experiment_config.model.model_type)
                pt_model.load_state_dict(modified_dict, strict=True)
            else:
                pt_model = get_pl_module(experiment_config).load_from_checkpoint(
                    pretrained_path,
                    map_location="cpu",
                    experiment_spec=experiment_config,
                    strict=True
                )

    print('model params', sum(p.numel() for p in pt_model.parameters()), flush=True)
    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm
    is_dry_run = experiment_config.train.is_dry_run
    distributed_strategy = experiment_config.train.distributed_strategy
    log_every_n_steps = experiment_config.train.log_every_n_steps

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32, fp16, and bf16 are supported")

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

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=clip_grad_norm,
        gradient_clip_algorithm="value",
        use_distributed_sampler=False,
        fast_dev_run=is_dry_run,
        log_every_n_steps=log_every_n_steps
    )
    # Overload connector to enable intermediate ckpt encryption & decryption.
    if resume_ckpt and resume_ckpt.endswith('.tlt'):
        trainer._checkpoint_connector = TLTCheckpointConnector(trainer)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="Depth Net", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
