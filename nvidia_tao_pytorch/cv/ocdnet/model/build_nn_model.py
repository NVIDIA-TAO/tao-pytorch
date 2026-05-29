# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top model builder interface."""

from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model


def build_ocd_model(experiment_config,
                    export=False):
    """Build ocdnet model according to config."""
    model_config = experiment_config["model"]
    model_config['activation_checkpoint'] = False if experiment_config["train"]['model_ema'] else model_config['activation_checkpoint']
    load_pruned_graph = model_config['load_pruned_graph']

    if load_pruned_graph:
        assert model_config['pruned_graph_path'], (
            "The load_pruned_graph is set to True. But the pruned_graph_path is not available. "
            "Please set the pruned_graph_path in the spec file."
            "If you are resuming training, please set resume_training_checkpoint_path as well.")
        pruned_graph_path = model_config['pruned_graph_path']
        model = load_checkpoint(pruned_graph_path, only_state_dict=False)
        print(f'loading pruned model from {pruned_graph_path}')
    else:
        model_config['pruned_graph_path'] = None
        model = Model(model_config)

        # Load pretrained weights or resume model
        if experiment_config['train']['resume_training_checkpoint_path']:
            assert (experiment_config['train']['resume_training_checkpoint_path']).endswith(".pth"), (
                "Will resume training. Please set the file path in 'resume_training_checkpoint_path' for resuming training."
                " If not resume training, please set resume_training_checkpoint_path:None")
            finetune = False
        elif model_config['pretrained_model_path']:
            model_path = model_config['pretrained_model_path']
            print(f'loading pretrained model from {model_path}')
            finetune = True
        else:
            finetune = False

        if finetune:
            state_dict = load_checkpoint(model_path, to_cpu=True)
            model.load_state_dict(state_dict, strict=False)

    # Default to training mode
    model.train()
    return model
