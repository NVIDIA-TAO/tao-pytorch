# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top model builder interface."""
from nvidia_tao_pytorch.cv.pose_classification.model.pc_model import get_basemodel


def build_pc_model(experiment_config,
                   export=False):
    """
    Build a pose classification model according to the provided configuration.

    This function uses the configuration provided in experiment_config to create a pose classification model.
    Currently, only the "ST-GCN" model type is supported. If a different model type is provided in the
    configuration, a NotImplementedError will be raised.

    Args:
        experiment_config (dict): A dictionary containing the configuration for the experiment.
                                  This should include specifications for the model, such as its type,
                                  the path to a pre-trained model if one is being used, the number of input channels,
                                  the number of classes in the dataset, dropout rate, and graph parameters
                                  like layout, strategy, and edge importance weighting.
        export (bool, optional): A flag that indicates whether the model is being built for export.
                                 This is currently not used in the function. Defaults to False.

    Returns:
        torch.nn.Module: The created pose classification model.

    Raises:
        NotImplementedError: If a model type other than "ST-GCN" is specified in the configuration.
    """
    model_config = experiment_config["model"]
    model_type = model_config["model_type"]
    pretrained_model_path = model_config["pretrained_model_path"]
    input_channels = model_config["input_channels"]
    num_classes = experiment_config["dataset"]["num_classes"]
    dropout = model_config["dropout"]
    graph_layout = model_config["graph_layout"]
    graph_strategy = model_config["graph_strategy"]
    edge_importance_weighting = model_config["edge_importance_weighting"]

    if model_type == "ST-GCN":
        model = get_basemodel(pretrained_model_path=pretrained_model_path,
                              model_type=model_type,
                              input_channels=input_channels,
                              num_classes=num_classes,
                              graph_layout=graph_layout,
                              graph_strategy=graph_strategy,
                              edge_importance_weighting=edge_importance_weighting,
                              dropout=dropout)
    else:
        raise NotImplementedError("Only the type \"ST-GCN\" is supported")

    return model
