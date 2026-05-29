# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top model builder interface."""
import torch
from .ar_model import (get_basemodel, JointModel, JointModel_ONNX, get_basemodel3d,
                       load_pretrained_weights)


def build_ar_model(experiment_config,
                   imagenet_pretrained=True,
                   export=False):
    """
    Build action recognition model according to config.

    This function constructs an action recognition model according to the specified experiment configuration.

    Args:
        experiment_config (dict): The experiment configuration dictionary.
        imagenet_pretrained (bool, optional): Whether to use imagenet pretrained weights. Defaults to True.
        export (bool, optional): Whether to build the model that can be exported to ONNX format. Defaults to False.

    Returns:
        nn.Module: The action recognition model.
    """
    model_config = experiment_config["model"]
    dataset_config = experiment_config["dataset"]
    nb_classes = len(dataset_config["label_map"].keys())
    model_type = model_config["model_type"]
    input_type = model_config["input_type"]
    backbone = model_config["backbone"]
    imagenet_pretrained = model_config["imagenet_pretrained"]
    dropout = model_config["dropout_ratio"]

    if input_type == "2d":
        if model_type == "of":
            model = get_basemodel(backbone=backbone,
                                  input_channel=model_config['of_seq_length'] * 2,
                                  nb_classes=nb_classes,
                                  imagenet_pretrained=imagenet_pretrained,
                                  pretrained_backbone_path=model_config["of_pretrained_model_path"],
                                  pretrained_class_num=model_config["of_pretrained_num_classes"],
                                  dropout_ratio=dropout)
        elif model_type == "rgb":
            model = get_basemodel(backbone=backbone,
                                  input_channel=model_config['rgb_seq_length'] * 3,
                                  nb_classes=nb_classes,
                                  imagenet_pretrained=imagenet_pretrained,
                                  pretrained_backbone_path=model_config["rgb_pretrained_model_path"],
                                  pretrained_class_num=model_config["rgb_pretrained_num_classes"],
                                  dropout_ratio=dropout)
        elif model_type == "joint":
            if export:
                model = JointModel_ONNX(backbone=backbone,
                                        input_type="2d",
                                        of_seq_length=model_config['of_seq_length'],
                                        rgb_seq_length=model_config['rgb_seq_length'],
                                        nb_classes=nb_classes,
                                        num_fc=model_config['num_fc'],
                                        pretrain_of_model=model_config["of_pretrained_model_path"],
                                        pretrain_rgb_model=model_config["rgb_pretrained_model_path"],
                                        imagenet_pretrained=imagenet_pretrained,
                                        dropout_ratio=dropout)
            else:
                model = JointModel(backbone=backbone,
                                   input_type="2d",
                                   of_seq_length=model_config['of_seq_length'],
                                   rgb_seq_length=model_config['rgb_seq_length'],
                                   nb_classes=nb_classes,
                                   num_fc=model_config['num_fc'],
                                   pretrain_of_model=model_config["of_pretrained_model_path"],
                                   pretrain_rgb_model=model_config["rgb_pretrained_model_path"],
                                   imagenet_pretrained=imagenet_pretrained,
                                   dropout_ratio=dropout)

                if model_config["joint_pretrained_model_path"]:
                    temp = torch.load(model_config["joint_pretrained_model_path"])
                    model.load_state_dict(temp["state_dict"])
        else:
            raise ValueError("Only the type in [of, rgb, joint] is supported")
    elif input_type == "3d":
        if model_type in ("of", "rgb"):
            if model_type == "of":
                pretrained_backbone_path = model_config["of_pretrained_model_path"]
                pretrained_class_num = model_config["of_pretrained_num_classes"]
            elif model_type == "rgb":
                pretrained_backbone_path = model_config["rgb_pretrained_model_path"]
                pretrained_class_num = model_config["rgb_pretrained_num_classes"]

            model = get_basemodel3d(backbone=backbone,
                                    nb_classes=nb_classes,
                                    modality=model_type,
                                    pretrained_backbone_path=pretrained_backbone_path,
                                    pretrained_class_num=pretrained_class_num,
                                    imagenet_pretrained=imagenet_pretrained,
                                    dropout_ratio=dropout)
        elif model_type == "joint":
            if export:
                model = JointModel_ONNX(backbone=backbone,
                                        of_seq_length=model_config['of_seq_length'],
                                        rgb_seq_length=model_config['rgb_seq_length'],
                                        nb_classes=nb_classes,
                                        num_fc=model_config['num_fc'],
                                        pretrain_of_model=model_config["of_pretrained_model_path"],
                                        pretrain_rgb_model=model_config["rgb_pretrained_model_path"],
                                        imagenet_pretrained=imagenet_pretrained,
                                        input_type=input_type,
                                        dropout_ratio=dropout)
            else:
                model = JointModel(backbone=backbone,
                                   of_seq_length=model_config['of_seq_length'],
                                   rgb_seq_length=model_config['rgb_seq_length'],
                                   nb_classes=nb_classes,
                                   num_fc=model_config['num_fc'],
                                   pretrain_of_model=model_config["of_pretrained_model_path"],
                                   pretrain_rgb_model=model_config["rgb_pretrained_model_path"],
                                   imagenet_pretrained=imagenet_pretrained,
                                   input_type=input_type,
                                   dropout_ratio=dropout)

                if model_config["joint_pretrained_model_path"]:
                    pretrained_weights = \
                        load_pretrained_weights(model_config["joint_pretrained_model_path"])
                    model.load_state_dict(pretrained_weights)
        else:
            raise ValueError("Only the type in [of, rgb, joint] is supported")

    return model
