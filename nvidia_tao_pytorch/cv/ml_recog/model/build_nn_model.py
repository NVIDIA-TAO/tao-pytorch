# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top model builder interface."""
import os
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.ml_recog.model.backbone import RecognitionBase, Trunk, Embedder


def check_and_load(model, checkpoint_path, config_name):
    """Check if the checkpoint path exists and load the checkpoint to the model.

    Args:
        model (torch.Module): the model to load checkpoint to.
        checkpoint_path (str): the path to the checkpoint file.
        config_name (str): the name of the config item that specifies the checkpoint path.

    Returns:
        model (torch.Module): the model with checkpoint loaded.
    """
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            error_mesage = f"{config_name} file does not exist."
            status_logging.get_status_logger().write(
                message=error_mesage,
                status_level=status_logging.Status.FAILURE)
            raise ValueError(error_mesage)
        status_logging.get_status_logger().write(
            message=f"Loading pretrained path: {checkpoint_path}.",
            status_level=status_logging.Status.RUNNING)
        model.load_checkpoint(checkpoint_path)

    return model


def build_model(cfg, checkpoint_to_load=False):
    """Builds metric learning recognition model according to config. If
    `checkpoint_to_load` is True, nothing would be returned as the model is already
    loaded somewhere else. If `checkpoint_to_load` is False, the function would
    do following things: first the model trunk and embedder would be initialized randomly.
    if `model.pretrain_choice` is `imagenet`, the pretrained weights from
    `Torch IMAGENET1K_V2` would be loaded to the trunk. If `model.pretrained_model_path`
    is specified, the pretrained weights from the weights file would be loaded to
    the trunk. If `model.pretrain_choice` is empty and `model.pretrained_model_path`
    is not specified, the trunk would keep its random weights. Notice that the
    embedder would not be loaded with pretrained weights in any case.

    In the end, the embedder and trunk would be combined to a Baseline model and
    the model would be returned.

    Args:
        cfg (DictConfig): Hydra config object.
        checkpoint_to_load (Bool): If True, a checkpoint would be loaded after building the model so the pretrained weights should not be loaded.

    Returns:
        model (torch.Module): the Baseline torch module.
    """
    status_logging.get_status_logger().write(
        message="Constructing model graph...",
        status_level=status_logging.Status.RUNNING)
    model_configs = cfg["model"]
    trunk_model = model_configs["backbone"]
    embed_dim = model_configs["feat_dim"]

    if trunk_model in ("resnet_50", "resnet_101"):
        # only supports linear head
        trunk = Trunk(trunk_model, train_backbone=cfg["train"]["train_trunk"])
        embedder = Embedder("linear_head", trunk.output_size, embed_dim,
                            train_embedder=cfg["train"]["train_embedder"])

    elif trunk_model in ("fan_base", "fan_large", "fan_small", "fan_tiny", "nvdinov2_vit_large_legacy"):
        # only supports dual head
        # both nvdinov2 and fan model has output size fixed
        img_size = (model_configs["input_width"], model_configs["input_height"])
        trunk = Trunk(trunk_model, train_backbone=cfg["train"]["train_trunk"],
                      img_size=img_size)
        embedder = Embedder("dual_head", trunk.output_size, embed_dim,
                            train_embedder=cfg["train"]["train_embedder"])
    else:
        raise NotImplementedError(f"Trunk model {trunk_model} is not supported.")

    if checkpoint_to_load:
        model = RecognitionBase(trunk, embedder)
        status_logging.get_status_logger().write(
            message="Skipped loading pretrained weights as checkpoint is to load.",
            status_level=status_logging.Status.SKIPPED)
    else:
        trunk = check_and_load(trunk, model_configs["pretrained_trunk_path"], "model.pretrained_trunk_path")
        embedder = check_and_load(embedder, model_configs["pretrained_embedder_path"], "model.pretrained_embedder_path")
        model = RecognitionBase(trunk, embedder)
        check_and_load(model, model_configs["pretrained_model_path"], "model.pretrained_model_path")

    return model
