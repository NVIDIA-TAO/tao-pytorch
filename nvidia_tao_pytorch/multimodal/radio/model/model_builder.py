# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO model builder."""

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import interpolate_patch_embed, interpolate_pos_embed
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights
from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import DINOV2
from nvidia_tao_pytorch.multimodal.radio.model.utils import cls_parser, ptm_adapter


def build_model(experiment_config,
                export=False):
    """Build backbone model according to configuration.

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    freeze_norm = model_config.backbone['freeze_norm']
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    try:
        model = BACKBONE_REGISTRY.get(backbone)(
            num_classes=dataset_config.num_classes,
            freeze_at='all' if freeze_backbone else None,
            freeze_norm=freeze_norm,
            pretrained_backbone_path=pretrained_backbone_path,
            export=export)
        # We should unfreeze the head for training, even `freeze_backbone` is set to `True`.
        if freeze_backbone:
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
            head.train()
    except KeyError as e:
        logger.error(f"Error building model: {e}")
        logger.warning(f"BACKBONE_REGISTRY: {BACKBONE_REGISTRY}")
        raise e

    # Backbones that load their own pretrained weights inside the factory
    # (e.g. SAM3) opt out of the generic post-construction state-dict load by
    # setting ``_consumes_pretrained_path = True`` on the wrapper class.
    consumed_by_factory = getattr(model, "_consumes_pretrained_path", False)

    # Load pretrained backbone
    if pretrained_backbone_path and not consumed_by_factory:
        logger.info(f"Loading pretrained weights from {pretrained_backbone_path}")
        state_dict = load_pretrained_weights(pretrained_backbone_path, parser=cls_parser, ptm_adapter=ptm_adapter)
        if isinstance(model, DINOV2):
            state_dict = interpolate_vit_checkpoint(
                checkpoint=state_dict,
                target_patch_size=14,
                target_resolution=518,
            )
        msg = model.load_state_dict(state_dict, strict=False)
        if get_global_rank() == 0:
            logger.info(f"Loaded pretrained weights from {pretrained_backbone_path}")
            logger.info(f"{msg}")

    return model


def interpolate_vit_checkpoint(checkpoint, target_patch_size, target_resolution):
    """ Interpolate ViT backbone position embedding and patch embedding

    Args:
        checkpoint: pretrained ViT checkpoint
        target_patch_size: target patch size to interpolate to. ex: 14, 16, etc
        target_resolution: target image size to interpolate to. ex: 224, 512, 518, etc

    Returns:
        interpolated model checkpoints

    """
    if checkpoint is None:
        return checkpoint

    if get_global_rank() == 0:
        logger.info("Do ViT pretrained backbone interpolation")
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)
    checkpoint = interpolate_pos_embed(
        checkpoint_model=checkpoint, new_resolution=target_resolution, new_patch_size=target_patch_size
    )
    return checkpoint
