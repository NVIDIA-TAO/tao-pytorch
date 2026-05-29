# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Pretrained backbone builder. """

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as zoomodels
import timm

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks import clip
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.vit import _make_vit_b16_backbone, forward_vit
from nvidia_tao_pytorch.sdg.stylegan_xl.model.feature_networks.constants import ALL_MODELS, VITS, EFFNETS, REGNETS


def _feature_splitter(model, idcs):
    """Splits the model features into layers based on provided indices.

    Args:
        model (nn.Module): The model to split.
        idcs (list): List of indices to split the features.

    Returns:
        nn.Module: Pretrained model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.features[:idcs[0]])
    pretrained.layer1 = nn.Sequential(model.features[idcs[0]:idcs[1]])
    pretrained.layer2 = nn.Sequential(model.features[idcs[1]:idcs[2]])
    pretrained.layer3 = nn.Sequential(model.features[idcs[2]:idcs[3]])
    return pretrained


def _make_resnet(model):
    """Creates a pretrained ResNet model with split layers.

    Args:
        model (nn.Module): ResNet model.

    Returns:
        nn.Module: Pretrained ResNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, model.layer1,
    )
    pretrained.layer1 = model.layer2
    pretrained.layer2 = model.layer3
    pretrained.layer3 = model.layer4
    return pretrained


def _make_regnet(model):
    """Creates a pretrained RegNet model with split layers.

    Args:
        model (nn.Module): RegNet model.

    Returns:
        nn.Module: Pretrained RegNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.stem, model.s1
    )
    pretrained.layer1 = model.s2
    pretrained.layer2 = model.s3
    pretrained.layer3 = model.s4
    return pretrained


def _make_nfnet(model):
    """Creates a pretrained NFNet model with split layers.

    Args:
        model (nn.Module): NFNet model.

    Returns:
        nn.Module: Pretrained NFNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.stem, model.stages[0]
    )
    pretrained.layer1 = model.stages[1]
    pretrained.layer2 = model.stages[2]
    pretrained.layer3 = model.stages[3]
    return pretrained


def _make_resnet_v2(model):
    """Creates a pretrained ResNet V2 model with split layers.

    Args:
        model (nn.Module): ResNet V2 model.

    Returns:
        nn.Module: Pretrained ResNet V2 model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.stem, model.stages[0])
    pretrained.layer1 = model.stages[1]
    pretrained.layer2 = model.stages[2]
    pretrained.layer3 = model.stages[3]
    return pretrained


def _make_resnet_clip(model):
    """Creates a pretrained ResNet model for CLIP with split layers.

    Args:
        model (nn.Module): ResNet model for CLIP.

    Returns:
        nn.Module: Pretrained ResNet model for CLIP with split layers.
    """
    pretrained = nn.Module()

    # slightly more complicated than the standard resnet
    pretrained.layer0 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.conv2,
        model.bn2,
        model.relu,
        model.conv3,
        model.bn3,
        model.relu,
        model.avgpool,
        model.layer1,
    )

    pretrained.layer1 = model.layer2
    pretrained.layer2 = model.layer3
    pretrained.layer3 = model.layer4

    return pretrained


def _make_densenet(model):
    """Creates a pretrained DenseNet model with split layers.

    Args:
        model (nn.Module): DenseNet model.

    Returns:
        nn.Module: Pretrained DenseNet model with split layers.
    """
    pretrained = nn.Module()

    pretrained.layer0 = model.features[:6]

    pretrained.layer1 = model.features[6:8]
    pretrained.layer1[-1][-1] = nn.Identity()
    pretrained.layer1 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer1)

    pretrained.layer2 = model.features[8:10]
    pretrained.layer2[-1][-1] = nn.Identity()
    pretrained.layer2 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer2)

    pretrained.layer3 = model.features[10:12]
    pretrained.layer3 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer3)

    return pretrained


def _make_shufflenet(model):
    """Creates a pretrained ShuffleNet model with split layers.

    Args:
        model (nn.Module): ShuffleNet model.

    Returns:
        nn.Module: Pretrained ShuffleNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv1, model.maxpool)
    pretrained.layer1 = model.stage2
    pretrained.layer2 = model.stage3
    pretrained.layer3 = model.stage4
    return pretrained


def _make_cspresnet(model):
    """Creates a pretrained CSPResNet model with split layers.

    Args:
        model (nn.Module): CSPResNet model.

    Returns:
        nn.Module: Pretrained CSPResNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.stem, model.stages[0])
    pretrained.layer1 = model.stages[1]
    pretrained.layer2 = model.stages[2]
    pretrained.layer3 = model.stages[3]
    return pretrained


def _make_efficientnet(model):
    """Creates a pretrained EfficientNet model with split layers.

    Args:
        model (nn.Module): EfficientNet model.

    Returns:
        nn.Module: Pretrained EfficientNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, *model.blocks[0:2]
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def _make_ghostnet(model):
    """Creates a pretrained GhostNet model with split layers.

    Args:
        model (nn.Module): GhostNet model.

    Returns:
        nn.Module: Pretrained GhostNet model with split layers.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, model.act1, *model.blocks[0:3],
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer2 = nn.Sequential(*model.blocks[5:7])
    pretrained.layer3 = nn.Sequential(*model.blocks[7:-1])
    return pretrained


def _make_vit(model, name):
    """Creates a pretrained Vision Transformer model with split layers.

    Args:
        model (nn.Module): Vision Transformer model.
        name (str): Name of the Vision Transformer model.

    Returns:
        nn.Module: Pretrained Vision Transformer model with split layers.
    """
    if 'tiny' in name:
        features = [24, 48, 96, 192]
        hooks = [2, 5, 8, 11]
        vit_features = 192

    elif 'small' in name:
        features = [48, 96, 192, 384]
        hooks = [2, 5, 8, 11]
        vit_features = 384

    elif 'base' in name:
        features = [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 768

    elif 'large' in name:
        features = [256, 512, 1024, 1024]
        hooks = [5, 11, 17, 23]
        vit_features = 1024

    else:
        raise NotImplementedError('Invalid ViT backbone not available')

    return _make_vit_b16_backbone(
        model,
        features=features,
        size=[224, 224],
        hooks=hooks,
        vit_features=vit_features,
        start_index=2 if 'deit' in name else 1,
    )


def calc_dims(pretrained, is_vit=False):
    """Calculates the dimensions of the pretrained model.

    Args:
        pretrained (nn.Module): Pretrained model.
        is_vit (bool, optional): Whether the model is a Vision Transformer. Default is False.

    Returns:
        tuple: Channels and resolution multiplier.
    """
    dims = []
    inp_res = 256
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    if not is_vit:
        tmp = pretrained.layer0(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer1(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer2(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer3(tmp)
        dims.append(tmp.shape[1:3])
    else:
        tmp = forward_vit(pretrained, tmp)
        dims = [out.shape[1:3] for out in tmp]

    # split to channels and resolution multiplier
    dims = np.array(dims)
    channels = dims[:, 0]
    res_mult = dims[:, 1] / inp_res
    return channels, res_mult


def _make_pretrained(backbone, verbose=False):
    """Creates a pretrained model with specified backbone.

    Args:
        backbone (str): Name of the backbone model.
        verbose (bool, optional): Whether to print detailed information. Default is False.

    Returns:
        nn.Module: Pretrained model with split layers.
    """
    assert backbone in ALL_MODELS

    if backbone == 'vgg11_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [7, 14, 21, 28]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'vgg13_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 20, 27, 34]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'vgg16_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 23, 33, 43]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'vgg19_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 26, 39, 52]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'densenet121':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)

    elif backbone == 'densenet169':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)

    elif backbone == 'densenet201':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)

    elif backbone == 'resnet18':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'resnet34':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'resnet50':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'resnet101':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'resnet152':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'wide_resnet50_2':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'wide_resnet101_2':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)

    elif backbone == 'shufflenet_v2_x0_5':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_shufflenet(model)

    elif backbone == 'mobilenet_v2':
        model = zoomodels.__dict__[backbone](True)
        idcs = [4, 7, 14, 18]
        pretrained = _feature_splitter(model, idcs)  # same structure as vgg

    elif backbone == 'mnasnet0_5':
        model = zoomodels.__dict__[backbone](True)
        model.features = model.layers
        idcs = [9, 10, 12, 14]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'mnasnet1_0':
        model = zoomodels.__dict__[backbone](True)
        model.features = model.layers
        idcs = [9, 10, 12, 14]
        pretrained = _feature_splitter(model, idcs)

    elif backbone == 'ghostnet_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_ghostnet(model)

    elif backbone == 'cspresnet50':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'fbnetc_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone == 'spnasnet_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone == 'resnet50d':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'resnet26':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'resnet26d':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'seresnet50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'resnetblur50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'resnetrs50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'tf_mixnet_s':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone == 'tf_mixnet_m':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone == 'tf_mixnet_l':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone == 'dm_nfnet_f0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'dm_nfnet_f1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'ese_vovnet19b_dw':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'ese_vovnet39b':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'res2next50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)

    elif backbone == 'gernet_s':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'gernet_m':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'repvgg_a2':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'repvgg_b0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'repvgg_b1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'repvgg_b1g4':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)

    elif backbone == 'dm_nfnet_f1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_nfnet(model)

    elif backbone == 'nfnet_l0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_nfnet(model)

    elif backbone in REGNETS:
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_regnet(model)

    elif backbone in EFFNETS:
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)

    elif backbone in VITS:
        model = timm.create_model(backbone, pretrained=True, dynamic_img_size=True)  # The timm package has been udapted and now supports `dynamic_img_size` for ViT.
        pretrained = _make_vit(model, backbone)

    elif backbone == 'resnet50_clip':
        model = clip.load('RN50', device='cpu', jit=False)[0].visual
        pretrained = _make_resnet_clip(model)

    else:
        raise NotImplementedError('Wrong model name?')

    pretrained.CHANNELS, pretrained.RES_MULT = calc_dims(pretrained, is_vit=backbone in VITS)

    if verbose:
        logging.info(f"Succesfully loaded:    {backbone}")
        logging.info(f"Channels:              {pretrained.CHANNELS}")
        logging.info(f"Resolution Multiplier: {pretrained.RES_MULT}")
        logging.info(f"Out Res for 256      : {pretrained.RES_MULT * 256}")

    return pretrained
