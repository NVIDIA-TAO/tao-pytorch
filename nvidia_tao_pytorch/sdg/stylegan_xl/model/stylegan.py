# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StyleGAN model builder"""

import copy
import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.sdg.stylegan_xl.model.generator.networks_styleganxl import Generator, SuperresGenerator
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.discriminator import ProjectedDiscriminator as Discriminator
from nvidia_tao_pytorch.sdg.stylegan_xl.model.losses.loss import ProjectedGANLoss
from nvidia_tao_pytorch.sdg.stylegan_xl.model.metrics.inception_net import InceptionV3
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib


class IntermediateInception(torch.nn.Module):
    """ InceptionNet for extracting intermediate features."""

    def __init__(self, inception_checkpoint=None):
        """
        Initialization.

        Args:
            inception_checkpoint (str): InceptionNet checkpoint.
        """
        super().__init__()
        self.inception = InceptionV3()

        for _, param in self.inception.named_parameters():
            param.requires_grad = False

    def load_pretrained_model(self, inception_checkpoint):
        """Internal function for loading pretrained weights of InceptionNet."""
        model_path = inception_checkpoint
        # self.inception_checkpoint = '/tao-pt/nvidia_tao_pytorch/sdg/stylegan_xl/pretrained_modules/InceptionV3.pth'
        if (model_path is not None):
            self.inception.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu'))  # avoid GPU memory leak and not released
            )
        else:
            logging.warning("The pretrained InceptionNet checkpoint for calculating FID metrics is missing !!!")

    def forward(self, x):
        """
        Forward pass for extracting intermediate features.
        """
        features = self.inception(x, return_features=True)

        return features


def build_generator(experiment_config, return_stem=False, export=False):
    """ Build StyleGAN-XL generator according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export # TODO

    Returns:
        G: generator

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    G_kwargs = get_generator_kwargs(experiment_config)
    common_kwargs = dict(c_dim=dataset_config['common']['num_classes'],
                         img_channels=dataset_config['common']['img_channels'])

    if model_config['generator']['superres']:
        # SuperresGenerator
        head_layers_list = model_config['generator']['added_head_superres']['head_layers']
        up_factor_list = model_config['generator']['added_head_superres']['up_factor']
        final_resolution = model_config['generator']['stem']['resolution']
        for up_factor in up_factor_list:
            final_resolution *= up_factor
        assert dataset_config['common']['img_resolution'] == final_resolution
    else:
        assert dataset_config['common']['img_resolution'] == model_config['generator']['stem']['resolution']

    if model_config['generator']['superres'] is not True:
        G = Generator(**G_kwargs,
                      **common_kwargs,
                      img_resolution=model_config['generator']['stem']['resolution'],
                      )
    else:
        G_stem = Generator(**G_kwargs,
                           **common_kwargs,
                           img_resolution=model_config['generator']['stem']['resolution'],
                           )

        resolution_per_stage = model_config['generator']['stem']['resolution']
        for i, (up_factor, head_layers) in enumerate(zip(up_factor_list, head_layers_list)):
            G_stem = G if i > 0 else G_stem
            resolution_per_stage *= up_factor
            G = SuperresGenerator(copy.deepcopy(G_stem),
                                  up_factor=up_factor,
                                  head_layers=head_layers,
                                  )

    if model_config['generator']['superres'] is not True:
        logging.info(f"constructed base stem generator with res={model_config['generator']['stem']['resolution']}")
        return G
    else:
        if (return_stem):
            logging.info(f"returned stem generator with res={resolution_per_stage / up_factor}")
            return G_stem
        else:
            logging.info(f"constructed superres generator with res={resolution_per_stage}")
            return G


def get_generator_kwargs(experiment_config):
    """ Build StyleGAN-XL generator kwargs according to configuration

    Args:
        experiment_config: experiment configuration

    Returns:
        G_kwargs: generator kwargs

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    # check inconsistence of conditional training
    if (dataset_config['common']['cond'] is False and dataset_config['common']['num_classes'] != 0):
        raise AssertionError

    # Construct networks.
    if (model_config['generator']['backbone'] in ["stylegan3-r", "stylegan3-t"]):
        G_kwargs = dnnlib.EasyDict(z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
        G_kwargs.channel_base = model_config['generator']['stem']['cbase']
        G_kwargs.channel_max = model_config['generator']['stem']['cmax']
        G_kwargs.magnitude_ema_beta = 0.5 ** (dataset_config['batch_size'] / (20 * 1e3))
        G_kwargs.channel_base *= 2  # increase for StyleGAN-XL
        G_kwargs.channel_max *= 2   # increase for StyleGAN-XL

        if model_config['generator']['backbone'] == 'stylegan3-r':
            G_kwargs.channel_base *= 2
            G_kwargs.channel_max *= 2
            G_kwargs.conv_kernel = 1
            G_kwargs.use_radial_filters = True
        else:
            G_kwargs.conv_kernel = 3
            G_kwargs.use_radial_filters = False

        if (model_config['generator']['stem']['fp32']):
            G_kwargs.num_fp16_res = 0
            G_kwargs.conv_clamp = None

        G_kwargs.embed_path = model_config['input_embeddings_path']

        ##################################  # noqa: E266
        ########## StyleGAN-XL ###########  # noqa: E266
        ##################################  # noqa: E266
        G_kwargs.w_dim = 512
        G_kwargs.z_dim = 64
        G_kwargs.mapping_kwargs.rand_embedding = False
        G_kwargs.num_layers = model_config['generator']['stem']['syn_layers']
        G_kwargs.mapping_kwargs.num_layers = 2

    else:
        raise NotImplementedError(f"model_config['generator']['backbone'] unavailable for {model_config['generator']['backbone']}. Please choose backbone among: ['stylegan3-r', 'stylegan3-t']")

    return G_kwargs


def build_discriminator(experiment_config):
    """ Build StyleGAN-XL discriminator according to configuration

    Args:
        experiment_config: experiment configuration

    Returns:
        D: discriminator

    """
    dataset_config = experiment_config.dataset
    D_kwargs = get_discriminator_kwargs(experiment_config)
    common_kwargs = dict(c_dim=dataset_config['common']['num_classes'],
                         img_channels=dataset_config['common']['img_channels'])

    D = Discriminator(**D_kwargs,
                      **common_kwargs,
                      img_resolution=dataset_config['common']['img_resolution']
                      )

    return D


def get_discriminator_kwargs(experiment_config):
    """ Build StyleGAN-XL discriminator kwargs according to configuration

    Args:
        experiment_config: experiment configuration

    Returns:
        D_kwargs: discriminator kwargs

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    # Discriminator
    D_kwargs = dnnlib.EasyDict(
        backbones=model_config['stylegan']['discriminator']['backbones'],
        diffaug=True,
        interp224=(dataset_config['common']['img_resolution'] < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )
    D_kwargs.backbone_kwargs.cout = 64
    D_kwargs.backbone_kwargs.expand = True
    D_kwargs.backbone_kwargs.proj_type = 2 if dataset_config['common']['img_resolution'] <= 16 else 2  # CCM only works better on very low resolutions
    D_kwargs.backbone_kwargs.num_discs = 4
    D_kwargs.backbone_kwargs.cond = dataset_config['common']['cond']

    D_kwargs.backbone_kwargs.embed_path = model_config['input_embeddings_path']

    return D_kwargs


def get_projectedloss_kwargs(experiment_config):
    """ Build StyleGAN-XL discriminator kwargs according to configuration

    Args:
        experiment_config: experiment configuration

    Returns:
        loss_kwargs: discriminator kwargs

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    loss_kwargs = dnnlib.EasyDict()
    loss_kwargs.blur_init_sigma = 2  # Blur the images seen by the discriminator.
    loss_kwargs.blur_fade_kimg = 300
    loss_kwargs.pl_weight = 2.0
    loss_kwargs.pl_no_weight_grad = True
    loss_kwargs.style_mixing_prob = 0.0
    loss_kwargs.cls_weight = 0.0  # use classifier guidance only for superresolution training (i.e., with pretrained stem)
    loss_kwargs.cls_model = 'deit_small_distilled_patch16_224'
    loss_kwargs.train_head_only = False
    if model_config['generator']['superres']:
        loss_kwargs.pl_weight = 0.0
        loss_kwargs.cls_weight = model_config['stylegan']['loss']['cls_weight'] if dataset_config['common']['cond'] else 0
        loss_kwargs.train_head_only = model_config['generator']['added_head_superres']['train_head_only']

    return loss_kwargs


def build_projectedloss(experiment_config):
    """ Build StyleGAN-XL projected loss according to configuration

    Args:
        experiment_config: experiment configuration

    Returns:
        projectedloss: projected loss

    """
    loss_kwargs = get_projectedloss_kwargs(experiment_config)
    projectedloss = ProjectedGANLoss(augment_pipe=None, **loss_kwargs)  # TODO  augment_pipe=None

    return projectedloss


def build_inception(experiment_config):
    """ Build StyleGAN-XL pretrained inception for calculating FID scores

    Args:
        experiment_config: experiment configuration

    Returns:
        inception: itermediate inception which can output intermediate features with forward function

    """
    inception = IntermediateInception().eval()

    return inception


def retrieve_generator_checkpoint_from_stylegan_pl_model(stylegan_pl_model_checkpoint_path):
    """ Retreive generator only checkpoint from the trained stylegan pytorch lightning model checkpoint

    Args:
        stylegan_pl_model_checkpoint_path: trained stylegan pytorch lightning model checkpoint path

    Returns:
        generator_ema_checkpoint: generator EMA checkpoint's state_dict

    """
    checkpoint = torch.load(stylegan_pl_model_checkpoint_path, map_location="cpu")['state_dict']
    filtered_checkpoint = {key: value for key, value in checkpoint.items() if key.startswith('G_ema')}
    replaced_checkpoint = {key.replace('G_ema.', ''): value for key, value in filtered_checkpoint.items()}
    generator_ema_checkpoint = replaced_checkpoint

    return generator_ema_checkpoint
